/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/non_max_suppression_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/cuda_launch_config.h"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_segmented_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define NMS_BLOCK_DIM 16
#define NMS_BLOCK_DIM_MAX 16
#define NMS_CHUNK_SIZE 2000

#define CUDA_CHECK(result)                                    \
  do {                                                        \
    cudaError_t error(result);                                \
    CHECK(error == cudaSuccess) << cudaGetErrorString(error); \
  } while (0)

struct __align__(16) Box {
  float x1, y1, x2, y2;
};
namespace tensorflow {
const int NMS_BOXES_PER_THREAD = 8 * sizeof(int);

__launch_bounds__(NMS_BLOCK_DIM* NMS_BLOCK_DIM, 4) __global__
    void NMSKernel(const Box* d_desc_sorted_boxes, const int nboxes,
                   const float thresh, const int mask_ld, int* d_delete_mask) {
  // Storing boxes used by this CUDA block in the shared memory
  __shared__ Box shared_i_boxes[NMS_BLOCK_DIM];
  // Same thing with areas
  __shared__ float shared_i_areas[NMS_BLOCK_DIM];
  // The condition of the for loop is common to all threads in the block
  // This is necessary to be able to call __syncthreads() inside of the loop
  for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < nboxes;
       i_block_offset += blockDim.x * gridDim.x) {
    const int i_to_load = i_block_offset + threadIdx.x;
    if (i_to_load < nboxes) {
      // One 1D line load the boxes for x-dimension
      if (threadIdx.y == 0) {
        const Box box = d_desc_sorted_boxes[i_to_load];
        shared_i_areas[threadIdx.x] =
            (box.x2 - box.x1 + 1.0f) * (box.y2 - box.y1 + 1.0f);
        shared_i_boxes[threadIdx.x] = box;
      }
    }
    __syncthreads();
    const int i = i_block_offset + threadIdx.x;
    for (int j_thread_offset =
             NMS_BOXES_PER_THREAD * (blockIdx.y * blockDim.y + threadIdx.y);
         j_thread_offset < nboxes;
         j_thread_offset += NMS_BOXES_PER_THREAD * blockDim.y * gridDim.y) {
      // Note : We can do everything using multiplication,
      // and use fp16 - we are comparing against a low precision
      // threshold
      int above_thresh = 0;
      bool valid = false;
      for (int ib = 0; ib < NMS_BOXES_PER_THREAD; ++ib) {
        // This thread will compare Box i and Box j
        const int j = j_thread_offset + ib;
        if (i < j && i < nboxes && j < nboxes) {
          valid = true;
          const Box j_box = d_desc_sorted_boxes[j];
          const Box i_box = shared_i_boxes[threadIdx.x];
          const float j_area =
              (j_box.x2 - j_box.x1 + 1.0f) * (j_box.y2 - j_box.y1 + 1.0f);
          const float i_area = shared_i_areas[threadIdx.x];
          // The following code will not be valid with empty boxes
          if (i_area == 0.0f || j_area == 0.0f) continue;
          const float xx1 = fmaxf(i_box.x1, j_box.x1);
          const float yy1 = fmaxf(i_box.y1, j_box.y1);
          const float xx2 = fminf(i_box.x2, j_box.x2);
          const float yy2 = fminf(i_box.y2, j_box.y2);

          // fdimf computes the positive difference between xx2+1 and xx1
          const float w = fdimf(xx2 + 1.0f, xx1);
          const float h = fdimf(yy2 + 1.0f, yy1);
          const float intersection = w * h;

          // Testing for a/b > t
          // eq with a > b*t (b is !=0)
          // avoiding divisions
          const float a = intersection;
          const float b = i_area + j_area - intersection;
          const float bt = b * thresh;
          // eq. to if ovr > thresh
          if (a > bt) {
            // we have score[j] <= score[i]
            above_thresh |= (1U << ib);
          }
        }
      }
      if (valid) {
        d_delete_mask[i * mask_ld + j_thread_offset / NMS_BOXES_PER_THREAD] =
            above_thresh;
      }
    }
    __syncthreads();  // making sure everyone is done reading smem
  }
}

tensorflow::Status nms_gpu_upright(const float* d_desc_sorted_boxes_float_ptr,
                                   const int N, const float thresh,
                                   int* d_keep_sorted_list, int* h_nkeep,
                                   int* dev_delete_mask, int* host_delete_mask,
                                   OpKernelContext* context) {
  // d_desc_sorted_boxes_float_ptr is a pointer
  //    to device memory float array containing the box corners for N boxes.
  // threshod is the iou threshold for elimination
  // d_keep_sorted_list is a device pointer to an int bitmask array.
  //
  // Making sure we respect the __align(16)__ we promised to the compiler
  auto iptr = reinterpret_cast<std::uintptr_t>(d_desc_sorted_boxes_float_ptr);
  CHECK_EQ((iptr & 15), 0);

  // The next kernel expects squares

  const int mask_ld = (N + NMS_BOXES_PER_THREAD - 1) / NMS_BOXES_PER_THREAD;
  const Box* d_desc_sorted_boxes =
      reinterpret_cast<const Box*>(d_desc_sorted_boxes_float_ptr);
  int* d_delete_mask = dev_delete_mask;
  auto stream_exec = context->op_device_context()->stream();
  auto device = context->eigen_gpu_device();
  dim3 grid, block;
  int block_size = (N + NMS_BLOCK_DIM - 1) / NMS_BLOCK_DIM;
  block_size = std::max(std::min(block_size, NMS_BLOCK_DIM_MAX), 1);
  grid.x = block_size;
  grid.y = block_size;
  block.x = NMS_BLOCK_DIM;
  block.y = NMS_BLOCK_DIM;
  NMSKernel<<<grid, block, 0, device.stream()>>>(d_desc_sorted_boxes, N, thresh,
                                                 mask_ld, d_delete_mask);
  int* h_delete_mask = host_delete_mask;
  CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
  // Overlapping CPU computes and D2H memcpy
  // both take about the same time
  int nto_copy = std::min(NMS_CHUNK_SIZE, N);
  cudaEvent_t copy_done;
  cudaEventCreate(&copy_done);
  device.memcpyDeviceToHost(&h_delete_mask[0], &d_delete_mask[0],
                            nto_copy * mask_ld * sizeof(int));
  CUDA_CHECK(cudaEventRecord(copy_done, device.stream()));
  int offset = 0;
  std::vector<int> h_keep_sorted_list;
  std::vector<int> rmv(mask_ld, 0);
  memset(h_delete_mask, N, sizeof(int));
  while (offset < N) {
    const int ncopied = nto_copy;
    int next_offset = offset + ncopied;
    nto_copy = std::min(NMS_CHUNK_SIZE, N - next_offset);
    if (nto_copy > 0) {
      device.memcpyDeviceToHost(&h_delete_mask[next_offset * mask_ld],
                                &d_delete_mask[next_offset * mask_ld],
                                nto_copy * mask_ld * sizeof(int));
    }
    // Waiting for previous copy
    CUDA_CHECK(cudaEventSynchronize(copy_done));
    if (nto_copy > 0) CUDA_CHECK(cudaEventRecord(copy_done, device.stream()));
    for (int i = offset; i < next_offset; ++i) {
      int iblock = i / NMS_BOXES_PER_THREAD;
      int inblock = i % NMS_BOXES_PER_THREAD;
      if (!(rmv[iblock] & (1 << inblock))) {
        h_keep_sorted_list.push_back(i);
        int* p = &h_delete_mask[i * mask_ld];
        for (int ib = 0; ib < mask_ld; ++ib) {
          rmv[ib] |= p[ib];
        }
      }
    }
    offset = next_offset;
  }
  cudaEventDestroy(copy_done);

  const int nkeep = h_keep_sorted_list.size();
  device.memcpyHostToDevice(d_keep_sorted_list, &h_keep_sorted_list[0],
                            nkeep * sizeof(int));

  *h_nkeep = nkeep;
  return Status::OK();
}

}  // namespace tensorflow
#endif
