/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
// clean these

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/roi_align_op.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_device_functions.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/cub/device/device_segmented_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// taken from caffe2 implementation caffe2/operators/roi_align_(gradient)?_op.*
#define CUDA_CHECK(condition)                                 \
  do {                                                        \
    cudaError_t error = condition;                            \
    CHECK(error == cudaSuccess) << cudaGetErrorString(error); \
  } while (0)
namespace {
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
bilinear_interpolate(const T* bottom_data, const int height, const int width,
                     T y, T x, const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__global__ void RoIAlignForward(const CudaLaunchConfig nthreads,
                                const T* bottom_data, const T spatial_scale,
                                const int channels, const int height,
                                const int width, const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio, const T* bottom_rois,
                                int roi_cols, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads.virtual_thread_count) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // RoI could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;
    // T roi_start_w = roundf(offset_bottom_rois[0] * spatial_scale);
    // T roi_start_h = roundf(offset_bottom_rois[1] * spatial_scale);
    // T roi_end_w = roundf(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_h = roundf(offset_bottom_rois[3] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = Eigen::numext::maxi(roi_end_w - roi_start_w, (T)1.);
    T roi_height = Eigen::numext::maxi(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(offset_bottom_data, height, width, y, x,
                                     index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

template <typename T>
__global__ void RoIAlignBackwardFeature(
    const CudaLaunchConfig nthreads,
    const T* top_diff,  // grads
    const int num_rois, const T spatial_scale, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int sampling_ratio, const int roi_cols,
    const T* bottom_rois,  // rois
    T* bottom_diff /* input_grad */) {
  CUDA_1D_KERNEL_LOOP(index, nthreads.virtual_thread_count) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    // this part is buggy in caffe2. Inputs are allowed to be 4 or 5 columns
    // but caffe2 implementation gradient assumes 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;
    // T roi_start_w = roundf(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = roundf(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = roundf(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = roundf(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = Eigen::numext::maxi(roi_end_w - roi_start_w, (T)1.);
    T roi_height = Eigen::numext::maxi(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          CudaAtomicAdd(offset_bottom_diff + y_low * width + x_low,
                        static_cast<T>(g1));
          CudaAtomicAdd(offset_bottom_diff + y_low * width + x_high,
                        static_cast<T>(g2));
          CudaAtomicAdd(offset_bottom_diff + y_high * width + x_low,
                        static_cast<T>(g3));
          CudaAtomicAdd(offset_bottom_diff + y_high * width + x_high,
                        static_cast<T>(g4));
        }  // if
      }    // ix
    }      // iy
  }        // CUDA_1D_KERNEL_LOOP
}  // RoIAlignBackward

//  Adding caffe defines here
//
//
//
struct __align__(16) Box {
  float x1, y1, x2, y2;
};
// FIX THESE. Make use of the TF utilities and dimensions
#define NMS_BOXES_PER_THREAD (8 * sizeof(int))
constexpr int NMS_BOXES_PER_THREAD_BITS = 5;  // 3+2
#define NMS_CHUNK_SIZE 2000
#define CAFFE_CUDA_NUM_THREADS_2D_DIMX 16
#define CAFFE_CUDA_NUM_THREADS_2D_DIMY CAFFE_CUDA_NUM_THREADS_2D_DIMX
#define CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX 128
#define CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX
inline dim3 CAFFE_GET_BLOCKS_2D(const int N, const int /* M */) {
  dim3 grid;
  // Not calling the 1D version for each dim to keep all constants as literals

  grid.x =
      std::max(std::min((N + CAFFE_CUDA_NUM_THREADS_2D_DIMX - 1) /
                            CAFFE_CUDA_NUM_THREADS_2D_DIMX,
                        CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX),
               // Use at least 1 block, since CUDA does not allow empty block
               1);

  grid.y =
      std::max(std::min((N + CAFFE_CUDA_NUM_THREADS_2D_DIMY - 1) /
                            CAFFE_CUDA_NUM_THREADS_2D_DIMY,
                        CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY),
               // Use at least 1 block, since CUDA does not allow empty block
               1);

  return grid;
}

const dim3 CAFFE_CUDA_NUM_THREADS_2D = {
    static_cast<unsigned int>(CAFFE_CUDA_NUM_THREADS_2D_DIMX),
    static_cast<unsigned int>(CAFFE_CUDA_NUM_THREADS_2D_DIMY), 1u};
__launch_bounds__(
    CAFFE_CUDA_NUM_THREADS_2D_DIMX* CAFFE_CUDA_NUM_THREADS_2D_DIMY,
    4) __global__
    void NMSKernel(const Box* d_desc_sorted_boxes, const int nboxes,
                   const float thresh, const int mask_ld, int* d_delete_mask) {
  // Storing boxes used by this CUDA block in the shared memory
  __shared__ Box shared_i_boxes[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
  // Same thing with areas
  __shared__ float shared_i_areas[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
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
      if (valid)
        d_delete_mask[i * mask_ld + j_thread_offset / NMS_BOXES_PER_THREAD] =
            above_thresh;
    }
    __syncthreads();  // making sure everyone is done reading smem
  }
}
#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

// This kernel should execute in thenexecute otherwise memcpy and
tensorflow::Status nms_gpu_upright(const float* d_desc_sorted_boxes_float_ptr,
                                   const int N, const float thresh,
                                   int* d_keep_sorted_list, int* h_nkeep,
                                   int* dev_delete_mask, int* host_delete_mask,
                                   OpKernelContext* context) {
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
  NMSKernel<<<CAFFE_GET_BLOCKS_2D(N, mask_ld), CAFFE_CUDA_NUM_THREADS_2D, 0,
              device.stream()>>>(d_desc_sorted_boxes, N, thresh, mask_ld,
                                 d_delete_mask);

  int* h_delete_mask = host_delete_mask;

  // Overlapping CPU computes and D2H memcpy
  // both take about the same time
  int nto_copy = std::min(NMS_CHUNK_SIZE, N);
  cudaEvent_t copy_done;
  cudaEventCreate(&copy_done);
  CUDA_CHECK(cudaMemcpyAsync(&h_delete_mask[0], &d_delete_mask[0],
                             nto_copy * mask_ld * sizeof(int),
                             cudaMemcpyDeviceToHost, device.stream()));
  CUDA_CHECK(cudaEventRecord(copy_done, device.stream()));
  int offset = 0;
  std::vector<int> h_keep_sorted_list;
  std::vector<int> rmv(mask_ld, 0);
  while (offset < N) {
    const int ncopied = nto_copy;
    int next_offset = offset + ncopied;
    nto_copy = std::min(NMS_CHUNK_SIZE, N - next_offset);
    if (nto_copy > 0) {
      CUDA_CHECK(cudaMemcpyAsync(&h_delete_mask[next_offset * mask_ld],
                                 &d_delete_mask[next_offset * mask_ld],
                                 nto_copy * mask_ld * sizeof(int),
                                 cudaMemcpyDeviceToHost, device.stream()));
    }
    // Waiting for previous copy
    CUDA_CHECK(cudaEventSynchronize(copy_done));
    if (nto_copy > 0) cudaEventRecord(copy_done, device.stream());
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
  cudaMemcpyAsync(d_keep_sorted_list, &h_keep_sorted_list[0],
                  nkeep * sizeof(int), cudaMemcpyHostToDevice, device.stream());

  *h_nkeep = nkeep;
  // se::DeviceMemoryBase dev_ptr(dev_delete_mask, N * mask_ld * sizeof(int32));
  // const bool status =
  //     stream->ThenMemcpy(host_delete_mask, dev_ptr, N * mask_ld *
  //     sizeof(int32))
  //         .ok();

  // if (!status) {
  //   return errors::Internal("Failed to launch copy from device to host.");
  // }
  // // device.memcpy(host_delete_mask,dev_delete_mask,N * mask_ld *
  // // sizeof(int32));
  // // CUDA_CHECK(cudaMemcpyAsync(&h_delete_mask[0], &d_delete_mask[0],
  // //                            nto_copy * mask_ld * sizeof(int),
  // //                            cudaMemcpyDeviceToHost,
  // //                            context->cuda_stream()));
  // auto host_filtering = [N, h_delete_mask, d_keep_sorted_list, mask_ld,
  // h_nkeep,
  //                        context, done]() {
  //   auto stream = context->op_device_context()->stream();
  //   ScopedActivateExecutorContext scoped_activation{stream->parent()};
  //   std::vector<int> h_keep_sorted_list;
  //   h_keep_sorted_list.reserve(N);
  //   std::vector<int> rmv(mask_ld, 0);
  //   for (int i = 0; i < N; ++i) {
  //     int iblock = i / NMS_BOXES_PER_THREAD;
  //     int inblock = i % NMS_BOXES_PER_THREAD;
  //     if (!(rmv[iblock] & (1 << inblock))) {
  //       h_keep_sorted_list.push_back(i);
  //       int* p = &h_delete_mask[i * mask_ld];
  //       for (int ib = 0; ib < mask_ld; ++ib) {
  //         rmv[ib] |= p[ib];
  //       }
  //     }
  //   }
  //   *h_nkeep = h_keep_sorted_list.size();
  //   se::DeviceMemoryBase dev_ptr(d_keep_sorted_list,
  //                                h_keep_sorted_list.size() * sizeof(int32));
  //   const bool status = stream
  //                           ->ThenMemcpy(dev_ptr, &h_keep_sorted_list[0],
  //                                        *h_nkeep * sizeof(int32))
  //                           .ok();

  //   if (!status) {
  //     context->SetStatus(
  //         errors::Internal("Failed to launch copy from device to host."));
  //   }
  // };
  // context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
  //     stream, host_filtering);
  // const int nkeep = h_keep_sorted_list.size();

  // cudaMemcpyAsync(d_keep_sorted_list, &h_keep_sorted_list[0],
  //                 nkeep * sizeof(int), cudaMemcpyHostToDevice,
  //                 context->cuda_stream());
  return Status::OK();
}

/**
 * d_sorted_score_keys -- indexes into _original_ scores
 * nboxes_to_generate -- pre_nms_topn
 */
__global__ void GeneratePreNMSUprightBoxesKernel(
    const Cuda2DLaunchConfig config, const int* d_sorted_scores_keys,
    const float4* d_bbox_deltas, const float4* d_anchors, const int H,
    const int W, const int A, const float feat_stride, const float min_size,
    const float* d_img_info_vec, const float bbox_xform_clip,
    const bool correct_transform, float4* d_out_boxes,
    const int prenms_nboxes,  // leading dimension of out_boxes
    float* d_inout_scores, char* d_boxes_keep_flags) {
  const int K = H * W;
  const int WA = W * A;
  const int KA = K * A;
  int nboxes_to_generate=config.virtual_thread_count.x;
  int num_images=config.virtual_thread_count.y;
  CUDA_AXIS_KERNEL_LOOP(image_index, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(ibox, config.virtual_thread_count.x, X) {
      // CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images){
      // { box_conv_index : # of the same box, but indexed in the scores from
      // the conv layer, of shape (A,H,W) the num_images dimension was already
      // removed box_conv_index = a*K + h*W + w
      const int box_conv_index = d_sorted_scores_keys[image_index * KA + ibox];

      // We want to decompose box_conv_index in (h,w,a)
      // such as box_conv_index = h*W*A + W*A + a
      // (avoiding modulos in the process)
      int remaining = box_conv_index;
      const int dH = WA;  // stride of H
      const int h = remaining / dH;
      remaining -= h * dH;
      const int dW = A; // stride of H
      const int w = remaining/dW;  
      remaining -= w * dW;
      const int a = remaining; // dA = 1
      // Loading the anchor a
      // float4 is a struct with float x,y,z,w
      const float4 anchor = d_anchors[a];
      // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
      const float shift_w = feat_stride * w;
      float x1 = shift_w + anchor.x;
      float x2 = shift_w + anchor.z;
      const float shift_h = feat_stride * h;
      float y1 = shift_h + anchor.y;
      float y2 = shift_h + anchor.w;

      // TODO use fast math when possible

    // Deltas of shape (N,H,W,A4)
    int deltas_idx = box_conv_index; 
     float4 deltas = d_bbox_deltas[deltas_idx];
     float dx = deltas.x;
     float dy = deltas.y;
     float dw = deltas.z;
     float dh = deltas.w;
     //printf("deltas_idx=%d dx=%f, dy=%f, dw=%f, dh=%f\n",deltas_idx,dx,dy,dw,dh);
      // Upper bound on dw,dh
      dw = fmin(dw, bbox_xform_clip);
      dh = fmin(dh, bbox_xform_clip);

      // Applying the deltas
      float width = x2 - x1 + 1.0f;
      const float ctr_x = x1 + 0.5f * width;
      const float pred_ctr_x = ctr_x + width * dx;  // TODO fuse madd
      const float pred_w = width * expf(dw);
      x1 = pred_ctr_x - 0.5f * pred_w;
      x2 = pred_ctr_x + 0.5f * pred_w;

      float height = y2 - y1 + 1.0f;
      const float ctr_y = y1 + 0.5f * height;
      const float pred_ctr_y = ctr_y + height * dy;
      const float pred_h = height * expf(dh);
      y1 = pred_ctr_y - 0.5f * pred_h;
      y2 = pred_ctr_y + 0.5f * pred_h;

      if (correct_transform) {
        x2 -= 1.0f;
        y2 -= 1.0f;
      }

      // Clipping box to image
      const float img_height = d_img_info_vec[2 * image_index + 0];
      const float img_width = d_img_info_vec[2 * image_index + 1];
      const float min_size_scaled = min_size;//*0.166667;
          //min_size * d_img_info_vec[3 * image_index + 2];
      x1 = fmax(fmin(x1, img_width - 1.0f), 0.0f);
      y1 = fmax(fmin(y1, img_height - 1.0f), 0.0f);
      x2 = fmax(fmin(x2, img_width - 1.0f), 0.0f);
      y2 = fmax(fmin(y2, img_height - 1.0f), 0.0f);

      // Filter boxes
      // Removing boxes with one dim < min_size
      // (center of box is in image, because of previous step)
      width = x2 - x1 + 1.0f;  // may have changed
      height = y2 - y1 + 1.0f;
      bool keep_box = fmin(width, height) >= min_size_scaled;

      // We are not deleting the box right now even if !keep_box
      // we want to keep the relative order of the elements stable
      // we'll do it in such a way later
      // d_boxes_keep_flags size: (num_images,prenms_nboxes)
      // d_out_boxes size: (num_images,prenms_nboxes)
      const int out_index = image_index * prenms_nboxes + ibox;
      d_boxes_keep_flags[out_index] = keep_box;
      d_out_boxes[out_index] = {x1, y1, x2, y2};

      // d_inout_scores size: (num_images,KA)
      if (!keep_box)
        d_inout_scores[image_index * KA + ibox] = FLT_MIN;  // for NMS
    }
  }
}
__global__ void WriteUprightBoxesOutput(const CudaLaunchConfig nboxes,
                                        const float4* d_image_boxes,
                                        const float* d_image_scores,
                                        const int* d_image_boxes_keep_list,
                                        const int image_index,
                                        float* d_image_out_rois,
                                        float* d_image_out_rois_probs) {
  CUDA_1D_KERNEL_LOOP(i, nboxes.virtual_thread_count) {
    const int ibox = d_image_boxes_keep_list[i];
    const float4 box = d_image_boxes[ibox];
    const float score = d_image_scores[ibox];
    // Scattered memory accesses
    // postnms_nboxes is small anyway
    d_image_out_rois_probs[i] = score;
    const int base_idx = 5 * i;
    d_image_out_rois[base_idx + 0] = image_index;
    d_image_out_rois[base_idx + 1] = box.x;
    d_image_out_rois[base_idx + 2] = box.y;
    d_image_out_rois[base_idx + 3] = box.z;
    d_image_out_rois[base_idx + 4] = box.w;
  }
}

__global__ void InitializeDataKernel(const Cuda2DLaunchConfig config,
                                     int* d_image_offsets,
                                     int* d_boxes_keys_iota) {
  const int KA = config.virtual_thread_count.x;
  const int num_images = config.virtual_thread_count.y;
  //printf("num_images %d KA %d\n",num_images,KA);
   CUDA_AXIS_KERNEL_LOOP(img_idx, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(box_idx, config.virtual_thread_count.x, X) {
   //CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {
      d_boxes_keys_iota[img_idx * KA + box_idx] = box_idx;

      // One 1D line sets the 1D data
      if (box_idx == 0) {
        d_image_offsets[img_idx] = KA * img_idx;
        // One thread sets the last+1 offset
        if (img_idx == 0) d_image_offsets[num_images] = KA * num_images;
      }
    }
   }
}

}  // namespace
namespace functor {

template <typename T>
struct ROIAlign<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor X,
                  typename TTypes<T, 2>::ConstTensor RoIs,
                  const int pooled_height, const int pooled_width,
                  const int sampling_ratio, const T spatial_scale,
                  typename TTypes<float, 4>::Tensor Y) {
    const int channels = X.dimension(1);
    const int height = X.dimension(2);
    const int width = X.dimension(3);
    int roi_cols = RoIs.dimension(1);

    const int total_count = Y.size();
    if (total_count == 0) return;
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    // clang-format off
    RoIAlignForward<T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            config, X.data(), spatial_scale, channels,
            height, width, pooled_height, pooled_width, sampling_ratio, RoIs.data(),
            roi_cols, Y.data());
    // clang-format on
  }
};

template <typename T>
struct ROIAlignGrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor grads,
                  typename TTypes<T, 4>::ConstTensor inputs,
                  typename TTypes<T, 2>::ConstTensor rois,
                  const int pooled_height, const int pooled_width,
                  const int sampling_ratio, const T spatial_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int channels = inputs.dimension(1);
    const int height = inputs.dimension(2);
    const int width = inputs.dimension(3);
    const int num_rois = rois.dimension(0);
    int total_count = output.size();
    // reset grads
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, output.data());
    total_count = grads.size();
    if (total_count == 0) return;
    config = GetCudaLaunchConfig(total_count, d);
    // clang-format off
    RoIAlignBackwardFeature<T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            config, grads.data(), num_rois,spatial_scale, channels,
            height, width, pooled_height, pooled_width, sampling_ratio, rois.dimension(1), rois.data(),
            output.data());
    // clang-format on
  }
};
// template <typename T>
// struct NMSGPUUpright<GPUDevice, T> {
//   void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor
//   boxes,
//                   const int N, const float treshold) {
//     Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(
//         N, (N + GENRPN_BOXES_PER_THREAD - 1) / GENRPN_BOXES_PER_THREAD, d);
//   }
// };

// template <typename T>
// struct GeneratePreNMSUprightBoxes<GPUDevice & d, T> {
//   void operator()(
//       const Device& d,
//       typename TTypes<T, 4>::ConstTensor digits,  // Scores [N, A, H, W]
//       typename TTypes<T, 4>::ConstTensor
//           bbox_deltas,  // [N, A*4, H, W] (full, unsorted / sliced)
//       typename TTypes<T, 2>::ConstTensor
//           image_shapes,  // (N, 3 ) (h, w, scale) of images
//       typename TTypes<T, 2>::ConstTensor anchors,  // (A,4)
//       const T spatial_scale, const int pre_nms_topN, const int post_nms_topN,
//       const T nms_thresh, const T min_size, const bool
//       correct_transform_coords, typename TTypes<T, 2>::Tensor rois, typename
//       TTypes<T, 1>::Tensor roi_probs) {
//     constexpr int box_dim = 4;
//     const int K = H * W;
//     const int conv_layer_nboxes = K * A;
//     int total_count = boxes.size();
//     CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
//     SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
//         config.virtual_thread_count, boxes.data());
//     total_count = boxes_keep_flags.size();
//     CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
//     SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
//         config.virtual_thread_count, boxes_keep_flags.data());
//     Cuda2DLaunchConfig config2d = GetCuda2DLaunchConfig(
//         pre_nms_nboxes, num_images, d) GeneratePreNMSUprightBoxes
//   }
// };
}  // namespace functor

Status AllocateGenerationTempTensors(
    OpKernelContext* context, Tensor* d_conv_layer_indexes,
    Tensor* d_image_offset, Tensor* d_cub_sort_buffer,
    Tensor* d_cub_select_buffer, Tensor* d_sorted_conv_layer_indexes,
    Tensor* d_sorted_scores, Tensor* dev_boxes, Tensor* dev_boxes_keep_flags,
    int num_images, int conv_layer_nboxes, size_t cub_sort_temp_storage_bytes,
    size_t cub_select_temp_storage_bytes, int nboxes_to_generate, int box_dim) {
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images, conv_layer_nboxes}),
      d_conv_layer_indexes));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images + 1}), d_image_offset));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_sort_temp_storage_bytes}),
      d_cub_sort_buffer));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_select_temp_storage_bytes}),
      d_cub_select_buffer));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images, conv_layer_nboxes}),
      d_sorted_conv_layer_indexes));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_images, conv_layer_nboxes}),
      d_sorted_scores));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT,
      TensorShape({num_images, box_dim * nboxes_to_generate}), dev_boxes));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({num_images, nboxes_to_generate}),
      dev_boxes_keep_flags));
  return Status::OK();
}

Status AllocatePreNMSTempTensors(
    OpKernelContext* context, Tensor* dev_image_prenms_boxes,
    Tensor* dev_image_prenms_scores, Tensor* dev_image_boxes_keep_list,
    Tensor* dev_postnms_rois, Tensor* dev_postnms_rois_probs,
    Tensor* dev_prenms_nboxes, Tensor* dev_nms_mask, Tensor* host_nms_mask,
    int num_images, int nboxes_to_generate, int box_dim, int post_nms_topn,
    int pre_nms_topn) {
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({box_dim * nboxes_to_generate}),
      dev_image_prenms_boxes));
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_FLOAT,
                                            TensorShape({nboxes_to_generate}),
                                            dev_image_prenms_scores));
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({nboxes_to_generate}),
                                            dev_image_boxes_keep_list));
  const int roi_cols = box_dim + 1;
  const int max_postnms_nboxes = std::min(nboxes_to_generate, post_nms_topn);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT,
      TensorShape({roi_cols * num_images * max_postnms_nboxes}),
      dev_postnms_rois));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_images * max_postnms_nboxes}),
      dev_postnms_rois_probs));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images}), dev_prenms_nboxes));

  int64 max_nms_mask_size =
      pre_nms_topn *
      ((pre_nms_topn + NMS_BOXES_PER_THREAD - 1) / NMS_BOXES_PER_THREAD);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({max_nms_mask_size}), dev_nms_mask));
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  alloc_attr.set_gpu_compatible(true);
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({max_postnms_nboxes}),
                                            host_nms_mask, alloc_attr));

  // se::DeviceMemoryBase wrapped(partition_count.flat<int32>().data(),
  //                              num_partitions_ * sizeof(int32));
  // const bool status =
  //     stream
  //         ->ThenMemcpy(cpu_tensor.flat<int32>().data(), wrapped,
  //                      num_partitions_ * sizeof(int32))
  //         .ok();
  // OP_REQUIRES_ASYNC(
  //     c, status,
  //     errors::Internal("Failed to launch copy from device to host."), done);

  return Status::OK();
}
namespace sami {
// This implemantation is a pytorch compatible implementation and is not good
// for tensorflow. Synchronizations will cause this op to show up as consuming
// more resources than it actually is in profiling. This either should be split
// into multiple ops or need to be executed with ThenExecute() but that will
// slow down other events since execution will be done by event polling thread.

class GenerateBoundingBoxProposals : public tensorflow::AsyncOpKernel {
 public:
  explicit GenerateBoundingBoxProposals(
      tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
    feat_stride_ = 1.0 / spatial_scale_;
    OP_REQUIRES_OK(context, context->GetAttr("pre_nms_topn", &pre_nms_topn_));
    OP_REQUIRES_OK(context, context->GetAttr("post_nms_topn", &post_nms_topn_));
    OP_REQUIRES_OK(context, context->GetAttr("nms_threshold", &nms_threshold_));
    OP_REQUIRES_OK(context, context->GetAttr("min_size", &min_size_));
    // compatibility for detectron like networks. False for generic case
    OP_REQUIRES_OK(context, context->GetAttr("correct_transform_coords", &correct_transform_coords_));
    CHECK_GT(spatial_scale_, 0);
    CHECK_GT(pre_nms_topn_, 0);
    CHECK_GT(post_nms_topn_, 0);
    CHECK_GT(nms_threshold_, 0);
    CHECK_GT(min_size_, 0);
    bbox_xform_clip_default_ = log(1000.0 / 16.);
  }

  void ComputeAsync(tensorflow::OpKernelContext* context,
                    DoneCallback done) override {
    // .Input("scores: float")
    // .Input("bbox_deltas: float")
    // .Input("image_info: float")
    // .Input("anchors: float")

    const auto scores = context->input(0);
    const auto bbox_deltas = context->input(1);
    const auto image_info = context->input(2);
    const auto anchors = context->input(3);
    const auto num_images = scores.dim_size(0);
    const auto A = scores.dim_size(3);
    const auto H = scores.dim_size(1);
    const auto W = scores.dim_size(2);
    const auto box_dim = anchors.dim_size(1);
    // TODO(skama): make sure that inputs are ok.
    const int K = H * W;
    VLOG(0)<<"num_images="<<num_images<<" A="<<A<<" H="<<H<<" W="<<W;
    const int conv_layer_nboxes = K * A;
    // The following calls to CUB primitives do nothing
    // (because the first arg is nullptr)
    // except setting cub_*_temp_storage_bytes
    auto cuda_stream = GetCudaStream(context);
    size_t cub_sort_temp_storage_bytes = 0;
    float* flt_ptr = nullptr;
    int* int_ptr = nullptr;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, cub_sort_temp_storage_bytes, flt_ptr, flt_ptr, int_ptr,
        int_ptr, num_images * conv_layer_nboxes, num_images, int_ptr, int_ptr,
        0, 8 * sizeof(float),  // sort all bits
        cuda_stream);

    // get the size of select temp buffer
    size_t cub_select_temp_storage_bytes = 0;
    char* char_ptr = nullptr;
    cub::DeviceSelect::Flagged(nullptr, cub_select_temp_storage_bytes, flt_ptr,
                               char_ptr, flt_ptr, int_ptr, K * A, cuda_stream);
    Tensor d_conv_layer_indexes;
    Tensor d_image_offset;
    Tensor d_cub_sort_buffer;
    Tensor d_cub_select_buffer;
    Tensor d_sorted_conv_layer_indexes;
    Tensor dev_sorted_scores;
    Tensor dev_boxes;
    Tensor dev_boxes_keep_flags;
    const int nboxes_to_generate = std::min(conv_layer_nboxes, pre_nms_topn_);
    OP_REQUIRES_OK_ASYNC(
        context,
        AllocateGenerationTempTensors(
            context, &d_conv_layer_indexes, &d_image_offset, &d_cub_sort_buffer,
            &d_cub_select_buffer, &d_sorted_conv_layer_indexes,
            &dev_sorted_scores, &dev_boxes, &dev_boxes_keep_flags, num_images,
            conv_layer_nboxes, cub_sort_temp_storage_bytes,
            cub_select_temp_storage_bytes, nboxes_to_generate, box_dim),
        done);
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    Cuda2DLaunchConfig conf2d =
        GetCuda2DLaunchConfig(conv_layer_nboxes, num_images, d);

    InitializeDataKernel<<<conf2d.block_count, conf2d.thread_per_block, 0,
                           d.stream()>>>(
        conf2d, d_image_offset.flat<int>().data(),
        d_conv_layer_indexes.flat<int>().data());
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_cub_sort_buffer.flat<int8>().data(), cub_sort_temp_storage_bytes,
        scores.flat<float>().data(), dev_sorted_scores.flat<float>().data(),
        d_conv_layer_indexes.flat<int>().data(),
        d_sorted_conv_layer_indexes.flat<int>().data(),
        num_images * conv_layer_nboxes, num_images,
        d_image_offset.flat<int>().data(),
        d_image_offset.flat<int>().data() + 1, 0,
        8 * sizeof(float),  // sort all bits
        cuda_stream);
    // Keeping only the topN pre_nms

    conf2d = GetCuda2DLaunchConfig(nboxes_to_generate, num_images, d);
    GeneratePreNMSUprightBoxesKernel<<<
        conf2d.block_count, conf2d.thread_per_block, 0, d.stream()>>>(
        conf2d, d_sorted_conv_layer_indexes.flat<int>().data(),
        reinterpret_cast<const float4*>(bbox_deltas.flat<float>().data()),
        reinterpret_cast<const float4*>(anchors.flat<float>().data()), H, W, A,
        feat_stride_, min_size_, image_info.flat<float>().data(),
        bbox_xform_clip_default_, correct_transform_coords_,
        reinterpret_cast<float4*>(dev_boxes.flat<float>().data()),
        nboxes_to_generate, dev_sorted_scores.flat<float>().data(),
        (char*)dev_boxes_keep_flags.flat<int8>().data());
    const int nboxes_generated = nboxes_to_generate;
    const int roi_cols = box_dim + 1;
    const int max_postnms_nboxes = std::min(nboxes_generated, post_nms_topn_);
    Tensor dev_image_prenms_boxes;
    Tensor dev_image_prenms_scores;
    Tensor dev_image_boxes_keep_list;
    Tensor dev_postnms_rois;
    Tensor dev_postnms_rois_probs;
    Tensor dev_prenms_nboxes;
    Tensor dev_nms_mask;
    Tensor host_nms_mask;
    OP_REQUIRES_OK_ASYNC(
        context,
        AllocatePreNMSTempTensors(
            context, &dev_image_prenms_boxes, &dev_image_prenms_scores,
            &dev_image_boxes_keep_list, &dev_postnms_rois,
            &dev_postnms_rois_probs, &dev_prenms_nboxes, &dev_nms_mask,
            &host_nms_mask, num_images, nboxes_generated, box_dim,
            this->post_nms_topn_, this->pre_nms_topn_),
        done);
    int* d_prenms_nboxes = dev_prenms_nboxes.flat<int>().data();
    int h_prenms_nboxes;
    char* d_boxes_keep_flags = (char*)dev_boxes_keep_flags.flat<int8>().data();
    float* d_postnms_rois = dev_postnms_rois.flat<float>().data();
    float* d_postnms_rois_probs = dev_postnms_rois_probs.flat<float>().data();
    char* d_cub_select_temp_storage =
        (char*)d_cub_select_buffer.flat<int8>().data();
    float* d_image_prenms_boxes = dev_image_prenms_boxes.flat<float>().data();
    float* d_image_prenms_scores = dev_image_prenms_scores.flat<float>().data();
    int* d_image_boxes_keep_list = dev_image_boxes_keep_list.flat<int>().data();
    int nrois_in_output = 0;
    float* d_boxes = dev_boxes.flat<float>().data();
    int* h_nms_mask = host_nms_mask.flat<int>().data();
    int* d_nms_mask = dev_nms_mask.flat<int>().data();
    float* d_sorted_scores = dev_sorted_scores.flat<float>().data();
    for (int image_index = 0; image_index < num_images; ++image_index) {
      // Sub matrices for current image
      const float* d_image_boxes =
          &d_boxes[image_index * nboxes_generated * box_dim];
      const float* d_image_sorted_scores =
          &d_sorted_scores[image_index * K * A];
      char* d_image_boxes_keep_flags =
          &d_boxes_keep_flags[image_index * nboxes_generated];

      float* d_image_postnms_rois = &d_postnms_rois[roi_cols * nrois_in_output];
      float* d_image_postnms_rois_probs =
          &d_postnms_rois_probs[nrois_in_output];

      // Moving valid boxes (ie the ones with d_boxes_keep_flags[ibox] == true)
      // to the output tensors
      cub::DeviceSelect::Flagged(
          d_cub_select_temp_storage, cub_select_temp_storage_bytes,
          reinterpret_cast<const float4*>(d_image_boxes),
          d_image_boxes_keep_flags,
          reinterpret_cast<float4*>(d_image_prenms_boxes), d_prenms_nboxes,
          nboxes_generated, d.stream());

      cub::DeviceSelect::Flagged(
          d_cub_select_temp_storage, cub_select_temp_storage_bytes,
          d_image_sorted_scores, d_image_boxes_keep_flags,
          d_image_prenms_scores, d_prenms_nboxes, nboxes_generated, d.stream());
      d.memcpyDeviceToHost(&h_prenms_nboxes, d_prenms_nboxes, sizeof(int));
      d.synchronize();
      // We know prenms_boxes <= topN_prenms, because nboxes_generated <=
      // topN_prenms. Calling NMS on the generated boxes
      const int prenms_nboxes = h_prenms_nboxes;
      int nkeep;
      nms_gpu_upright(d_image_prenms_boxes, prenms_nboxes, nms_threshold_,
                      d_image_boxes_keep_list, &nkeep, d_nms_mask, h_nms_mask,
                      context);

      // All operations done after previous sort were keeping the relative order
      // of the elements the elements are still sorted keep topN <=> truncate
      // the array
      const int postnms_nboxes = std::min(nkeep, post_nms_topn_);

      // Moving the out boxes to the output tensors,
      // adding the image_index dimension on the fly
      CudaLaunchConfig config = GetCudaLaunchConfig(postnms_nboxes, d);
      WriteUprightBoxesOutput<<<config.block_count, config.thread_per_block, 0,
                                d.stream()>>>(
          config, reinterpret_cast<const float4*>(d_image_prenms_boxes),
          d_image_prenms_scores, d_image_boxes_keep_list, image_index,
          d_image_postnms_rois, d_image_postnms_rois_probs);
      nrois_in_output += postnms_nboxes;
    }
    Tensor* output_rois = nullptr;
    Tensor* output_roi_probs = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(0, TensorShape({nrois_in_output, roi_cols}),
                                 &output_rois),
        done);
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(1, TensorShape({nrois_in_output}),
                                 &output_roi_probs),
        done);
    d.memcpyDeviceToHost(output_rois->flat<float>().data(), d_postnms_rois,
                         nrois_in_output * roi_cols * sizeof(float));
    d.memcpyDeviceToHost(output_roi_probs->flat<float>().data(),
                         d_postnms_rois_probs, nrois_in_output * sizeof(float));
    done();
  }

 private:
  // .Output("rois: float")
  // .Output("roi_probabilities: float")
  // .Attr("spatial_scale: float = 1.0")
  // .Attr("pre_nms_topN: int")
  // .Attr("post_nms_topN: int")
  // .Attr("nms_threshold: float")
  // .Attr("min_size: float")
  struct SharedData {
    SharedData() : n_rois_in_output(0), prenms_nboxes(0), nkeep(0){};
    int n_rois_in_output;
    int prenms_nboxes;
    int nkeep;
    Status last_status;
  };
  float spatial_scale_;
  int pre_nms_topn_;
  int post_nms_topn_;
  float nms_threshold_;
  float min_size_;
  float feat_stride_;
  float bbox_xform_clip_default_;
  bool correct_transform_coords_;
};

#undef GENRPN_BOXES_PER_THREAD
#undef GENRPN_CHUNK_SIZE
}  // namespace sami
template struct functor::ROIAlignGrad<GPUDevice, float>;
template struct functor::ROIAlign<GPUDevice, float>;
REGISTER_KERNEL_BUILDER(
    Name("GenerateBoundingBoxProposals").Device(tensorflow::DEVICE_GPU),
    tensorflow::sami::GenerateBoundingBoxProposals)
}  // namespace tensorflow
#endif