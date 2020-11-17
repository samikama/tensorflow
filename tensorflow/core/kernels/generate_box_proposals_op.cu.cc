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

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/non_max_suppression_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;
#define TF_RETURN_IF_CUDA_ERROR(result)                   \
  do {                                                    \
    cudaError_t error(result);                            \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {         \
      return errors::Internal("Cuda call failed with ",   \
                              cudaGetErrorString(error)); \
    }                                                     \
  } while (0)

#define TF_OP_REQUIRES_CUDA_SUCCESS(context, result)                   \
  do {                                                                 \
    cudaError_t error(result);                                         \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {                      \
      context->SetStatus(errors::Internal("Cuda call failed with",     \
                                          cudaGetErrorString(error))); \
      return;                                                          \
    }                                                                  \
  } while (0)

constexpr int threadsPerBlock = 64;
constexpr int block_size = sizeof(unsigned long long) * 8;
constexpr int max_shmem_size = 49152;

namespace GBBP {

// Decode d_bbox_deltas with respect to anchors into absolute coordinates,
// clipping if necessary.
// prenms_nboxes maximum number of boxes per image to decode.
// d_boxes_keep_flags mask for boxes to consider in NMS.
// min_size is the lower bound of the shortest edge for the boxes to consider.
// bbox_xform_clip is the upper bound of encoded width and height.
__global__ void GeneratePreNMSUprightBoxesKernel(
    const Cuda2DLaunchConfig config, const int* d_sorted_scores_keys,
    const float4* d_bbox_deltas, const float4* d_anchors, const int height,
    const int width, const int num_anchors, const float min_size,
    const float* d_img_info_vec,  // Input "image_info" to the op [N,5]
    const float bbox_xform_clip, float4* d_out_boxes,
    const int prenms_nboxes,  // leading dimension of out_boxes
    char* d_boxes_keep_flags) {
  // constants to calculate offsets in to the input and output arrays.
  const int anchor_stride = height * width;              // Stride of Anchor
  const int height_stride = width * num_anchors;         // Stride of height
  const int image_stride = anchor_stride * num_anchors;  // Stride of image
  CUDA_AXIS_KERNEL_LOOP(image_index, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(ibox, config.virtual_thread_count.x, X) {
      // box_conv_index : # of the same box, but indexed in the
      // scores from the conv layer, of shape (height,width,num_anchors) the
      // num_images dimension was already removed box_conv_index =
      // a*image_stride + h*width + w
      const int box_conv_index =
          d_sorted_scores_keys[image_index * image_stride + ibox];

      // We want to decompose box_conv_index in (h,w,a)
      // such as box_conv_index = h*width*num_anchors + width*num_anchors + a
      // (avoiding modulos in the process)
      int remaining = box_conv_index;
      const int delta_height = height_stride;  // stride of height
      const int h = remaining / delta_height;
      remaining -= h * delta_height;
      const int delta_width = num_anchors;  // stride of width
      const int w = remaining / delta_width;
      remaining -= w * delta_width;
      // Loading the anchor a
      // float4 is a struct with float x,y,z,w
      const float4 anchor = d_anchors[box_conv_index];
      // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
      float x1 = anchor.y;
      float x2 = anchor.w;
      float y1 = anchor.x;
      float y2 = anchor.z;

      // TODO use fast math when possible

      // Deltas of shape (N,height,width,num_anchors x 4)
      int deltas_idx = box_conv_index + image_index * image_stride;
      float4 deltas = d_bbox_deltas[deltas_idx];
      float dx = deltas.y;
      float dy = deltas.x;
      float dw = deltas.w;
      float dh = deltas.z;
      // Upper bound on dw,dh
      dw = fmin(dw, bbox_xform_clip);
      dh = fmin(dh, bbox_xform_clip);

      // Applying the deltas
      float width = x2 - x1;
      const float ctr_x = x1 + 0.5f * width;
      const float pred_ctr_x = ctr_x + width * dx;  // TODO fuse madd
      const float pred_w = width * expf(dw);
      x1 = pred_ctr_x - 0.5f * pred_w;
      x2 = pred_ctr_x + 0.5f * pred_w;

      float height = y2 - y1;
      const float ctr_y = y1 + 0.5f * height;
      const float pred_ctr_y = ctr_y + height * dy;
      const float pred_h = height * expf(dh);
      y1 = pred_ctr_y - 0.5f * pred_h;
      y2 = pred_ctr_y + 0.5f * pred_h;

      // Clipping box to image
      const float img_height = d_img_info_vec[5 * image_index + 0];
      const float img_width = d_img_info_vec[5 * image_index + 1];
      const float min_size_scaled =
          min_size * d_img_info_vec[5 * image_index + 2];
      x1 = fmax(fmin(x1, img_width), 0.0f);
      y1 = fmax(fmin(y1, img_height), 0.0f);
      x2 = fmax(fmin(x2, img_width), 0.0f);
      y2 = fmax(fmin(y2, img_height), 0.0f);

      // Filter boxes
      // Removing boxes with one dim < min_size
      // (center of box is in image, because of previous step)
      width = x2 - x1;  // may have changed
      height = y2 - y1;
      bool keep_box = fmin(width, height) >= min_size_scaled;

      // We are not deleting the box right now even if !keep_box
      // we want to keep the relative order of the elements stable
      // we'll do it in such a way later
      // d_boxes_keep_flags size: (num_images,prenms_nboxes)
      // d_out_boxes size: (num_images,prenms_nboxes)
      const int out_index = image_index * prenms_nboxes + ibox;

      d_boxes_keep_flags[out_index] = keep_box;
      d_out_boxes[out_index] = {x1, y1, x2, y2};
    }
  }
}

// Copy the selected boxes and scores to output tensors.
//
__global__ void WriteUprightBoxesOutput(
    const CudaLaunchConfig nboxes, const float4* d_image_boxes,
    const float* d_image_scores, const int* d_image_boxes_keep_list,
    const int n_rois, float* d_image_out_rois, float* d_image_out_rois_probs) {
  CUDA_1D_KERNEL_LOOP(i, nboxes.virtual_thread_count) {
    if (i < n_rois) {  // copy rois to output
      const int ibox = d_image_boxes_keep_list[i];
      const float4 box = d_image_boxes[ibox];
      const float score = d_image_scores[ibox];
      // Scattered memory accesses
      // postnms_nboxes is small anyway
      d_image_out_rois_probs[i] = score;
      const int base_idx = 4 * i;
      d_image_out_rois[base_idx + 0] = box.y;
      d_image_out_rois[base_idx + 1] = box.x;
      d_image_out_rois[base_idx + 2] = box.w;
      d_image_out_rois[base_idx + 3] = box.z;
    } else {  // set trailing entries to 0
      d_image_out_rois_probs[i] = 0.;
      const int base_idx = 4 * i;
      d_image_out_rois[base_idx + 0] = 0.;
      d_image_out_rois[base_idx + 1] = 0.;
      d_image_out_rois[base_idx + 2] = 0.;
      d_image_out_rois[base_idx + 3] = 0.;
    }
  }
}

template <typename T>
Status ResetTensor(Tensor* t, const Eigen::GpuDevice& d) {
  CudaLaunchConfig zconfig = GetCudaLaunchConfig(t->NumElements(), d);
  return GpuLaunchKernel(SetZero<T>, zconfig.block_count,
                         zconfig.thread_per_block, 0, d.stream(),
                         zconfig.virtual_thread_count, (*t).flat<T>().data());
}
// Allocate scratch spaces that are needed for operation
//

Status AllocateGenerationTempTensors(
    OpKernelContext* context, Tensor* d_conv_layer_indexes,
    Tensor* d_image_offset, Tensor* d_cub_temp_buffer,
    Tensor* d_sorted_conv_layer_indexes, Tensor* d_sorted_scores,
    Tensor* dev_boxes, Tensor* dev_boxes_keep_flags, int num_images,
    int conv_layer_nboxes, size_t cub_temp_storage_bytes,
    int num_boxes_to_generate, int box_dim) {
  auto d = context->eigen_gpu_device();
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images, conv_layer_nboxes}),
      d_conv_layer_indexes));
  TF_RETURN_IF_ERROR(ResetTensor<int>(d_conv_layer_indexes, d));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images + 1}), d_image_offset));
  TF_RETURN_IF_ERROR(ResetTensor<int>(d_image_offset, d));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_temp_storage_bytes}),
      d_cub_temp_buffer));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images, conv_layer_nboxes}),
      d_sorted_conv_layer_indexes));
  TF_RETURN_IF_ERROR(ResetTensor<int32>(d_sorted_conv_layer_indexes, d));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_images, conv_layer_nboxes}),
      d_sorted_scores));
  TF_RETURN_IF_ERROR(ResetTensor<float>(d_sorted_scores, d));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT,
      TensorShape({num_images, box_dim * num_boxes_to_generate}), dev_boxes));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_boxes, d));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({num_images, num_boxes_to_generate}),
      dev_boxes_keep_flags));
  TF_RETURN_IF_ERROR(ResetTensor<int8>(dev_boxes_keep_flags, d));
  return Status::OK();
}

// Allocate workspace for NMS operation
Status AllocatePreNMSTempTensors(
    OpKernelContext* context, Tensor* dev_image_prenms_boxes,
    Tensor* dev_image_prenms_scores, Tensor* dev_image_boxes_keep_list,
    Tensor* dev_postnms_rois, Tensor* dev_postnms_rois_probs,
    Tensor* dev_prenms_nboxes, int num_images, int num_boxes_to_generate,
    int box_dim, int post_nms_topn, int pre_nms_topn) {
  auto d = context->eigen_gpu_device();
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({box_dim * num_boxes_to_generate}),
      dev_image_prenms_boxes));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_image_prenms_boxes, d));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_boxes_to_generate}),
      dev_image_prenms_scores));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_image_prenms_scores, d));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes_to_generate}),
      dev_image_boxes_keep_list));
  TF_RETURN_IF_ERROR(ResetTensor<int32>(dev_image_boxes_keep_list, d));

  const int max_postnms_nboxes = std::min(num_boxes_to_generate, post_nms_topn);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT,
      TensorShape({box_dim * num_images * max_postnms_nboxes}),
      dev_postnms_rois));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_postnms_rois, d));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_images * max_postnms_nboxes}),
      dev_postnms_rois_probs));
  TF_RETURN_IF_ERROR(ResetTensor<float>(dev_postnms_rois_probs, d));

  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images}), dev_prenms_nboxes));
  TF_RETURN_IF_ERROR(ResetTensor<int32>(dev_prenms_nboxes, d));

  return Status::OK();
}

// Initialize index and offset arrays.
// num_images is the batch size.
__global__ void InitializeDataKernel(const Cuda2DLaunchConfig config,
                                     int* d_image_offsets,
                                     int* d_boxes_keys_iota) {
  const int image_size = config.virtual_thread_count.x;
  const int num_images = config.virtual_thread_count.y;
  CUDA_AXIS_KERNEL_LOOP(img_idx, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(box_idx, config.virtual_thread_count.x, X) {
      d_boxes_keys_iota[img_idx * image_size + box_idx] = box_idx;

      // One 1D line sets the 1D data
      if (box_idx == 0) {
        d_image_offsets[img_idx] = image_size * img_idx;
        // One thread sets the last+1 offset
        if (img_idx == 0) d_image_offsets[num_images] = image_size * num_images;
      }
    }
  }
}

__device__ inline float devIoU(float const* const a, float const* const b) {
  float left = std::max(a[0], b[0]), right = std::min(a[2], b[2]);
  float top = std::max(a[1], b[1]), bottom = std::min(a[3], b[3]);
  float width = std::max(right - left, 0.f),
        height = std::max(bottom - top, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_reduce_batched(const int* n_boxes_arr,
                                   unsigned long long* dev_mask_cat,
                                   unsigned char* initial_pos_mask,
                                   unsigned char* res_mask_byte_arr,
                                   int mask_dev_stride, int mask_res_stride) {
  int fmap_id = blockIdx.x;
  int tid = threadIdx.x;
  const int n_boxes = n_boxes_arr[fmap_id];  // boxes in current block
  int offset = 0;
  for (int i = 0; i < fmap_id; i++)
    offset += n_boxes_arr[i];  // offset in the masks
  const unsigned long long* dev_mask =
      dev_mask_cat +
      mask_dev_stride * fmap_id;  // pointer to beginning of current mask
  initial_pos_mask += offset;     // lookup index ?
  unsigned char* res_mask_byte = res_mask_byte_arr + offset;

  const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
  // compute largest block we can fit in shared memory
  const unsigned int block_rows_max =
      max_shmem_size / sizeof(unsigned long long) / col_blocks - 1;
  // use intrinsics functions to compute largest block that is power of 2
  // power of 2 helps the main loop to be more efficient
  const unsigned int block_rows =
      1 << (8 * sizeof(unsigned int) - __ffs(__brev(block_rows_max)));
  extern __shared__ unsigned long long mask_buf_sh[];
  unsigned long long* res_mask_sh = mask_buf_sh;
  unsigned long long* mask_block = mask_buf_sh + col_blocks;
  for (int i = tid; i < col_blocks; i += blockDim.x) {
    res_mask_sh[i] = 0;
    for (int j = 0; j < 8 * sizeof(unsigned long long); j++) {  // take 64 bytes
      if ((i * 64 + j) < n_boxes && (initial_pos_mask[i * 64 + j] == 0))
        res_mask_sh[i] |= 1ULL << j;
    }
  }

  __syncthreads();
  unsigned int* mask_block32 = (unsigned int*)mask_block;
  unsigned int* res_mask_sh32 = (unsigned int*)res_mask_sh;
  for (unsigned int i = 0; i < n_boxes; i += block_rows) {
    int num_rows = std::min(n_boxes - i, block_rows);
    int block_max_elements = num_rows * col_blocks;
    for (int j = tid; j < block_max_elements; j += block_size)
      mask_block[j] = dev_mask[i * col_blocks + j];
    __syncthreads();
    int nblock = i / block_size;
    int num_rows_inner_loop;
    for (int k_start = 0; k_start < block_rows; k_start += block_size) {
      num_rows_inner_loop = std::min(num_rows, k_start + block_size) - k_start;
      for (int k = 0; k < num_rows_inner_loop; k++) {
        if (!(res_mask_sh[nblock] & 1ULL << k)) {
          for (int t = tid; t < col_blocks; t += block_size)
            res_mask_sh[t] |= mask_block[(k + k_start) * col_blocks + t];
        }
        __syncthreads();
      }
      nblock++;
    }
  }
  for (int i = tid; i < n_boxes; i += block_size) {
    int nblock = i / block_size;
    int in_block = i % block_size;
    res_mask_byte[i] =
        1 -
        (unsigned char)((res_mask_sh[nblock] & 1ULL << in_block) >> in_block);
  }
}

__global__ void nms_kernel_batched(const int* n_boxes_arr,
                                   const float nms_overlap_thresh,
                                   const float* dev_boxes_cat,
                                   unsigned long long* dev_mask_cat,
                                   int mask_stride) {
  const int tPBlock = threadsPerBlock;
  const int fmap_id = blockIdx.z;    // image
  const int row_start = blockIdx.y;  //
  const int col_start = blockIdx.x;
  const int n_boxes = n_boxes_arr[fmap_id];  // boxes in layer
  int offset = 0;
  for (int i = 0; i < fmap_id; i++) offset += n_boxes_arr[i];
  const float* dev_boxes =
      dev_boxes_cat + offset * 4;  // concatenated boxes array
  unsigned long long* dev_mask =
      dev_mask_cat + mask_stride * fmap_id;  // rounded up bit vector
  const int row_size = std::min(n_boxes - row_start * tPBlock, tPBlock);
  const int col_size = std::min(n_boxes - col_start * tPBlock, tPBlock);
  if (row_size < 0 || col_size < 0) return;
  __shared__ float block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {  // copy a section of boxes to shared memory
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float* cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 4) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

}  // namespace GBBP

class GenerateBoundingBoxProposals : public tensorflow::OpKernel {
 public:
  explicit GenerateBoundingBoxProposals(
      tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("post_nms_topn", &post_nms_topn_));
    OP_REQUIRES(context, post_nms_topn_ > 0,
                errors::InvalidArgument("post_nms_topn can't be 0 or less"));
    bbox_xform_clip_default_ = log(1000.0 / 16.);
  }

  template <typename T>
  Status GetScalarValue(OpKernelContext* context, int input, T* value) {
    const Tensor& scalar_tensor = context->input(input);
    if (!TensorShapeUtils::IsScalar(scalar_tensor.shape())) {
      return errors::InvalidArgument("Expected a scalar in input ", input,
                                     "but got shape ",
                                     scalar_tensor.shape().DebugString());
    }
    *value = scalar_tensor.scalar<T>()();
    return Status::OK();
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    VLOG(1) << "Starting Compute " << name();
    const auto scores = context->input(0);
    const auto bbox_deltas = context->input(1);
    const auto image_info = context->input(2);
    const auto anchors = context->input(3);
    const auto num_images = scores.dim_size(0);
    const auto num_anchors = scores.dim_size(3);
    const auto height = scores.dim_size(1);
    const auto width = scores.dim_size(2);
    const auto box_dim = anchors.dim_size(2) / num_anchors;
    OP_REQUIRES(context, box_dim == 4,
                errors::OutOfRange("Box dimensions need to be 4"));
    // TODO(skama): make sure that inputs are ok.
    const int image_stride = height * width;
    const int conv_layer_nboxes =
        image_stride *
        num_anchors;  // total number of boxes when decoded on anchors.
    // The following calls to CUB primitives do nothing
    // (because the first arg is nullptr)
    // except setting cub_*_temp_storage_bytes
    float nms_threshold;
    int pre_nms_topn;
    float min_size;
    OP_REQUIRES_OK(context, GetScalarValue(context, 4, &nms_threshold));
    if (nms_threshold < 0 || nms_threshold > 1.0) {
      context->SetStatus(errors::InvalidArgument(
          "nms_threshold should be between 0 and 1. Got ", nms_threshold));
      return;
    }
    OP_REQUIRES_OK(context, GetScalarValue(context, 5, &pre_nms_topn));
    if (pre_nms_topn <= 0) {
      context->SetStatus(errors::InvalidArgument(
          "pre_nms_topn should be greater than 0", pre_nms_topn));
      return;
    }
    OP_REQUIRES_OK(context, GetScalarValue(context, 6, &min_size));
    auto cuda_stream = GetGpuStream(context);
    size_t cub_sort_temp_storage_bytes = 0;
    float* flt_ptr = nullptr;
    int* int_ptr = nullptr;
    cudaError_t cuda_ret = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, cub_sort_temp_storage_bytes, flt_ptr, flt_ptr, int_ptr,
        int_ptr, num_images * conv_layer_nboxes, num_images, int_ptr, int_ptr,
        0, 8 * sizeof(float),  // sort all bits
        cuda_stream);
    TF_OP_REQUIRES_CUDA_SUCCESS(context, cuda_ret);
    // get the size of select temp buffer
    size_t cub_select_temp_storage_bytes = 0;
    char* char_ptr = nullptr;
    float4* f4_ptr = nullptr;
    TF_OP_REQUIRES_CUDA_SUCCESS(
        context, cub::DeviceSelect::Flagged(
                     nullptr, cub_select_temp_storage_bytes, f4_ptr, char_ptr,
                     f4_ptr, int_ptr, image_stride * num_anchors, cuda_stream));
    Tensor d_conv_layer_indexes;  // box indices on device
    Tensor d_image_offset;        // starting offsets boxes for each image
    Tensor d_cub_temp_buffer;     // buffer for cub sorting
    Tensor d_sorted_conv_layer_indexes;  // output of cub sorting, indices of
                                         // the sorted boxes
    Tensor dev_sorted_scores;            // sorted scores, cub output
    Tensor dev_boxes;                    // boxes on device
    Tensor dev_boxes_keep_flags;  // bitmask for keeping the boxes or rejecting
                                  // from output
    const int nboxes_to_generate = std::min(conv_layer_nboxes, pre_nms_topn);
    size_t cub_temp_storage_bytes =
        std::max(cub_sort_temp_storage_bytes, cub_select_temp_storage_bytes);
    OP_REQUIRES_OK(
        context,
        GBBP::AllocateGenerationTempTensors(
            context, &d_conv_layer_indexes, &d_image_offset, &d_cub_temp_buffer,
            &d_sorted_conv_layer_indexes, &dev_sorted_scores, &dev_boxes,
            &dev_boxes_keep_flags, num_images, conv_layer_nboxes,
            cub_temp_storage_bytes, nboxes_to_generate, box_dim));
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    Cuda2DLaunchConfig conf2d =
        GetCuda2DLaunchConfig(conv_layer_nboxes, num_images, d);
    // create box indices and offsets for each image on device
    OP_REQUIRES_OK(
        context, GpuLaunchKernel(GBBP::InitializeDataKernel, conf2d.block_count,
                                 conf2d.thread_per_block, 0, d.stream(), conf2d,
                                 d_image_offset.flat<int>().data(),
                                 d_conv_layer_indexes.flat<int>().data()));

    // sort boxes with their scores.
    // d_sorted_conv_layer_indexes will hold the pointers to old indices.
    TF_OP_REQUIRES_CUDA_SUCCESS(
        context,
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            d_cub_temp_buffer.flat<int8>().data(), cub_temp_storage_bytes,
            scores.flat<float>().data(), dev_sorted_scores.flat<float>().data(),
            d_conv_layer_indexes.flat<int>().data(),
            d_sorted_conv_layer_indexes.flat<int>().data(),
            num_images * conv_layer_nboxes, num_images,
            d_image_offset.flat<int>().data(),
            d_image_offset.flat<int>().data() + 1, 0,
            8 * sizeof(float),  // sort all bits
            cuda_stream));
    // Keeping only the topN pre_nms
    conf2d = GetCuda2DLaunchConfig(nboxes_to_generate, num_images, d);

    // create box y1,x1,y2,x2 from box_deltas and anchors (decode the boxes) and
    // mark the boxes which are smaller that min_size ignored.
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            GBBP::GeneratePreNMSUprightBoxesKernel, conf2d.block_count,
            conf2d.thread_per_block, 0, d.stream(), conf2d,
            d_sorted_conv_layer_indexes.flat<int>().data(),
            reinterpret_cast<const float4*>(bbox_deltas.flat<float>().data()),
            reinterpret_cast<const float4*>(anchors.flat<float>().data()),
            height, width, num_anchors, min_size,
            image_info.flat<float>().data(), bbox_xform_clip_default_,
            reinterpret_cast<float4*>(dev_boxes.flat<float>().data()),
            nboxes_to_generate,
            (char*)dev_boxes_keep_flags.flat<int8>().data()));
    const int nboxes_generated = nboxes_to_generate;
    const int roi_cols = box_dim;

    Tensor dev_image_prenms_boxes;
    Tensor dev_image_prenms_scores;
    Tensor dev_image_boxes_keep_list;
    Tensor dev_postnms_rois;
    Tensor dev_postnms_rois_probs;
    Tensor dev_prenms_nboxes;
    // Allocate workspaces needed for NMS
    OP_REQUIRES_OK(
        context, GBBP::AllocatePreNMSTempTensors(
                     context, &dev_image_prenms_boxes, &dev_image_prenms_scores,
                     &dev_image_boxes_keep_list, &dev_postnms_rois,
                     &dev_postnms_rois_probs, &dev_prenms_nboxes, num_images,
                     nboxes_generated, box_dim, post_nms_topn_, pre_nms_topn));
    // get the pointers for temp storages
    int* d_prenms_nboxes = dev_prenms_nboxes.flat<int>().data();
    int h_prenms_nboxes = 0;
    char* d_cub_temp_storage = (char*)d_cub_temp_buffer.flat<int8>().data();
    float* d_image_prenms_boxes = dev_image_prenms_boxes.flat<float>().data();
    float* d_image_prenms_scores = dev_image_prenms_scores.flat<float>().data();
    int* d_image_boxes_keep_list = dev_image_boxes_keep_list.flat<int>().data();

    int nrois_in_output = 0;
    // get the pointers to boxes and scores
    char* d_boxes_keep_flags = (char*)dev_boxes_keep_flags.flat<int8>().data();
    float* d_boxes = dev_boxes.flat<float>().data();
    float* d_sorted_scores = dev_sorted_scores.flat<float>().data();

    // Create output tensors
    Tensor* output_rois = nullptr;
    Tensor* output_roi_probs = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({num_images, post_nms_topn_, roi_cols}),
                       &output_rois));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({num_images, post_nms_topn_}),
                                &output_roi_probs));
    float* d_postnms_rois = (*output_rois).flat<float>().data();
    float* d_postnms_rois_probs = (*output_roi_probs).flat<float>().data();

    gpuEvent_t copy_done;
    gpuEventCreate(&copy_done);

    // Do  per-image nms
    for (int image_index = 0; image_index < num_images; ++image_index) {
      // reset output workspaces
      OP_REQUIRES_OK(context,
                     GBBP::ResetTensor<int32>(&dev_image_boxes_keep_list, d));
      // Sub matrices for current image
      // boxes
      const float* d_image_boxes =
          &d_boxes[image_index * nboxes_generated * box_dim];
      // scores
      const float* d_image_sorted_scores =
          &d_sorted_scores[image_index * image_stride * num_anchors];
      // keep flags
      char* d_image_boxes_keep_flags =
          &d_boxes_keep_flags[image_index * nboxes_generated];

      // Output buffer for image
      float* d_image_postnms_rois =
          &d_postnms_rois[image_index * roi_cols * post_nms_topn_];
      float* d_image_postnms_rois_probs =
          &d_postnms_rois_probs[image_index * post_nms_topn_];

      // Moving valid boxes (ie the ones with d_boxes_keep_flags[ibox] ==
      // true) to the output tensors
      TF_OP_REQUIRES_CUDA_SUCCESS(
          context, cub::DeviceSelect::Flagged(
                       d_cub_temp_storage, cub_temp_storage_bytes,
                       reinterpret_cast<const float4*>(d_image_boxes),
                       d_image_boxes_keep_flags,
                       reinterpret_cast<float4*>(d_image_prenms_boxes),
                       d_prenms_nboxes, nboxes_generated, d.stream()));
      TF_OP_REQUIRES_CUDA_SUCCESS(
          context,
          cub::DeviceSelect::Flagged(
              d_cub_temp_storage, cub_temp_storage_bytes, d_image_sorted_scores,
              d_image_boxes_keep_flags, d_image_prenms_scores, d_prenms_nboxes,
              nboxes_generated, d.stream()));
      d.memcpyDeviceToHost(&h_prenms_nboxes, d_prenms_nboxes, sizeof(int));
      TF_OP_REQUIRES_CUDA_SUCCESS(context,
                                  cudaEventRecord(copy_done, d.stream()));
      TF_OP_REQUIRES_CUDA_SUCCESS(context, cudaEventSynchronize(copy_done));
      // We know prenms_boxes <= topN_prenms, because nboxes_generated <=
      // topN_prenms. Calling NMS on the generated boxes
      const int prenms_nboxes = h_prenms_nboxes;
      int nkeep;
      OP_REQUIRES_OK(context, NmsGpu(d_image_prenms_boxes, prenms_nboxes,
                                     nms_threshold, d_image_boxes_keep_list,
                                     &nkeep, context, post_nms_topn_));
      // All operations done after previous sort were keeping the relative
      // order of the elements the elements are still sorted keep topN <=>
      // truncate the array
      const int postnms_nboxes = std::min(nkeep, post_nms_topn_);
      // Moving the out boxes to the output tensors,
      // adding the image_index dimension on the fly
      CudaLaunchConfig config = GetCudaLaunchConfig(post_nms_topn_, d);
      // make this single kernel
      OP_REQUIRES_OK(
          context,
          GpuLaunchKernel(GBBP::WriteUprightBoxesOutput, config.block_count,
                          config.thread_per_block, 0, d.stream(), config,
                          reinterpret_cast<const float4*>(d_image_prenms_boxes),
                          d_image_prenms_scores, d_image_boxes_keep_list,
                          postnms_nboxes, d_image_postnms_rois,
                          d_image_postnms_rois_probs));
      nrois_in_output += postnms_nboxes;
      TF_OP_REQUIRES_CUDA_SUCCESS(context, cudaGetLastError());
    }
  }

 private:
  int post_nms_topn_;
  float bbox_xform_clip_default_;
};

REGISTER_KERNEL_BUILDER(Name("GenerateBoundingBoxProposals")
                            .Device(tensorflow::DEVICE_GPU)
                            .HostMemory("nms_threshold")
                            .HostMemory("min_size")
                            .HostMemory("pre_nms_topn"),
                        tensorflow::GenerateBoundingBoxProposals);
}  // namespace tensorflow
#endif
