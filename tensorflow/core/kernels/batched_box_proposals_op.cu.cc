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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include <algorithm>
#include <limits>
#include <vector>

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/batched_non_max_suppression_op.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace {

// Decode d_bbox_deltas with respect to anchors into absolute coordinates,
// clipping if necessary. Copy the boxes to new box array and new score array
// If a box is less then min_size, its score will be set to
// std::numerical_limits<float>::min() since scores are already limited by
// sigmoid, this will effectively crop them in nms layer min_size is the lower
// bound of the shortest edge for the boxes to consider. bbox_xform_clip is the
// upper bound of encoded width and height.
__global__ void SelectAndGeneratePreNMSUprightBoxes(
    const Gpu2DLaunchConfig config, const float* d_original_scores,
    const float4* d_bbox_deltas, const float4* d_anchors,
    const int* layer_offsets, const int num_layers, const int num_anchors,
    const float min_size,
    const float* d_img_info_vec,  // Input "image_info" to the op [N,5]
    const float bbox_xform_clip, float4* d_out_boxes,
    float* d_out_unsorted_scores) {
  // constants to calculate offsets in to the input and output arrays.
  int image_offset = 0;
  for (int i = 0; i < num_layers; ++i) {
    image_offset += layer_offsets[i];
  }
  CUDA_AXIS_KERNEL_LOOP(image_index, config.virtual_thread_count.y, Y) {
    const float4* curr_boxes = d_bbox_deltas + image_index * image_offset;
    const float* curr_scores = d_original_scores + image_index * image_offset;
    /// All layer boxes are identical here so we will loop over all boxes
    CUDA_AXIS_KERNEL_LOOP(ibox, config.virtual_thread_count.x, X) {
      // box_conv_index : # of the same box, but indexed in the
      // scores from the conv layer, of shape (height,width,num_anchors) the
      // num_images dimension was already removed box_conv_index =
      // a*image_stride + h*width + w
      // const int box_conv_index =image_index *image_stride+ibox
      // float curr_score=d_original_scores[box_conv_index];

      // float4 is a struct with float x,y,z,w
      const float4 anchor = d_anchors[ibox];
      // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
      float x1 = anchor.y;
      float x2 = anchor.w;
      float y1 = anchor.x;
      float y2 = anchor.z;

      // TODO use fast math when possible

      // Deltas of shape (N,height,width,num_anchors x 4)
      // int deltas_idx = box_conv_index + image_index * image_stride;
      float4 deltas = curr_boxes[ibox];
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

      d_out_boxes[ibox] = {x1, y1, x2, y2};
      if (keep_box) {
        d_out_unsorted_scores[image_index * image_offset * num_anchors + ibox] =
            curr_scores[ibox];
      } else {
        d_out_unsorted_scores[image_index * image_offset * num_anchors + ibox] =
            std::numeric_limits<float>::lowest();
        printf("El %d %d\n", image_index, ibox);
      }
    }
  }
}

template <typename Index, typename T, typename... Args>
__global__ void BatchedIndexGather(const int* num_elements,
                                   const int* input_strides,
                                   const int* output_strides,
                                   const int batch_size, const int index_offset,
                                   const Index* indices, const T* original,
                                   T* selected, Args... args) {
  for (const int y : GpuGridRangeY(batch_size)) {
    int istride = input_strides[y];
    int ostride = output_strides[y];
    for (const int idx : GpuGridRangeX(num_elements[y])) {
      // printf("%d %d %d %d %d\n", y, idx, ostride, istride,
      //        indices[idx + ostride]);
      SelectHelper(idx + ostride, istride + indices[idx + y * index_offset],
                   original, selected, args...);
    }
  }
}

__global__ void SetupSelectionIndices(const int* num_selected,
                                      const int num_batches,
                                      const int num_levels,
                                      int* begin_end_offsets) {
  // fill the begin end offsets
  // first num_batches*num_levels = cumsum(selected)
  // second num_batches*num_levels =cumsum(selected)+num_elements[i]
  // next num_batches = sum_{num_levels}(num_selected)
  // next num_batches = sum_{num_levels}(num_selected)+num_elements in
  // curr_batch
  // next num_batches, number of entries in a batch
  for (int y : GpuGridRangeY(num_batches)) {
    int batch_start_offset = 0;  // batch start
    for (int b = 0; b < y; ++b) {
      for (int l = 0; l < num_levels; ++l)
        batch_start_offset += num_selected[b * num_levels + l];
    }
    for (int x : GpuGridRangeX(num_levels)) {
      int level_offset = 0;
      for (int l = 0; l < x; ++l) {
        level_offset += num_selected[y * num_levels + l];
      }
      begin_end_offsets[y * num_levels + x] = batch_start_offset + level_offset;
      begin_end_offsets[y * num_levels + x + num_batches * num_levels] =
          batch_start_offset + level_offset + num_selected[y * num_levels + x];
    }
    if (threadIdx.x == 0) {
      int in_batch = 0;
      for (int i = 0; i < num_levels; ++i) {
        in_batch += num_selected[y * num_levels + i];
      }
      begin_end_offsets[num_batches * num_levels * 2 + y] = batch_start_offset;
      begin_end_offsets[num_batches * num_levels * 2 + num_batches + y] =
          batch_start_offset + in_batch;
      begin_end_offsets[num_batches * num_levels * 2 + 2 * num_batches + y] =
          in_batch;
    }
  }
}
// Copy the selected boxes and scores to output tensors.
//
__global__ void CopyTopKtoOutput(const int batch_size, int* d_selected_counts,
                                 int* d_batch_offsets, int max_boxes,
                                 const float4* d_boxes, const float* d_scores,
                                 float4* d_output_boxes,
                                 float* d_output_scores) {
  for (int batch : GpuGridRangeY(batch_size)) {
    const float* curr_scores = d_scores + d_batch_offsets[batch];
    const float4* curr_boxes = d_boxes + d_batch_offsets[batch];
    float* out_scores = d_output_scores + batch * max_boxes;
    float4* out_boxes = d_output_boxes + batch * max_boxes;
    int num_selected_boxes = d_selected_counts[batch];
    for (int box : GpuGridRangeX(max_boxes)) {
      if (box < num_selected_boxes) {
        out_scores[box] = curr_scores[box];
        out_boxes[box] = curr_boxes[box];
      } else {
        out_scores[box] = 0.0;
        out_boxes[box] = {0., 0., 0., 0.};
      }
    }
  }
}
}  // namespace

class BatchedBoxProposals : public tensorflow::OpKernel {
 public:
  explicit BatchedBoxProposals(tensorflow::OpKernelConstruction* context)
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
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Starting Compute " << name();
    }
    // multi-level boxes,scores and anchors for inputs
    // at each level , B,H,W,A*4 is reshaped to B,H*W,A4
    // Anchors are shaped as [H*W,A*4]
    // Scores are shaped as [B,H*L,A]
    const auto scores = context->input(0);
    const auto bbox_deltas = context->input(1);
    const auto image_info = context->input(2);
    const auto anchors = context->input(3);
    const auto num_entries_per_level = context->input(4);
    int num_images = scores.dim_size(0);
    int num_anchors = scores.dim_size(2);
    int num_levels = num_entries_per_level.dim_size(0);
    int box_dim = anchors.dim_size(1) / num_anchors;
    float nms_threshold;
    int pre_nms_topn;
    float min_size;
    OP_REQUIRES(context, box_dim == 4,
                errors::OutOfRange("Box dimensions need to be 4"));

    OP_REQUIRES_OK(context, GetScalarValue(context, 5, &nms_threshold));
    if (nms_threshold < 0 || nms_threshold > 1.0) {
      context->SetStatus(errors::InvalidArgument(
          "nms_threshold should be between 0 and 1. Got ", nms_threshold));
      return;
    }
    OP_REQUIRES_OK(context, GetScalarValue(context, 6, &pre_nms_topn));
    if (pre_nms_topn <= 0) {
      context->SetStatus(errors::InvalidArgument(
          "pre_nms_topn should be greater than 0", pre_nms_topn));
      return;
    }
    OP_REQUIRES_OK(context, GetScalarValue(context, 7, &min_size));
    auto d = context->eigen_gpu_device();
    Tensor d_decoded_boxes, d_selected_boxes, d_filtered_scores,
        d_selected_scores, d_layer_counts_tensor, d_output_indices_tensor,
        h_layer_counts_tensor;

    OP_REQUIRES_OK(
        context, context->allocate_temp(bbox_deltas.dtype(),
                                        bbox_deltas.shape(), &d_decoded_boxes));

    OP_REQUIRES_OK(context, context->allocate_temp(bbox_deltas.dtype(),
                                                   bbox_deltas.shape(),
                                                   &d_selected_boxes));

    OP_REQUIRES_OK(context,
                   context->allocate_temp(scores.dtype(), scores.shape(),
                                          &d_filtered_scores));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(scores.dtype(), scores.shape(),
                                          &d_selected_scores));
    // layer_counts,start_offset,end_offset +  3 x num_images for batch offset,
    // batch end and batch counts
    int indices_count = num_levels * num_images * 6 + 3 * num_images;
    OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_INT32,
                                                   TensorShape({indices_count}),
                                                   &d_layer_counts_tensor));
    // OP_REQUIRES_OK(
    //     context, context->allocate_temp(DataType::DT_INT32,
    //                                     TensorShape({num_levels *
    //                                     num_images}), d_output_indices));

    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    alloc_attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(DataType::DT_INT32, TensorShape({indices_count}),
                               &h_layer_counts_tensor, alloc_attr));
    Tensor per_image_box_counts_tensor_h;
    OP_REQUIRES_OK(
        context, context->allocate_temp(
                     DataType::DT_INT32, TensorShape({num_images * num_levels}),
                     &per_image_box_counts_tensor_h, alloc_attr));

    int32_t* per_image_box_counts =
        per_image_box_counts_tensor_h.flat<int32_t>().data();
    const int32_t* nboxes_per_level =
        num_entries_per_level.flat<int32_t>().data();
    std::vector<int32_t> box_counts_and_offsets(num_levels);
    int* h_orig_layer_counts = h_layer_counts_tensor.flat<int>().data();
    int* h_orig_layer_begin = h_orig_layer_counts + num_images * num_levels;
    int* h_orig_layer_end = h_orig_layer_begin + num_images * num_levels;
    int* h_nms_io_counts = h_orig_layer_end + num_images * num_levels;

    int* d_layer_counts = d_layer_counts_tensor.flat<int>().data();
    int total_num_boxes_per_level = 0;
    int global_offset = 0;
    for (int batch = 0; batch < num_images; ++batch) {
      for (int level = 0; level < num_levels; ++level) {
        h_orig_layer_counts[batch * num_images + level] =
            nboxes_per_level[level] * num_anchors;
        h_nms_io_counts[batch * num_images + level] =
            nboxes_per_level[level] * num_anchors;
        h_orig_layer_begin[batch * num_images + level] = global_offset;
        global_offset += nboxes_per_level[level] * num_anchors;
        h_orig_layer_end[batch * num_images + level] =
            h_orig_layer_begin[batch * num_images + level] +
            nboxes_per_level[level] * num_anchors;
        per_image_box_counts[batch * num_images + level] =
            nboxes_per_level[level] * num_anchors;
      }
    }
    for (int level = 0; level < num_levels; ++level)
      total_num_boxes_per_level += nboxes_per_level[level] * num_anchors;
    // for (int i = 0; i < num_levels; ++i) {
    //   for (int b = 0; b < num_images; ++b) {
    //     per_image_box_counts[b * num_levels + i] =
    //         nboxes_per_level[i] * num_anchors;
    //     h_orig_layer_counts[b*num_levels+i]=per_image_box_counts[b*num_levels+i];
    //     h_nms_io_counts[b*num_levels+i]=per_image_box_counts[b*num_levels+i]
    //   }
    //   total_num_boxes_per_level += nboxes_per_level[i] * num_anchors;
    // }
    int* d_orig_layer_counts = d_layer_counts_tensor.flat<int>().data();
    int* d_orig_layer_offsets_begin =
        d_orig_layer_counts + num_images * num_levels;
    int* d_nms_io_counts = d_orig_layer_counts + 3 * num_images * num_levels;
    d.memcpyHostToDevice(d_layer_counts, h_orig_layer_counts,
                         h_layer_counts_tensor.NumElements() * sizeof(int32_t));
    auto conf2d =
        GetGpu2DLaunchConfig(total_num_boxes_per_level, num_images, d);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            SelectAndGeneratePreNMSUprightBoxes, conf2d.block_count,
            conf2d.thread_per_block, 0, d.stream(), conf2d,
            scores.flat<float>().data(),
            reinterpret_cast<const float4*>(bbox_deltas.flat<float>().data()),
            reinterpret_cast<const float4*>(anchors.flat<float>().data()),
            d_layer_counts, num_levels, num_anchors, min_size,
            image_info.flat<float>().data(), bbox_xform_clip_default_,
            reinterpret_cast<float4*>(d_decoded_boxes.flat<float>().data()),
            d_filtered_scores.flat<float>().data()));
    // CheckKernel(context, "SelectAndGeneratePeNMS");
    Tensor* d_output_indices = &d_output_indices_tensor;
    OP_REQUIRES_OK(context, DoNMSBatchedGPUJagged(
                                context, d_decoded_boxes, d_filtered_scores,
                                per_image_box_counts_tensor_h, post_nms_topn_,
                                nms_threshold, -10000., true, d_nms_io_counts,
                                &d_output_indices, 0, false));
    // CheckKernel(context, "DoNMSBatchedGPUJagged");
    if (VLOG_IS_ON(3)) {
      LOG(INFO) << dumpDeviceTensor<int>("After DoNMS", d_layer_counts_tensor,
                                         context);
    }
    int* d_nms_layer_offsets_begin = d_nms_io_counts + num_images * num_levels;
    int* d_nms_layer_offsets_end =
        d_nms_layer_offsets_begin + num_images * num_levels;

    conf2d = GetGpu2DLaunchConfig(num_levels, num_images, d);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(SetupSelectionIndices, conf2d.block_count,
                        conf2d.thread_per_block, 0, d.stream(), d_nms_io_counts,
                        num_images, num_levels, d_nms_layer_offsets_begin));

    conf2d = GetGpu2DLaunchConfig(num_levels * post_nms_topn_, num_images, d);
    if (VLOG_IS_ON(3)) {
      LOG(INFO) << dumpDeviceTensor<int>("SetupSelectionIndices",
                                         d_layer_counts_tensor, context);
    }  // select the boxes and scores using nms result from nms inputs!
    if (VLOG_IS_ON(5)) {
      LOG(INFO) << dumpDeviceTensor<int>("NMSSelected", d_output_indices_tensor,
                                         context);
    }
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(
            BatchedIndexGather<int, float4, const float*, float*>,
            conf2d.block_count, conf2d.thread_per_block, 0, d.stream(),
            /*entry_counts*/ d_nms_io_counts,
            /*input offsets*/ d_orig_layer_offsets_begin,
            /*output offsets*/ d_nms_layer_offsets_begin,
            /*batch size*/ num_levels * num_images,
            /* index_offset*/ post_nms_topn_,
            /*selection index*/ d_output_indices->flat<int>().data(),
            /*original source*/
            reinterpret_cast<const float4*>(
                d_decoded_boxes.flat<float>().data()),
            /*destination*/
            reinterpret_cast<float4*>(d_selected_boxes.flat<float>().data()),
            /*original*/ scores.flat<float>().data(),
            /*destination*/ d_filtered_scores.flat<float>().data()));
    // CheckKernel(context, "PostNMS Box gathering");
    if (VLOG_IS_ON(3)) {
      LOG(INFO) << dumpDeviceTensor<int>(
          "BNMSOutput output d_layer_counts_tensor", d_layer_counts_tensor,
          context);
    }
    // TopK
    size_t cub_temp_storage_bytes = 0;
    cudaError_t cuda_ret = cudaSuccess;
    auto cuda_stream = GetGpuStream(context);
    // Calling cub with nullptrs as inputs will make it return
    // workspace size needed for the operation instead of doing the operation.
    // In this specific instance, cub_sort_storage_bytes will contain the
    // necessary workspace size for sorting after the call.
    cuda_ret = gpuprim::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, cub_temp_storage_bytes,
        static_cast<float*>(nullptr),     // selected scores
        static_cast<float*>(nullptr),     // sorted scores
        static_cast<float4*>(nullptr),    // selected Boxes
        static_cast<float4*>(nullptr),    // sorted Boxes
        d_output_indices->NumElements(),  // Total number of boxes in batch
        num_images,  // num segments, since we are doing topK on all levels, it
                     // is per image now
        static_cast<int*>(nullptr), static_cast<int*>(nullptr), 0,
        8 * sizeof(float),  // sort all bits
        cuda_stream);
    if (cuda_ret != cudaSuccess) {
      context->SetStatus(errors::Internal(
          "Topk sorting could not launch "
          "gpuprim::DeviceSegmentedRadixSort::SortPairsDescending to sort "
          "selected indices, "
          "temp_storage_bytes: ",
          cub_temp_storage_bytes, ", status: ", cudaGetErrorString(cuda_ret)));
      return;
    }

    Tensor d_cub_temp_buffer;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataType::DT_INT8,
                                TensorShape({(int64)cub_temp_storage_bytes}),
                                &d_cub_temp_buffer));
    // reuse decoded_boxes buffer for sorted boxes
    float4* sorted_boxes =
        reinterpret_cast<float4*>(d_decoded_boxes.flat<float>().data());
    int* selected_batch_offsets = d_nms_io_counts + 3 * num_images * num_levels;
    int* selected_batch_counts =
        d_nms_io_counts + 3 * num_images * num_levels + 2 * num_images;
    // sort all batches independently
    cuda_ret = gpuprim::DeviceSegmentedRadixSort::SortPairsDescending(
        d_cub_temp_buffer.flat<int8>().data(), cub_temp_storage_bytes,
        d_filtered_scores.flat<float>().data(),  // selected scores
        d_selected_scores.flat<float>().data(),  // sorted scores
        reinterpret_cast<const float4*>(d_selected_boxes.flat<float>().data()),
        sorted_boxes,
        d_output_indices->NumElements(),  // Total number of boxes in batch
        num_images,  // num segments, since we are doing topK on all levels,
                     // it is per image now
        selected_batch_offsets, selected_batch_offsets + num_images, 0,
        8 * sizeof(float),  // sort all bits
        cuda_stream);
    if (cuda_ret != cudaSuccess) {
      context->SetStatus(errors::Internal(
          "Topk sorting could not launch "
          "gpuprim::DeviceSegmentedRadixSort::SortPairsDescending to sort "
          "selected indices, "
          "temp_storage_bytes: ",
          cub_temp_storage_bytes, ", status: ", cudaGetErrorString(cuda_ret)));
      return;
    }

    Tensor* output_rois = nullptr;
    Tensor* output_roi_probs = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({num_images, post_nms_topn_, 4}),
                                &output_rois));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({num_images, post_nms_topn_}),
                                &output_roi_probs));
    float4* d_postnms_rois =
        reinterpret_cast<float4*>((*output_rois).flat<float>().data());
    float* d_postnms_rois_probs = (*output_roi_probs).flat<float>().data();
    conf2d = GetGpu2DLaunchConfig(post_nms_topn_, num_images, d);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(CopyTopKtoOutput, conf2d.block_count,
                        conf2d.thread_per_block, 0, d.stream(), num_images,
                        selected_batch_counts, selected_batch_offsets,
                        post_nms_topn_, sorted_boxes,
                        d_selected_scores.flat<float>().data(), d_postnms_rois,
                        d_postnms_rois_probs));
    // CheckKernel(context, "CopyTopKtoOutput");
  }

 private:
  int post_nms_topn_;
  float bbox_xform_clip_default_;
};  // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("BatchedBoxProposals")
                            .Device(tensorflow::DEVICE_GPU)
                            .HostMemory("entries_per_level")
                            .HostMemory("nms_threshold")
                            .HostMemory("min_size")
                            .HostMemory("pre_nms_topn"),
                        tensorflow::BatchedBoxProposals);
}  // namespace tensorflow
#endif
