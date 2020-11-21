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
#include "tensorflow/core/kernels/gpu_type_helpers.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define CHECK_KERNEL(x, y)         \
  if (VLOG_IS_ON(2)) {             \
    BatchedNMS::CheckKernel(x, y); \
  }
namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace BatchedBoxProps {
using namespace BatchedNMS;
using namespace GpuTypeHelpers;

template <typename T>
__device__ EIGEN_STRONG_INLINE T get_lowest();
template <>
__device__ EIGEN_STRONG_INLINE float get_lowest<float>() {
  return std::numeric_limits<float>::lowest();
};

// Eigen std::numeric_limits<Eigen::half> is defined for cpu only
template <>
__device__ EIGEN_STRONG_INLINE Eigen::half get_lowest<Eigen::half>() {
  return Eigen::half_impl::raw_uint16_to_half(0xfbff);
};

template <bool with_offset, typename T, typename U>
__device__ EIGEN_STRONG_INLINE T decode_box(const T& anchor, const T& deltas,
                                            const float BBOX_XFORM_CLIP);
template <>
__device__ EIGEN_STRONG_INLINE Box decode_box<true, Box, float>(
    const Box& anchor, const Box& deltas, const float BBOX_XFORM_CLIP) {
  float anchor_xmin = anchor.y1;
  float anchor_xmax = anchor.y2;
  float anchor_ymin = anchor.x1;
  float anchor_ymax = anchor.x2;
  float dx = deltas.y1;
  float dy = deltas.x1;
  float dw = fmin(deltas.y2, BBOX_XFORM_CLIP);
  float dh = fmin(deltas.x2, BBOX_XFORM_CLIP);

  float anchor_h = anchor_ymax - anchor_ymin + 1.0;
  float anchor_w = anchor_xmax - anchor_xmin + 1.0;
  float anchor_xc = anchor_xmin + 0.5 * anchor_w;
  float anchor_yc = anchor_ymin + 0.5 * anchor_h;

  float decoded_boxes_yc = dy * anchor_h + anchor_yc;
  float decoded_boxes_xc = dx * anchor_w + anchor_xc;
  float decoded_boxes_h = expf(dh) * anchor_h;
  float decoded_boxes_w = expf(dw) * anchor_w;

  float decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h;
  float decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w;
  float decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h - 1.0;
  float decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w - 1.0;
  return {decoded_boxes_xmin, decoded_boxes_ymin, decoded_boxes_xmax,
          decoded_boxes_ymax};
}

template <>
__device__ EIGEN_STRONG_INLINE Box decode_box<false, Box, float>(
    const Box& anchor, const Box& deltas, const float BBOX_XFORM_CLIP) {
  const float anchor_xmin = anchor.y1;
  const float anchor_xmax = anchor.y2;
  const float anchor_ymin = anchor.x1;
  const float anchor_ymax = anchor.x2;
  const float dx = deltas.y1;
  const float dy = deltas.x1;
  const float dw = fmin(deltas.y2, BBOX_XFORM_CLIP);
  const float dh = fmin(deltas.x2, BBOX_XFORM_CLIP);

  const float anchor_h = anchor_ymax - anchor_ymin;
  const float anchor_w = anchor_xmax - anchor_xmin;
  const float anchor_xc = anchor_xmin + 0.5 * anchor_w;
  const float anchor_yc = anchor_ymin + 0.5 * anchor_h;

  const float decoded_boxes_yc = dy * anchor_h + anchor_yc;
  const float decoded_boxes_xc = dx * anchor_w + anchor_xc;
  const float decoded_boxes_h = expf(dh) * anchor_h;
  const float decoded_boxes_w = expf(dw) * anchor_w;

  float decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h;
  float decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w;
  float decoded_boxes_ymax = decoded_boxes_yc + 0.5 * decoded_boxes_h;
  float decoded_boxes_xmax = decoded_boxes_xc + 0.5 * decoded_boxes_w;
  return {decoded_boxes_xmin, decoded_boxes_ymin, decoded_boxes_xmax,
          decoded_boxes_ymax};
}

template <>
__device__ EIGEN_STRONG_INLINE HalfBox decode_box<true, HalfBox, Eigen::half>(
    const HalfBox& anchor, const HalfBox& deltas, const float BBOX_XFORM_CLIP) {
  float anchor_xmin = asType<float>(anchor.y1);
  float anchor_xmax = asType<float>(anchor.y2);
  float anchor_ymin = asType<float>(anchor.x1);
  float anchor_ymax = asType<float>(anchor.x2);
  float dx = asType<float>(deltas.y1);
  float dy = asType<float>(deltas.x1);
  float dw = min(asType<float>(deltas.y2), BBOX_XFORM_CLIP);
  float dh = min(asType<float>(deltas.x2), BBOX_XFORM_CLIP);
  float anchor_h = anchor_ymax - anchor_ymin + 1.0f;
  float anchor_w = anchor_xmax - anchor_xmin + 1.0f;
  float anchor_xc = anchor_xmin + 0.5f * anchor_w;
  float anchor_yc = anchor_ymin + 0.5f * anchor_h;
  float decoded_boxes_yc = dy * anchor_h + anchor_yc;
  float decoded_boxes_xc = dx * anchor_w + anchor_xc;
  float decoded_boxes_h = exp(dh) * anchor_h;
  float decoded_boxes_w = exp(dw) * anchor_w;
  float decoded_boxes_ymin =
      decoded_boxes_yc - 0.5f * decoded_boxes_h;
  float decoded_boxes_xmin =
      decoded_boxes_xc - 0.5f * decoded_boxes_w;
  float decoded_boxes_ymax =
     decoded_boxes_ymin + decoded_boxes_h - 1.0f;
  float decoded_boxes_xmax =
      decoded_boxes_xmin + decoded_boxes_w - 1.0f;
  return {asType<Eigen::half>(decoded_boxes_xmin), asType<Eigen::half>(decoded_boxes_ymin), asType<Eigen::half>(decoded_boxes_xmax),
          asType<Eigen::half>(decoded_boxes_ymax)};
}

template <>
__device__ EIGEN_STRONG_INLINE HalfBox decode_box<false, HalfBox, Eigen::half>(
    const HalfBox& anchor, const HalfBox& deltas, const float BBOX_XFORM_CLIP) {
  float anchor_xmin = asType<float>(anchor.y1);
  float anchor_xmax = asType<float>(anchor.y2);
  float anchor_ymin = asType<float>(anchor.x1);
  float anchor_ymax = asType<float>(anchor.x2);
  float dx = asType<float>(deltas.y1);
  float dy = asType<float>(deltas.x1);
  float dw = min(asType<float>(deltas.y2), BBOX_XFORM_CLIP);
  float dh = min(asType<float>(deltas.x2), BBOX_XFORM_CLIP);
  float anchor_h = anchor_ymax - anchor_ymin;
  float anchor_w = anchor_xmax - anchor_xmin;
  float anchor_xc = anchor_xmin + 0.5f * anchor_w;
  float anchor_yc = anchor_ymin + 0.5f * anchor_h;
  float decoded_boxes_yc = dy * anchor_h + anchor_yc;
  float decoded_boxes_xc = dx * anchor_w + anchor_xc;
  float decoded_boxes_h = exp(dh) * anchor_h;
  float decoded_boxes_w = exp(dw) * anchor_w;

  float decoded_boxes_ymin =
      decoded_boxes_yc - 0.5f * decoded_boxes_h;
  float decoded_boxes_xmin =
      decoded_boxes_xc - 0.5f * decoded_boxes_w;
  float decoded_boxes_ymax =
      decoded_boxes_yc + 0.5f * decoded_boxes_h;
  float decoded_boxes_xmax =
      decoded_boxes_xc + 0.5f * decoded_boxes_w;
  return {asType<Eigen::half>(decoded_boxes_xmin), asType<Eigen::half>(decoded_boxes_ymin), asType<Eigen::half>(decoded_boxes_xmax),
          asType<Eigen::half>(decoded_boxes_ymax)};
}

template <bool, typename B, typename T>
__device__ EIGEN_STRONG_INLINE void clip_legacy(B& b, T width, T height);
template <>
__device__ EIGEN_STRONG_INLINE void clip_legacy<true, Box, float>(
    Box& decoded, float width, float height) {
  decoded.x1 = fmax(fmin(decoded.x1, width - 1.0), 0.0f);
  decoded.y1 = fmax(fmin(decoded.y1, height - 1.0), 0.0f);
  decoded.x2 = fmax(fmin(decoded.x2, width - 1.0), 0.0f);
  decoded.y2 = fmax(fmin(decoded.y2, height - 1.0), 0.0f);
}

template <>
__device__ EIGEN_STRONG_INLINE void clip_legacy<false, Box, float>(
    Box& decoded, float width, float height) {
  decoded.x1 = fmax(fmin(decoded.x1, width), 0.0f);
  decoded.y1 = fmax(fmin(decoded.y1, height), 0.0f);
  decoded.x2 = fmax(fmin(decoded.x2, width), 0.0f);
  decoded.y2 = fmax(fmin(decoded.y2, height), 0.0f);
}

template <>
__device__ EIGEN_STRONG_INLINE void clip_legacy<true, HalfBox, Eigen::half>(
    HalfBox& decoded, Eigen::half width, Eigen::half height) {
  decoded.x1 = max(min(decoded.x1, width - asType<Eigen::half>(1.0f)),
                   asType<Eigen::half>(0.0f));
  decoded.y1 = max(min(decoded.y1, height - asType<Eigen::half>(1.0f)),
                   asType<Eigen::half>(0.0f));
  decoded.x2 = max(min(decoded.x2, width - asType<Eigen::half>(1.0f)),
                   asType<Eigen::half>(0.0f));
  decoded.y2 = max(min(decoded.y2, height - asType<Eigen::half>(1.0f)),
                   asType<Eigen::half>(0.0f));
}

template <>
__device__ EIGEN_STRONG_INLINE void clip_legacy<false, HalfBox, Eigen::half>(
    HalfBox& decoded, Eigen::half width, Eigen::half height) {
  decoded.x1 = max(min(decoded.x1, width), asType<Eigen::half>(0.0f));
  decoded.y1 = max(min(decoded.y1, height), asType<Eigen::half>(0.0f));
  decoded.x2 = max(min(decoded.x2, width), asType<Eigen::half>(0.0f));
  decoded.y2 = max(min(decoded.y2, height), asType<Eigen::half>(0.0f));
}

template <bool, typename B, typename T>
__device__ EIGEN_STRONG_INLINE bool keep_legacy(B& b, T scale, T min_size,
                                                T img_width, T img_height);
template <>
__device__ EIGEN_STRONG_INLINE bool keep_legacy<true, Box, float>(
    Box& b, float scale, float min_size, float img_width, float img_height) {
  float h = b.y2 - b.y1 + 1.0;
  float w = b.x2 - b.x1 + 1.0;
  float xc = b.x1 + w / 2.0;
  float yc = b.y1 + h / 2.0;
  float min_scale = fmax(min_size, 1.0) * scale;
  return (fmin(h, w) >= min_scale) && (xc < img_width) && (yc < img_height);
}

template <>
__device__ EIGEN_STRONG_INLINE bool keep_legacy<false, Box, float>(
    Box& b, float scale, float min_size, float img_width, float img_height) {
  float min_scale = min_size * scale;
  float h = b.y2 - b.y1;
  float w = b.x2 - b.x1;
  return fmin(w, h) >= min_scale;
}

template <>
__device__ EIGEN_STRONG_INLINE bool keep_legacy<true, HalfBox, Eigen::half>(
    HalfBox& b, Eigen::half scale, Eigen::half min_size, Eigen::half img_width,
    Eigen::half img_height) {
  float h  = asType<float>(b.y2) - asType<float>(b.y1) + 1.0f;
  float w  = asType<float>(b.x2) - asType<float>(b.x1) + 1.0f;
  float xc = asType<float>(b.x1) + w / 2.0f;
  float yc = asType<float>(b.y1) + h / 2.0f;
  float min_scale = max(asType<float>(min_size), 1.0f) * asType<float>(scale);
  return (min(h, w) >= min_scale) && (xc < img_width) && (yc < img_height);
}

template <>
__device__ EIGEN_STRONG_INLINE bool keep_legacy<false, HalfBox, Eigen::half>(
    HalfBox& b, Eigen::half scale, Eigen::half min_size, Eigen::half img_width,
    Eigen::half img_height) {
  float min_scale = asType<float>(min_size) * asType<float>(scale);
  float h = asType<float>(b.y2) - asType<float>(b.y1);
  float w = asType<float>(b.x2) - asType<float>(b.x1);
  return min(w, h) >= min_scale;
}
// Decode d_bbox_deltas with respect to anchors into absolute coordinates,
// clipping if necessary. Copy the boxes to new box array and new score array
// If a box is less then min_size, its score will be set to
// std::numerical_limits<float>::min() since scores are already limited by
// sigmoid, this will effectively crop them in nms layer min_size is the lower
// bound of the shortest edge for the boxes to consider. bbox_xform_clip is the
// upper bound of encoded width and height.
template <bool use_legacy_offset, typename B, typename T>
__global__ void SelectAndGeneratePreNMSUprightBoxes(
    const int batch_size, const T* d_original_scores, const B* d_bbox_deltas,
    const B* d_anchors, const int* argsort_indices, const int* layer_counts,
    const int num_layers, const int num_anchors, int max_selection,
    const T min_size,
    const T* d_img_info_vec,  // Input "image_info" to the op [N,5]
    const float bbox_xform_clip, B* d_out_boxes, T* d_out_scores) {
  // constants to calculate offsets in to the input and output arrays.
  int image_offset = 0;
  int out_offset = 0;
  for (int i = 0; i < num_layers; ++i) {
    image_offset += layer_counts[i];
    out_offset += min(layer_counts[i], max_selection);
  }

  for (int image_index : GpuGridRangeY(batch_size)) {
    const B* curr_boxes = d_bbox_deltas + image_index * image_offset;
    const T* curr_scores = d_original_scores + image_index * image_offset;
    /// All layer boxes are identical here so we will loop over all boxes
    B* image_out_boxes = d_out_boxes + image_index * out_offset;
    T* image_out_scores = d_out_scores + image_index * out_offset;
    const int* curr_indices = argsort_indices + image_index * image_offset;
    const B* curr_anchors = d_anchors;  // anchors are per level
    for (int level = 0; level < num_layers; ++level) {
      int num_select = min(max_selection, layer_counts[level]);
      for (int ibox : GpuGridRangeX(num_select)) {
        // if (threadIdx.x == 0) {
        //   printf(
        //       "x=%d y=%d bx=%d by=%d b=%d l=%d ns=%d cb=%p cs=%p ob=%p os=%p
        //       " "ci=%p ca=%p i=%d oi=%d\n", threadIdx.x, threadIdx.y,
        //       blockIdx.x, blockIdx.y, image_index, level, num_select,
        //       curr_boxes, curr_scores, image_out_boxes, image_out_scores,
        //       curr_indices, curr_anchors, ibox, curr_indices[ibox]);
        // }
        // box_conv_index : # of the same box, but indexed in the
        // scores from the conv layer, of shape (height,width,num_anchors) the
        // num_images dimension was already removed box_conv_index =
        // a*image_stride + h*width + w
        // const int box_conv_index =image_index *image_stride+ibox
        // float curr_score=d_original_scores[box_conv_index];
        int orig_index = curr_indices[ibox];
        // float4 is a struct with float x,y,z,w
        // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
        // maskrcnn_layout is y1,x1,y2,x2
        // this function returns the data in x1,x2,y1,y2 order
        B decoded = BatchedBoxProps::decode_box<use_legacy_offset,B,T>(
            curr_anchors[orig_index], curr_boxes[orig_index], bbox_xform_clip);
        const T img_height = d_img_info_vec[5 * image_index + 0];
        const T img_width = d_img_info_vec[5 * image_index + 1];
        BatchedBoxProps::clip_legacy<use_legacy_offset, B, T>(
            decoded, img_width, img_height);
        bool keep_box = BatchedBoxProps::keep_legacy<use_legacy_offset, B, T>(
            decoded, d_img_info_vec[5 * image_index + 2], min_size, img_width,
            img_height);

        image_out_boxes[ibox] = decoded;
        // minsize elimination could be a problem since it may reduce the prenms
        // box count if a high scoring box is eliminated but not sure how much
        // that will effect the end result.
        // if (threadIdx.x == 0) {
        //   printf(
        //       "x=%d y=%d bx=%d by=%d b=%d l=%d ns=%d cb=%p cs=%p ob=%p os=%p
        //       " "ci =%p ca=%p i=%d oi =%d bx1=%6.3f by1=%6.3f bx2=%6.3f "
        //       "by2=%6.3f\n",
        //       threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, image_index,
        //       level, num_select, curr_boxes, curr_scores, image_out_boxes,
        //       image_out_scores, curr_indices, curr_anchors, ibox,
        //       curr_indices[ibox], decoded.x1, decoded.y1, decoded.x2,
        //       decoded.y2);
        // }

        if (keep_box) {
          image_out_scores[ibox] = curr_scores[ibox];
        } else {
          image_out_scores[ibox] = get_lowest<T>();
          printf("El %d %d\n", image_index, ibox);
        }
      }
      curr_boxes += layer_counts[level];
      curr_scores += layer_counts[level];
      curr_indices += layer_counts[level];
      curr_anchors += layer_counts[level];
      image_out_boxes += min(layer_counts[level], max_selection);
      image_out_scores += min(layer_counts[level], max_selection);
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
      BatchedNMS::SelectHelper(idx + ostride,
                               istride + indices[idx + y * index_offset],
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
template <typename B, typename T>
__global__ void CopyTopKtoOutput(const int batch_size, int* d_selected_counts,
                                 int* d_batch_offsets, int max_boxes,
                                 const B* d_boxes, const T* d_scores,
                                 B* d_output_boxes, T* d_output_scores) {
  for (int batch : GpuGridRangeY(batch_size)) {
    const T* curr_scores = d_scores + d_batch_offsets[batch];
    const B* curr_boxes = d_boxes + d_batch_offsets[batch];
    T* out_scores = d_output_scores + batch * max_boxes;
    B* out_boxes = d_output_boxes + batch * max_boxes;
    int num_selected_boxes = d_selected_counts[batch];
    for (int box : GpuGridRangeX(max_boxes)) {
      if (box < num_selected_boxes) {
        out_scores[box] = curr_scores[box];
        auto& ob = out_boxes[box];
        const auto& cb = curr_boxes[box];
        ob.x1 = cb.y1;
        ob.y1 = cb.x1;
        ob.y2 = cb.x2;
        ob.x2 = cb.y2;
      } else {
        out_scores[box] = asType<T>(0.0f);
        out_boxes[box] = {asType<T>(0.0f), asType<T>(0.0f), asType<T>(0.0f),
                          asType<T>(0.0f)};
      }
    }
  }
}
//
//  Generate indices for sorting each segment
// {0,1,2,3,4,5,0,1,2,3,0,1} for a bs=1 num_levels=3 and level counts[6,4,2]
//
__global__ void GenerateIndices(const int batch_size, const int num_levels,
                                const int* level_counts, int* output) {
  int batch_offset = 0;
  for (int i = 0; i < num_levels; ++i) batch_offset += level_counts[i];
  for (int batch : GpuGridRangeY(batch_size)) {
    int* curr_batch = output + batch_offset * batch;
    for (int l = 0; l < num_levels; ++l) {
      for (int i : GpuGridRangeX(level_counts[l])) {
        curr_batch[i] = i;
      }
      curr_batch += level_counts[l];
    }
  }
}
template <typename T>
__device__ EIGEN_STRONG_INLINE T Reorder(const T& t) {
  return t;
}
template <>
__device__ EIGEN_STRONG_INLINE Box Reorder(const Box& t) {
  return {t.y1, t.x1, t.y2, t.x2};
}

template <>
__device__ EIGEN_STRONG_INLINE HalfBox Reorder(const HalfBox& t) {
  return {t.y1, t.x1, t.y2, t.x2};
}
// this is inefficient but this is debug code so do not matter much
template <typename T>
__global__ void ScatterOutputs(const T* input, int* num_outputs, int batch_size,
                               int num_levels, int max_size, void* outputs) {
  for (int level : GpuGridRangeY(num_levels)) {
    T* dest = ((T**)outputs)[level];  // one tensor per level
    for (int batch = 0; batch < batch_size; ++batch) {
      T* batch_dest = dest + batch * max_size;
      int n_entries = num_outputs[batch * num_levels + level];
      int offset = 0;
      for (int i = 0; i < batch * num_levels + level; ++i) {
        offset += num_outputs[i];
      }
      const T* src = input + offset;
      int num_zero_elements = max_size * sizeof(T) / sizeof(float);
      float* zdest = reinterpret_cast<float*>(batch_dest);
      // if (threadIdx.x == 0) {
      //   printf("x=%d y=%d bx=%d by=%d b=%d d=%p s=%p z=%p o=%d e=%d n=%d\n",
      //          threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, batch,
      //          (void*)dest, (void*)src, (void*)zdest, offset, n_entries,
      //          num_zero_elements);
      // }
      for (int i : GpuGridRangeX(num_zero_elements)) {
        zdest[i] = 0.;
      }
      for (int i : GpuGridRangeX(n_entries)) {
        batch_dest[i] = BatchedBoxProps::Reorder<T>(src[i]);
      }
    }
  }
}

// template <>
// __global__ void ScatterOutputs<float4>(const float4* input, int* num_outputs,
//                                        int batch_size, int max_size,
//                                        void* outputs) {
//   for (int batch : GpuGridRangeY(batch_size)) {
//     float4* dest = ((float4**)outputs)[batch];
//     int n_entries = num_outputs[batch];
//     int offset = 0;
//     for (int i = 0; i < batch; ++i) {
//       offset += num_outputs[i];
//     }
//     const float4* src = input + offset;
//     int num_zero_elements = max_size * sizeof(float4) / sizeof(float);
//     float* zdest = reinterpret_cast<float*>(dest);
//     if (threadIdx.x == 0) {
//       printf("x=%d y=%d bx=%d by=%d b=%d d=%p s=%p z=%p o=%d e=%d n=%d\n",
//              threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, batch,
//              (void*)dest, (void*)src, (void*)zdest, offset, n_entries,
//              num_zero_elements);
//     }
//     for (int i : GpuGridRangeX(num_zero_elements)) {
//       zdest[i] = 0.;
//     }
//     for (int i : GpuGridRangeX(n_entries)) {
//       auto& db = dest[i];
//       const auto& sb = src[i];
//       db.x = sb.y;
//       db.y = sb.x;
//       db.z = sb.w;
//       db.w = sb.z;
//     }
//   }
// }
class Indexer {
 public:
  Indexer(int batch_size, int num_levels) {}
  Tensor& GetCountsForNMS() { return host_counts; }
  int* GetDeviceLayerCounts() { return nullptr; }

 private:
  Tensor device_counts;
  Tensor host_counts;
};
template <typename B, typename T>
Status BatchedBoxProposalsFunctor(
    OpKernelContext* context, const Tensor& scores, const Tensor& bbox_deltas,
    const Tensor& image_info, const Tensor& anchors,
    const Tensor& num_entries_per_level, T nms_threshold, int pre_nms_topn,
    T min_size, int post_nms_topn, float bbox_xform_clip_default,
    bool debug = false, bool use_legacy_offset = false) {
  int num_images = scores.dim_size(0);
  int num_anchors = scores.dim_size(2);
  int num_levels = num_entries_per_level.dim_size(0);
  int box_dim = anchors.dim_size(1) / num_anchors;
  auto d = context->eigen_gpu_device();
  if (box_dim != 4) {
    return errors::OutOfRange("Box dimensions need to be 4");
  }

  Tensor d_decoded_boxes, d_selected_boxes, d_filtered_scores,
      d_selected_scores, d_layer_counts_tensor, d_output_indices_tensor,
      h_layer_counts_tensor;

  // layer_counts,start_offset,end_offset  +  3 x num_images for batch offset,
  // batch end and batch counts
  int indices_count = num_levels * num_images * 9 + 3 * num_images;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({indices_count}),
                                            &d_layer_counts_tensor));
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  alloc_attr.set_gpu_compatible(true);
  TF_RETURN_IF_ERROR(
      context->allocate_temp(DataType::DT_INT32, TensorShape({indices_count}),
                             &h_layer_counts_tensor, alloc_attr));

  const int32_t* nboxes_per_level =
      num_entries_per_level.flat<int32_t>().data();

  int prenms_per_image_count = 0;
  for (int i = 0; i < num_levels; ++i) {
    prenms_per_image_count +=
        min(nboxes_per_level[i] * num_anchors, pre_nms_topn);
  }
  TF_RETURN_IF_ERROR(context->allocate_temp(
      scores.dtype(), TensorShape({num_images, prenms_per_image_count}),
      &d_filtered_scores));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      scores.dtype(), d_filtered_scores.shape(), &d_selected_scores));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      bbox_deltas.dtype(), TensorShape({num_images, prenms_per_image_count, 4}),
      &d_decoded_boxes));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      d_decoded_boxes.dtype(), d_decoded_boxes.shape(), &d_selected_boxes));

  Tensor per_image_box_counts_tensor_h;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images * num_levels}),
      &per_image_box_counts_tensor_h, alloc_attr));

  // make these a class!
  int32_t* per_image_box_counts =
      per_image_box_counts_tensor_h.flat<int32_t>().data();
  std::vector<int32_t> box_counts_and_offsets(num_levels);
  int* h_orig_layer_counts = h_layer_counts_tensor.flat<int>().data();
  int* h_orig_layer_begin = h_orig_layer_counts + num_images * num_levels;
  int* h_orig_layer_end = h_orig_layer_begin + num_images * num_levels;
  int* h_prenms_layer_counts = h_orig_layer_end + num_images * num_levels;
  int* h_prenms_layer_begin = h_prenms_layer_counts + num_images * num_levels;
  int* h_prenms_layer_end = h_prenms_layer_begin + num_images * num_levels;
  int* h_nms_io_counts = h_prenms_layer_end + num_images * num_levels;

  int* d_layer_counts = d_layer_counts_tensor.flat<int>().data();
  int total_num_boxes_per_level = 0;
  int total_prenms_boxes = 0;
  int global_offset = 0;
  int prenms_offset = 0;
  for (int batch = 0; batch < num_images; ++batch) {
    for (int level = 0; level < num_levels; ++level) {
      int boxes_in_level = nboxes_per_level[level] * num_anchors;
      int prenms_box_count = min(boxes_in_level, pre_nms_topn);
      // original offsets
      h_orig_layer_counts[batch * num_levels + level] = boxes_in_level;
      h_orig_layer_begin[batch * num_levels + level] = global_offset;
      h_orig_layer_end[batch * num_levels + level] =
          h_orig_layer_begin[batch * num_levels + level] + boxes_in_level;
      global_offset += boxes_in_level;
      // counts and offsets after prenms topn
      h_prenms_layer_counts[batch * num_levels + level] = prenms_box_count;
      h_prenms_layer_begin[batch * num_levels + level] = prenms_offset;
      prenms_offset += prenms_box_count;
      h_prenms_layer_end[batch * num_levels + level] =
          h_prenms_layer_begin[batch * num_levels + level] + prenms_box_count;
      // counts that will go in to nms
      h_nms_io_counts[batch * num_levels + level] = prenms_box_count;
      per_image_box_counts[batch * num_levels + level] = prenms_box_count;
    }
  }
  for (int level = 0; level < num_levels; ++level) {
    total_num_boxes_per_level += nboxes_per_level[level] * num_anchors;
    total_prenms_boxes +=
        min(nboxes_per_level[level] * num_anchors, pre_nms_topn);
  }
  Tensor OrigIndices, SortedIndices, SortedScores;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_images, total_num_boxes_per_level}),
      &OrigIndices));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, OrigIndices.shape(), &SortedIndices));
  TF_RETURN_IF_ERROR(
      context->allocate_temp(scores.dtype(), scores.shape(), &SortedScores));

  size_t cub_temp_storage_bytes = 0;
  cudaError_t cuda_ret = cudaSuccess;
  auto cuda_stream = GetGpuStream(context);
  // CHECK_KERNEL(context, "BBP Before starting");
  // Calling cub with nullptrs as inputs will make it return
  // workspace size needed for the operation instead of doing the operation.
  // In this specific instance, cub_sort_storage_bytes will contain the
  // necessary workspace size for sorting after the call.
  cuda_ret = gpuprim::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, cub_temp_storage_bytes,
      static_cast<typename BatchedNMS::ToCubType<T>::Type*>(
          nullptr),  // selected scores
      static_cast<typename BatchedNMS::ToCubType<T>::Type*>(
          nullptr),                            // sorted scores
      static_cast<int*>(nullptr),              // selected Boxes
      static_cast<int*>(nullptr),              // sorted Boxes
      total_num_boxes_per_level * num_images,  // Total number elements to sort
      num_images * num_levels,  // number of independent segments to sort
      static_cast<int*>(nullptr), static_cast<int*>(nullptr), 0,
      8 * sizeof(typename ToCubType<T>::Type),  // sort all bits
      cuda_stream);
  if (cuda_ret != cudaSuccess) {
    return errors::Internal(
        "Topk sorting could not launch "
        "gpuprim::DeviceSegmentedRadixSort::SortPairsDescending to sort "
        "selected indices, "
        "temp_storage_bytes: ",
        cub_temp_storage_bytes, ", status: ", cudaGetErrorString(cuda_ret));
    ;
  }

  Tensor d_cub_temp_buffer_tensor;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_temp_storage_bytes}),
      &d_cub_temp_buffer_tensor));
  char* d_cub_buffer = (char*)d_cub_temp_buffer_tensor.flat<int8>().data();
  int* d_orig_layer_counts = d_layer_counts_tensor.flat<int>().data();
  int* d_orig_layer_offsets_begin =
      d_orig_layer_counts + num_images * num_levels;
  int* d_prenms_layer_counts =
      d_orig_layer_counts + 3 * num_images * num_levels;
  int* d_prenms_layer_offsets_begin =
      d_prenms_layer_counts + num_images * num_levels;
  int* d_nms_io_counts = d_orig_layer_counts + 6 * num_images * num_levels;
  d.memcpyHostToDevice(d_layer_counts, h_orig_layer_counts,
                       h_layer_counts_tensor.NumElements() * sizeof(int32_t));
  auto conf2d = GetGpu2DLaunchConfig(total_num_boxes_per_level, num_images, d);
  // Sort the scores so that we can apply prenms selection afterwards in
  // SelectAndGeneratePreNMSUprightBoxes
  int* original_indices = OrigIndices.flat<int>().data();
  int* prenms_sorted_indices = SortedIndices.flat<int>().data();
  TF_RETURN_IF_ERROR(
      GpuLaunchKernel(BatchedBoxProps::GenerateIndices, conf2d.block_count,
                      conf2d.thread_per_block, 0, d.stream(), num_images,
                      num_levels, d_layer_counts, original_indices));
  // CHECK_KERNEL(context, "BBP Index Generation");
  cuda_ret = gpuprim::DeviceSegmentedRadixSort::SortPairsDescending(
      d_cub_buffer, cub_temp_storage_bytes,
      reinterpret_cast<const typename ToCubType<T>::Type*>(
          scores.flat<T>().data()),  // selected scores
      reinterpret_cast<typename ToCubType<T>::Type*>(
          SortedScores.flat<T>().data()),      // sorted scores
      original_indices,                        // initial indices
      prenms_sorted_indices,                   // sorted indices
      total_num_boxes_per_level * num_images,  // Total number of boxes in batch
      num_images * num_levels,  // num segments, since we are doing topK on
                                // all levels, it is per image now
      d_orig_layer_offsets_begin,
      d_orig_layer_offsets_begin + num_images * num_levels, 0,
      8 * sizeof(typename ToCubType<T>::Type),  // sort all bits
      cuda_stream);
  if (cuda_ret != cudaSuccess) {
    return errors::Internal(
        "Topk sorting could not launch "
        "gpuprim::DeviceSegmentedRadixSort::SortPairsDescending to sort "
        "selected indices, "
        "temp_storage_bytes: ",
        cub_temp_storage_bytes, ", status: ", cudaGetErrorString(cuda_ret));
  }
  // CHECK_KERNEL(context, "BBP Prenms sort");
  if (use_legacy_offset) {
    TF_RETURN_IF_ERROR(

        GpuLaunchKernel(
            BatchedBoxProps::SelectAndGeneratePreNMSUprightBoxes<true, B, T>,
            conf2d.block_count, conf2d.thread_per_block, 0, d.stream(),
            num_images, SortedScores.flat<T>().data(),
            reinterpret_cast<const B*>(bbox_deltas.flat<T>().data()),
            reinterpret_cast<const B*>(anchors.flat<T>().data()),
            prenms_sorted_indices, d_layer_counts, num_levels, num_anchors,
            pre_nms_topn, min_size, image_info.flat<T>().data(),
            bbox_xform_clip_default,
            reinterpret_cast<B*>(d_decoded_boxes.flat<T>().data()),
            d_filtered_scores.flat<T>().data()));
  } else {
    TF_RETURN_IF_ERROR(

        GpuLaunchKernel(
            BatchedBoxProps::SelectAndGeneratePreNMSUprightBoxes<false, B, T>,
            conf2d.block_count, conf2d.thread_per_block, 0, d.stream(),
            num_images, SortedScores.flat<T>().data(),
            reinterpret_cast<const B*>(bbox_deltas.flat<T>().data()),
            reinterpret_cast<const B*>(anchors.flat<T>().data()),
            prenms_sorted_indices, d_layer_counts, num_levels, num_anchors,
            pre_nms_topn, min_size, image_info.flat<T>().data(),
            bbox_xform_clip_default,
            reinterpret_cast<B*>(d_decoded_boxes.flat<T>().data()),
            d_filtered_scores.flat<T>().data()));
  }
  // CHECK_KERNEL(context, "SelectAndGeneratePeNMS");
  Tensor* d_output_indices = &d_output_indices_tensor;
  TF_RETURN_IF_ERROR(DoNMSBatchedGPUJagged(
      context, d_decoded_boxes, d_filtered_scores,
      per_image_box_counts_tensor_h, post_nms_topn, nms_threshold,
      GpuTypeHelpers::asType<T>(-10000.f),
      /*pad_to_max_output=*/true, d_nms_io_counts, &d_output_indices,
      /*kernel=*/0, /*pre_sorted_input=*/true));
  // CHECK_KERNEL(context, "DoNMSBatchedGPUJagged");
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << BatchedNMS::dumpDeviceTensor<int>(
        "layer counts After DoNMS", d_layer_counts_tensor, context);
  }
  int* d_nms_layer_offsets_begin = d_nms_io_counts + num_images * num_levels;
  int* d_nms_layer_offsets_end =
      d_nms_layer_offsets_begin + num_images * num_levels;

  conf2d = GetGpu2DLaunchConfig(num_levels, num_images, d);
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      BatchedBoxProps::SetupSelectionIndices, conf2d.block_count,
      conf2d.thread_per_block, 0, d.stream(), d_nms_io_counts, num_images,
      num_levels, d_nms_layer_offsets_begin));
  // CHECK_KERNEL(context, "BBP Setup Selection indices");
  conf2d = GetGpu2DLaunchConfig(num_levels * post_nms_topn, num_images, d);
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << BatchedNMS::dumpDeviceTensor<int>(
        "SetupSelectionIndices", d_layer_counts_tensor, context);
  }  // select the boxes and scores using nms result from nms inputs!
  if (VLOG_IS_ON(5)) {
    LOG(INFO) << BatchedNMS::dumpDeviceTensor<int>(
        "NMSSelected", d_output_indices_tensor, context);
  }
  // VLOG(1)<<"Sorted scores
  // address="<<(void*)SortedScores.flat<float>().data()<<" shape
  // "<<SortedScores.shape()
  // <<" Filtered_scores="<<(void*)d_filtered_scores.flat<float>().data()<<"
  // shape="<<d_filtered_scores.shape()
  // <<" Decoded boxes="<<(void*)d_decoded_boxes.flat<float>().data()<<"
  // shape="<<d_decoded_boxes.shape()
  // <<" Selected boxes="<<(void*)d_selected_boxes.flat<float>().data()<<"
  // shape="<<d_selected_boxes.shape();
  TF_RETURN_IF_ERROR(

      GpuLaunchKernel(
          BatchedBoxProps::BatchedIndexGather<int, B, const T*, T*>,
          conf2d.block_count, conf2d.thread_per_block, 0, d.stream(),
          /*entry_counts*/ d_nms_io_counts,
          /*input offsets*/ d_prenms_layer_offsets_begin,
          /*output offsets*/ d_nms_layer_offsets_begin,
          /*batch size*/ num_levels * num_images,
          /* index_offset*/ post_nms_topn,
          /*selection index*/ d_output_indices->flat<int>().data(),
          /*original source*/
          reinterpret_cast<const B*>(d_decoded_boxes.flat<T>().data()),
          /*destination*/
          reinterpret_cast<B*>(d_selected_boxes.flat<T>().data()),
          /*original*/ d_filtered_scores.flat<T>().data(),
          /*destination*/
          SortedScores.flat<T>().data()));  // reuse
                                            // sorted_scores
                                            // array to
                                            // save space.
  // CHECK_KERNEL(context, "PostNMS Box gathering");
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << BatchedNMS::dumpDeviceTensor<int>(
        "BNMSOutput output d_layer_counts_tensor", d_layer_counts_tensor,
        context);
  }
  // TopK
  if (debug) {
    std::vector<Tensor*> debug_outputs(2 * num_levels, nullptr);
    std::vector<T*> h_deb_score_ptrs(num_levels, nullptr);
    std::vector<B*> h_deb_box_ptrs(num_levels, nullptr);
    for (int i = 0; i < num_levels; ++i) {
      TF_RETURN_IF_ERROR(context->allocate_output(
          i + 2, TensorShape({num_images, post_nms_topn, 4}),
          &debug_outputs[i]));
      h_deb_box_ptrs[i] =
          reinterpret_cast<B*>(debug_outputs[i]->template flat<T>().data());
      TF_RETURN_IF_ERROR(context->allocate_output(
          i + 2 + num_levels, TensorShape({num_images, post_nms_topn}),
          &debug_outputs[i + num_levels]));
      h_deb_score_ptrs[i] =
          debug_outputs[i + num_levels]->template flat<T>().data();
      VLOG(2) << "Output level " << i << " box " << (void*)h_deb_box_ptrs[i]
              << " scores= " << (void*)h_deb_score_ptrs[i];
    }
    Tensor d_deb_box_ptrs, d_deb_score_ptrs;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT8,
        TensorShape({num_images * num_levels * int(sizeof(B*))}),
        &d_deb_box_ptrs));
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT8,
        TensorShape({num_images * num_levels * int(sizeof(T*))}),
        &d_deb_score_ptrs));
    void* d_dbox_ptrs =
        reinterpret_cast<void*>(d_deb_box_ptrs.flat<int8>().data());
    void* d_dscore_ptrs =
        reinterpret_cast<void*>(d_deb_score_ptrs.flat<int8>().data());
    d.memcpyHostToDevice(d_dbox_ptrs, h_deb_box_ptrs.data(),
                         h_deb_box_ptrs.size() * sizeof(B*));
    d.memcpyHostToDevice(d_dscore_ptrs, h_deb_score_ptrs.data(),
                         h_deb_score_ptrs.size() * sizeof(T*));
    d.synchronize();
    auto conf2deb = GetGpu2DLaunchConfig(num_levels, num_images, d);

    TF_RETURN_IF_ERROR(

        GpuLaunchKernel(BatchedBoxProps::ScatterOutputs<B>,
                        conf2deb.block_count, conf2deb.thread_per_block, 0,
                        d.stream(),
                        reinterpret_cast<B*>(d_selected_boxes.flat<T>().data()),
                        d_nms_io_counts, num_images, num_levels, post_nms_topn,
                        (void*)d_dbox_ptrs));
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        BatchedBoxProps::ScatterOutputs<T>, conf2deb.block_count,
        conf2deb.thread_per_block, 0, d.stream(),
        reinterpret_cast<T*>(SortedScores.flat<T>().data()), d_nms_io_counts,
        num_images, num_levels, post_nms_topn, (void*)d_dscore_ptrs));
    d.synchronize();
  } else {
    Tensor *dumm1, *dumm2;
    TF_RETURN_IF_ERROR(context->allocate_output(2, TensorShape({}), &dumm1));
    TF_RETURN_IF_ERROR(context->allocate_output(3, TensorShape({}), &dumm2));
  }
  size_t cub_output_tmp_size = 0;
  // Calling cub with nullptrs as inputs will make it return
  // workspace size needed for the operation instead of doing the operation.
  // In this specific instance, cub_sort_storage_bytes will contain the
  // necessary workspace size for sorting after the call.
  cuda_ret = gpuprim::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, cub_output_tmp_size,
      static_cast<const typename ToCubType<T>::Type*>(
          nullptr),                                        // selected scores
      static_cast<typename ToCubType<T>::Type*>(nullptr),  // sorted scores
      static_cast<B*>(nullptr),                            // selected Boxes
      static_cast<B*>(nullptr),                            // sorted Boxes
      d_output_indices->NumElements(),  // Total number of boxes in batch
      num_images,  // num segments, since we are doing topK on all levels, it
                   // is per image now
      static_cast<int*>(nullptr), static_cast<int*>(nullptr), 0,
      8 * sizeof(typename ToCubType<T>::Type),  // sort all bits
      cuda_stream);
  if (cuda_ret != cudaSuccess) {
    return errors::Internal(
        "Topk sorting could not launch "
        "gpuprim::DeviceSegmentedRadixSort::SortPairsDescending to sort "
        "selected indices, "
        "temp_storage_bytes: ",
        cub_temp_storage_bytes, ", status: ", cudaGetErrorString(cuda_ret));
  }

  if (cub_output_tmp_size > cub_temp_storage_bytes) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataType::DT_INT8, TensorShape({(int64)cub_output_tmp_size}),
        &d_cub_temp_buffer_tensor));
    d_cub_buffer = (char*)d_cub_temp_buffer_tensor.flat<int8>().data();
    LOG(WARNING) << "Cub buffer was insufficent. Had to reallocate. Was "
                 << cub_temp_storage_bytes << " needed " << cub_output_tmp_size;
    cub_temp_storage_bytes = cub_output_tmp_size;
  }
  // reuse decoded_boxes buffer for sorted boxes
  B* sorted_boxes = reinterpret_cast<B*>(d_decoded_boxes.flat<T>().data());
  int* selected_batch_offsets = d_nms_io_counts + 3 * num_images * num_levels;
  int* selected_batch_counts =
      d_nms_io_counts + 3 * num_images * num_levels + 2 * num_images;
  // sort all batches independently
  cuda_ret = gpuprim::DeviceSegmentedRadixSort::SortPairsDescending(
      d_cub_buffer, cub_temp_storage_bytes,
      reinterpret_cast<const typename ToCubType<T>::Type*>(
          SortedScores.flat<T>().data()),  // nms selected scores
      reinterpret_cast<typename ToCubType<T>::Type*>(
          d_selected_scores.flat<T>().data()),  // final topk scores
      reinterpret_cast<const B*>(
          d_selected_boxes.flat<T>().data()),  // nms selected boxes
      sorted_boxes,                            // final topk boxes
      d_output_indices->NumElements(),         // Total number of boxes to sort
      num_images,  // num segments, since we are doing topK on all levels,
                   // it is per image now
      selected_batch_offsets, selected_batch_offsets + num_images, 0,
      8 * sizeof(typename ToCubType<T>::Type),  // sort all bits
      cuda_stream);

  if (cuda_ret != cudaSuccess) {
    context->SetStatus(errors::Internal(
        "Topk sorting could not launch "
        "gpuprim::DeviceSegmentedRadixSort::SortPairsDescending to sort "
        "selected indices, "
        "temp_storage_bytes: ",
        cub_temp_storage_bytes, ", status: ", cudaGetErrorString(cuda_ret)));
    return Status::OK();
  }
  // CHECK_KERNEL(context, "BBP TopkSort");
  Tensor* output_rois = nullptr;
  Tensor* output_roi_probs = nullptr;
  TF_RETURN_IF_ERROR(context->allocate_output(
      0, TensorShape({num_images, post_nms_topn, 4}), &output_rois));
  TF_RETURN_IF_ERROR(context->allocate_output(
      1, TensorShape({num_images, post_nms_topn}), &output_roi_probs));
  B* d_postnms_rois = reinterpret_cast<B*>((*output_rois).flat<T>().data());
  T* d_postnms_rois_probs = (*output_roi_probs).flat<T>().data();
  conf2d = GetGpu2DLaunchConfig(post_nms_topn, num_images, d);
  TF_RETURN_IF_ERROR(

      GpuLaunchKernel(BatchedBoxProps::CopyTopKtoOutput<B, T>,
                      conf2d.block_count, conf2d.thread_per_block, 0,
                      d.stream(), num_images, selected_batch_counts,
                      selected_batch_offsets, post_nms_topn, sorted_boxes,
                      d_selected_scores.flat<T>().data(), d_postnms_rois,
                      d_postnms_rois_probs));
  // CHECK_KERNEL(context, "CopyTopKtoOutput");
  return Status::OK();
}
}  // namespace BatchedBoxProps

class BatchedBoxProposals : public tensorflow::OpKernel {
 public:
  explicit BatchedBoxProposals(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("post_nms_topn", &post_nms_topn_));
    OP_REQUIRES_OK(context, context->GetAttr("debug", &debug_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_legacy_offset", &use_legacy_offset_));
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
    // multi-level boxes,scores and anchors for inputs
    // at each level , B,H,W,A*4 is reshaped to B,H*W,A4
    // Anchors are shaped as [H*W,A*4]
    // Scores are shaped as [B,H*L,A]
    const auto scores = context->input(0);
    const auto bbox_deltas = context->input(1);
    const auto image_info = context->input(2);
    const auto anchors = context->input(3);
    const auto num_entries_per_level = context->input(4);
    int pre_nms_topn;

    OP_REQUIRES_OK(context, GetScalarValue(context, 6, &pre_nms_topn));
    if (pre_nms_topn <= 0) {
      context->SetStatus(errors::InvalidArgument(
          "pre_nms_topn should be greater than 0", pre_nms_topn));
      return;
    }
    if (scores.dtype() == DataType::DT_FLOAT) {
      float nms_threshold;
      float min_size;
      OP_REQUIRES_OK(context, GetScalarValue(context, 5, &nms_threshold));
      if (nms_threshold < 0 || nms_threshold > 1.0) {
        context->SetStatus(errors::InvalidArgument(
            "nms_threshold should be between 0 and 1. Got ", nms_threshold));
        return;
      }
      OP_REQUIRES_OK(context, GetScalarValue(context, 7, &min_size));
      auto status =
          BatchedBoxProps::BatchedBoxProposalsFunctor<BatchedNMS::Box, float>(
              context, scores, bbox_deltas, image_info, anchors,
              num_entries_per_level, nms_threshold, pre_nms_topn, min_size,
              post_nms_topn_, bbox_xform_clip_default_, debug_,
              use_legacy_offset_);
      if (!status.ok()) context->SetStatus(status);
    } else {
      Eigen::half nms_threshold;
      Eigen::half min_size;
      OP_REQUIRES_OK(context, GetScalarValue(context, 5, &nms_threshold));
      if (Eigen::half_impl::half_to_float(nms_threshold) < 0. ||
          Eigen::half_impl::half_to_float(nms_threshold) > 1.0) {
        context->SetStatus(errors::InvalidArgument(
            "nms_threshold should be between 0 and 1. Got ", nms_threshold));
        return;
      }
      OP_REQUIRES_OK(context, GetScalarValue(context, 7, &min_size));
      auto status =
          BatchedBoxProps::BatchedBoxProposalsFunctor<BatchedNMS::HalfBox,
                                                      Eigen::half>(
              context, scores, bbox_deltas, image_info, anchors,
              num_entries_per_level, nms_threshold, pre_nms_topn, min_size,
              post_nms_topn_,
              bbox_xform_clip_default_,
              debug_, use_legacy_offset_);
      if (!status.ok()) context->SetStatus(status);
    }
  }

 private:
  int post_nms_topn_;
  bool debug_;
  float bbox_xform_clip_default_;
  bool use_legacy_offset_;
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
