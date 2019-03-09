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
#include "third_party/cub/device/device_radix_sort.cuh"
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
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T bilinear_interpolate(
    const T* bottom_data, const int height, const int width, T y, T x,
    const int index, /* index for debug only*/ const T* lower_bound = nullptr,
    const T* upper_bound = nullptr, int chann = -1, bool debug = false) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }
  if (y <= 0) y = 0.;
  if (x <= 0) x = 0.;

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
  // if (debug && chann >= 0 && chann < 4) {
  //   int diff = bottom_data - lower_bound;
  //   printf(
  //       " BI y=%f x=%f yl=%d yh=%d xl=%d xh=%d w=%d h=%d lx=%f ly=%f v1=%f
  //       v2=%f v3=%f v4=%f c=%d index=%d " "offset%d %d %d %d\n", y, x, y_low,
  //       y_high, x_low, x_high, width, height, lx,ly, v1,v2,v3,v4 ,chann,
  //       index, diff + y_low * width + x_low, diff + y_low * width + x_high,
  //       diff + y_high * width + x_low, diff + y_high * width + x_high);
  // }
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/, const int level = -1,
    int chann = -1, bool debug = false) {
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
  // if (debug && chann >= 0 && chann < 4)
  //   printf(
  //       "BIG y=%f x=%f yl=%d yh=%d xl=%d xh=%d w=%d h=%d hx=%f hy=%f lx=%f "
  //       "ly=%f level=%d index=%d\n",
  //       y, x, y_low, y_high, x_low, x_high, width, height, hx, hy, lx, ly,
  //       level, index);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

__global__ void box_iou_kernel(Cuda2DLaunchConfig config, float4* boxes,
                               float4* ground_truths, long M, long N,
                               float* box_iou) {
  float xmin1, xmin2, xmax1, xmax2, ymin1, ymin2, ymax1, ymax2, x_tl, y_tl,
      x_br, y_br, w, h, inter, area1, area2, iou;
  size_t b1_idx, b2_idx, b1_row_offset, b2_row_offset;
  CUDA_AXIS_KERNEL_LOOP(img_count, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(i, config.virtual_thread_count.x, X) {
      b1_idx = i / N;
      b2_idx = i % N;
      b1_row_offset = b1_idx + img_count * M;
      b2_row_offset = b2_idx + img_count * N;

      xmin1 = boxes[b1_row_offset].x;
      ymin1 = boxes[b1_row_offset].y;
      xmax1 = boxes[b1_row_offset].z;
      ymax1 = boxes[b1_row_offset].w;
      xmin2 = ground_truths[b2_row_offset].x;
      ymin2 = ground_truths[b2_row_offset].y;
      xmax2 = ground_truths[b2_row_offset].z;
      ymax2 = ground_truths[b2_row_offset].w;
      if (ymin1 < 0. && ymin2 < 0.0) {
        box_iou[img_count * M * N + b1_idx * N + b2_idx] = -1;
      } else {
        x_tl = fmaxf(xmin1, xmin2);
        y_tl = fmaxf(ymin1, ymin2);

        x_br = fminf(xmax1, xmax2);
        y_br = fminf(ymax1, ymax2);
        w = (x_br - x_tl + 1) < 0 ? 0.0f : (x_br - x_tl + 1);
        h = (y_br - y_tl + 1) < 0 ? 0.0f : (y_br - y_tl + 1);
        inter = w * h;
        area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1);
        area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1);
        iou = inter / (area1 + area2 - inter);
        box_iou[img_count * M * N + b1_idx * N + b2_idx] = iou;
      }
    }
  }
}

__launch_bounds__(256) static __global__
    void max_along_gt_idx(float* match, unsigned char* pred_forgiven,
                          long* max_gt_idx, long long gt, long long preds,
                          bool include_low_quality, float low_th,
                          float high_th) {
  long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < preds) {
    float max_iou = 0.0f;
    int max_idx = 0;
    float iou;
    for (long long i = 0; i < gt; i++) {
      iou = match[i * preds + tid];
      if (iou > max_iou) {
        max_iou = iou;
        max_idx = i;
      }
    }
    if (max_iou >= high_th)
      max_gt_idx[tid] = max_idx;
    else if ((pred_forgiven[tid] == 1 && include_low_quality))
      max_gt_idx[tid] = max_idx;
    else if (max_iou < low_th)
      max_gt_idx[tid] = -1;
    else if (max_iou < high_th)
      max_gt_idx[tid] = -2;
  }
}
__device__ void warpReduce(volatile float* sdata, int tid) {
  sdata[tid] = fmax(sdata[tid], sdata[tid + 32]);
  sdata[tid] = fmax(sdata[tid], sdata[tid + 16]);
  sdata[tid] = fmax(sdata[tid], sdata[tid + 8]);
  sdata[tid] = fmax(sdata[tid], sdata[tid + 4]);
  sdata[tid] = fmax(sdata[tid], sdata[tid + 2]);
  sdata[tid] = fmax(sdata[tid], sdata[tid + 1]);
}

static __global__ void max_along_preds(float* match, float* inter_gt,
                                       long long gt, long long preds) {
  int gt_idx = blockIdx.x;
  int chunk_idx = blockIdx.y;
  int gt_offset = chunk_idx * 2048;
  int start_idx = gt_idx * preds + gt_offset;
  int idx = threadIdx.x;
  __shared__ float shbuf[1024];
  shbuf[idx] = 0.0f;
  __syncthreads();
  if (gt_offset + idx + 1024 < preds)
    shbuf[idx] = fmax(match[start_idx + idx], match[start_idx + idx + 1024]);
  else if (gt_offset + idx < preds)
    shbuf[idx] = match[start_idx + idx];
  __syncthreads();
  if (idx < 512) shbuf[idx] = fmax(shbuf[idx], shbuf[idx + 512]);
  __syncthreads();
  if (idx < 256) shbuf[idx] = fmax(shbuf[idx], shbuf[idx + 256]);
  __syncthreads();
  if (idx < 128) shbuf[idx] = fmax(shbuf[idx], shbuf[idx + 128]);
  __syncthreads();
  if (idx < 64) shbuf[idx] = fmax(shbuf[idx], shbuf[idx + 64]);
  __syncthreads();
  if (idx < 32) warpReduce(shbuf, idx);
  if (idx == 0)
    inter_gt[((preds + 2047) / 2048) * gt_idx + chunk_idx] = shbuf[idx];
}

__launch_bounds__(256) static __global__
    void max_along_preds_reduced(float* match, float* max_preds, long long gt,
                                 long long preds) {
  long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < gt) {
    float max_iou = 0.0f;
    float iou;
    for (long long i = 0; i < preds; i++) {
      iou = match[tid * preds + i];
      if (iou > max_iou) max_iou = iou;
    }
    max_preds[tid] = max_iou;
  }
}

__launch_bounds__(256) static __global__
    void forgive_preds(float* match_quality_data, float* d_best_pred_per_gt,
                       unsigned char* d_pred_forgiven, long gt, long preds) {
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < preds) {
    unsigned char forgiven = 0;
    float iou;
    for (int i = 0; i < gt; i++) {
      iou = match_quality_data[i * preds + tid];
      if (iou == d_best_pred_per_gt[i]) {
        forgiven = 1;
        break;
      }
    }
    d_pred_forgiven[tid] = forgiven;
  }
}

__global__ void box_encode_kernel(CudaLaunchConfig config, const float4* boxes,
                                  const float4* anchors, const float wx,
                                  const float wy, const float ww,
                                  const float wh, const float* labels,
                                  float4* output) {
  CUDA_1D_KERNEL_LOOP(i, config.virtual_thread_count) {
    float4& out = output[i];
    if (labels[i] > 0.) {
      const float4& box = boxes[i];
      const float4& anch = anchors[i];
      float bw = box.w - box.y + 1.0;  // layout is [y1,x1,y2,x2]
      float bh = box.z - box.x + 1.0;
      float bcx = box.y + 0.5 * bw;
      float bcy = box.x + 0.5 * bh;
      float aw = anch.w - anch.y;  // layout is [y1,x1,y2,x2]
      float ah = anch.z - anch.x;
      float acx = anch.y + 0.5 * aw;
      float acy = anch.x + 0.5 * ah;
      out.y = wx * (acx - bcx) / bw;
      out.x = wy * (acy - bcy) / bh;
      out.w = wh * log(ah / bh);
      out.z = ww * log(aw / bw);
    } else {
      out.x = 0.;
      out.y = 0.;
      out.z = 0.;
      out.w = 0.;
    }
  }
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

// scale [y1,x1,y2,x2] boxes with level calculated with eq1 in FPN paper
// arXiv:1612.03144 and return levels and scaled boxes
template <typename T>
__global__ void Boxes2ScaledBoxesAndLevels(const CudaLaunchConfig config,
                                           const T* boxes, int min_level,
                                           int max_level, float canonical_scale,
                                           int canonical_level, int* levels,
                                           T* scaled_boxes, bool is_bw = false,
                                           bool debug = false) {
  CUDA_1D_KERNEL_LOOP(i, config.virtual_thread_count) {
    const T* box = boxes + i * 4;
    T* scaled_box = scaled_boxes + i * 4;
    T y1 = box[0];
    T x1 = box[1];
    T y2 = box[2];
    T x2 = box[3];
    T height = y2 - y1;
    T width = x2 - x1;
    T box_area_sqrt = sqrtf(width * height);
    int level =
        max(min_level,
            min((int)floorf(canonical_level +
                            __log2f(box_area_sqrt / canonical_scale + 1e-6f)),
                max_level));
    levels[i] = level - min_level;
    T level_scale = 1 << level;

    scaled_box[0] = y1 / level_scale;
    scaled_box[1] = x1 / level_scale;
    scaled_box[2] = height / level_scale;
    scaled_box[3] = width / level_scale;
    // if(debug){
    // printf(
    //     "BS level=%d scale=%f min=%d max=%d x1=%f y1=%f x2=%f y2=%f h=%f w=%f
    //     sqa=%f l2=%f floor=%f " " sx1=%f sy1=%f sh=%f sw=%f i=%d is_bw=%d\n",
    //     level, level_scale, min_level, max_level,x1,y1,x2,y2, height, width,
    //     box_area_sqrt,
    //     __log2f(box_area_sqrt / canonical_scale + 1e-6f),
    //     floorf(__log2f(box_area_sqrt / canonical_scale + 1e-6f) +
    //            canonical_level),
    //     scaled_box[0],scaled_box[1],scaled_box[2],scaled_box[3],i, is_bw);
    // }
  }
}

template <typename T>
__global__ void RoIAlignForwardV2(
    const Cuda2DLaunchConfig nthreads, const T* bottom_data,
    const T spatial_scale, const int num_levels, const int channels,
    const int height, const int width, const int n_rois,
    const int pooled_height, const int pooled_width, const int sampling_ratio,
    const T* scaled_roi_boxes, const int32* levels, int roi_cols, T* top_data,
    bool debug = false) {
  CUDA_AXIS_KERNEL_LOOP(image_index, nthreads.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(index, nthreads.virtual_thread_count.x, X) {
      // CUDA_1D_KERNEL_LOOP(index, nthreads.virtual_thread_count) {
      // (n, c, ph, pw) is an element in the pooled output
      //  returns (b,n,c,h,w)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      // RoI could have 4 or 5 columns
      const T* offset_bottom_rois =
          scaled_roi_boxes + image_index * n_rois * roi_cols + n * roi_cols;
      T roi_start_w = offset_bottom_rois[1] * spatial_scale;
      T roi_start_h = offset_bottom_rois[0] * spatial_scale;

      // Force malformed ROIs to be 1x1
      T roi_width = Eigen::numext::maxi(offset_bottom_rois[3], (T)1.);
      T roi_height = Eigen::numext::maxi(offset_bottom_rois[2], (T)1.);
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
      int level = levels[image_index * n_rois + n];
      const T* offset_bottom_data =
          bottom_data + image_index * height * width * channels * num_levels +
          height * width * channels * level + c * height * width;
      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_height / pooled_height);  // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_width / pooled_width);

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

      T output_val = 0.;
      int level_height = (T)height / (T)(1 << level);
      int level_width = (T)width / (T)(1 << level);
      for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
      {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);
          // if (debug && c >= 0 && c < 4)
          //   printf(
          //       "AL im=%d, h=%d w=%d lh=%d lw=%d ch=%d nl=%d l=%d pw=%d "
          //       "ph=%d "
          //       "c=%d n=%d "
          //       "x=%f y=%f index=%d offset=%d\n",
          //       image_index, height, width, level_height, level_width,
          //       channels, num_levels, level, pw, ph, c, n, x, y,
          //       index,image_index * height * width * channels * num_levels +
          // height * width * channels * level + c * height * width);
          T val = bilinear_interpolate(offset_bottom_data, level_height,
                                       level_width, y, x, index, bottom_data,
                                       bottom_data + (5 * 256 * 256 * 256), c);
          output_val += val;
        }
      }
      output_val /= count;

      top_data[nthreads.virtual_thread_count.x * image_index + index] =
          output_val;
    }
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

template <typename T>
__global__ void RoIAlignBackwardFeatureV2(
    const Cuda2DLaunchConfig nthreads,
    const T* inp_grads,  // grads
    const T spatial_scale, const int num_levels, const int channels,
    const int height, const int width, const int n_rois,
    const int pooled_height, const int pooled_width, const int sampling_ratio,
    const int roi_cols, const T* input_rois,
    int32* levels,  // scaled rois,  levels
    T* output_grads /* input_grad */, bool debug = false) {
  CUDA_AXIS_KERNEL_LOOP(image_index, nthreads.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(index, nthreads.virtual_thread_count.x, X) {
      // CUDA_1D_KERNEL_LOOP(index, nthreads.virtual_thread_count) {
      // (n, c, ph, pw) is an element in the pooled output
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;
      // this part is buggy in caffe2. Inputs are allowed to be 4 or 5 columns
      // but caffe2 implementation gradient assumes 5 columns
      const T* offset_input_rois =
          input_rois + image_index * n_rois * roi_cols + n * roi_cols;
      // Do not using rounding; this implementation detail is critical
      T roi_start_w = offset_input_rois[1] * spatial_scale;
      T roi_start_h = offset_input_rois[0] * spatial_scale;

      // Force malformed ROIs to be 1x1
      T roi_width = Eigen::numext::maxi(offset_input_rois[3], (T)1.);
      T roi_height = Eigen::numext::maxi(offset_input_rois[2], (T)1.);
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
      int level = levels[n];
      T* offset_output_grads =
          output_grads + image_index * height * width * channels * num_levels +
          height * width * channels * level + c * height * width;
      int inp_grad_offset =
          (image_index * n_rois * channels + n * channels + c) * pooled_height *
          pooled_width;
      const T* offset_inp_grads = inp_grads + inp_grad_offset;
      const T inp_grads_this_bin = offset_inp_grads[ph * pooled_width + pw];
      // if (debug && isnan(inp_grads_this_bin)) {
      //   printf(
      //       "seen nan in grads index=%d feature_offset=%d im=%d, h=%d w=%d "
      //       "ch=%d nl=%d l=%d c=%d n=%d ph=%d pw=%d pooled_width=%d\n",
      //       index, inp_grad_offset, image_index, height, width, channels,
      //       num_levels, level, c, n, ph, pw, pooled_width);
      // }
      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_height / pooled_height);  // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceil(roi_width / pooled_width);

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4
      int level_height = (T)height / (T)(1 << level);
      int level_width = (T)width / (T)(1 << level);

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

          bilinear_interpolate_gradient(level_height, level_width, y, x, w1, w2,
                                        w3, w4, x_low, x_high, y_low, y_high,
                                        index, level, c, debug);

          T g1 = inp_grads_this_bin * w1 / count;
          T g2 = inp_grads_this_bin * w2 / count;
          T g3 = inp_grads_this_bin * w3 / count;
          T g4 = inp_grads_this_bin * w4 / count;
          // if(debug){
          //   printf("ALG im=%d lh=%d lw=%d l=%d pw=%d ph=%d c=%d n=%d x=%f
          //   y=%f xl=%d yl=%d xh=%d yh=%d v=%f w1=%f w2=%f w3=%f w4=%f
          //   count=%f index=%d\n",
          //   image_index,level_height,level_width,level,pw,ph,c,n,x,y,x_low,y_low,x_high,y_high,inp_grads_this_bin,w1,w2,w3,w4,count,index);
          // }
          // if (debug && (isnan(g1) || isnan(g2) || isnan(g3) || isnan(g4)) &&
          //     index < 20) {
          //   printf("nan in gs g1=%d g2=%d g3=%d g4=%d count=%d\n", g1, g2,
          //   g3,
          //          g4, count);
          // }
          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            CudaAtomicAdd(offset_output_grads + y_low * width + x_low,
                          static_cast<T>(g1));
            CudaAtomicAdd(offset_output_grads + y_low * width + x_high,
                          static_cast<T>(g2));
            CudaAtomicAdd(offset_output_grads + y_high * width + x_low,
                          static_cast<T>(g3));
            CudaAtomicAdd(offset_output_grads + y_high * width + x_high,
                          static_cast<T>(g4));
          }  // if
        }    // ix
      }      // iy
    }        // CUDA_1D_KERNEL_LOOP
  }          // RoIAlignBackward
}
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
          // if ((((fabs(i_box.y1 - 283.) < 2.) || (fabs(j_box.y1 - 283.) < 2.))
          // &&
          //     ((fabs(i_box.x1 - 371.) < 5.) || (fabs(j_box.x1 - 371.) <
          //     5)))|| i<10 ) {
          //   printf(
          //       "nms comparing  y1=%f x1=%f y2=%f x2=%f with y1=%f x1=%f
          //       y2=%f " "x2=%f a=%f b=%f bt=%f above_thr=%s pos=%d
          //       threadid=%d index=%d j=%d above_thresh=%d\n", j_box.y1,
          //       j_box.x1, j_box.y2, j_box.x2, i_box.y1, i_box.x1, i_box.y2,
          //       i_box.x2, a, b, bt, ((a > bt) ? "true" : "false"), i *
          //       mask_ld + j_thread_offset /
          //       NMS_BOXES_PER_THREAD,threadIdx.x,pos,j,above_thresh);
          // }
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
  // printf("n=%d mask_id=%d\n",N,mask_ld);
  NMSKernel<<<CAFFE_GET_BLOCKS_2D(N, mask_ld), CAFFE_CUDA_NUM_THREADS_2D, 0,
              device.stream()>>>(d_desc_sorted_boxes, N, thresh, mask_ld,
                                 d_delete_mask);
  int* h_delete_mask = host_delete_mask;
  CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
  // Overlapping CPU computes and D2H memcpy
  // both take about the same time
  int nto_copy = std::min(NMS_CHUNK_SIZE, N);
  cudaEvent_t copy_done;
  cudaEventCreate(&copy_done);
  device.memcpyDeviceToHost(&h_delete_mask[0], &d_delete_mask[0],
                            nto_copy * mask_ld * sizeof(int));
  // CUDA_CHECK(cudaMemcpyAsync(&h_delete_mask[0], &d_delete_mask[0],
  //                            nto_copy * mask_ld * sizeof(int),
  //                            cudaMemcpyDeviceToHost, device.stream()));
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
      // CUDA_CHECK(cudaMemcpyAsync(&h_delete_mask[next_offset * mask_ld],
      //                            &d_delete_mask[next_offset * mask_ld],
      //                            nto_copy * mask_ld * sizeof(int),
      //                            cudaMemcpyDeviceToHost, device.stream()));
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
      // printf("index=%d ibloc=%d inblock=%d bits ",i,iblock,inblock);
      if (!(rmv[iblock] & (1 << inblock))) {
        h_keep_sorted_list.push_back(i);
        int* p = &h_delete_mask[i * mask_ld];
        for (int ib = 0; ib < mask_ld; ++ib) {
          rmv[ib] |= p[ib];
          // printf("%d ",p[ib]);
        }
      }
      // printf("\n");
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
// This kernel should execute in thenexecute otherwise memcpy and

tensorflow::Status nms_gpu_upright_single(
    const float* d_desc_sorted_boxes_float_ptr, const int N, const float thresh,
    int* d_keep_sorted_list, int* h_nkeep, int* dev_delete_mask,
    int* host_delete_mask, OpKernelContext* context) {
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
  // printf("n=%d mask_id=%d\n",N,mask_ld);
  NMSKernel<<<CAFFE_GET_BLOCKS_2D(N, mask_ld), CAFFE_CUDA_NUM_THREADS_2D, 0,
              device.stream()>>>(d_desc_sorted_boxes, N, thresh, mask_ld,
                                 d_delete_mask);
  int* h_delete_mask = host_delete_mask;
  CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
  // Overlapping CPU computes and D2H memcpy
  // both take about the same time
  int nto_copy = std::min(NMS_CHUNK_SIZE, N);
  cudaEvent_t copy_done;
  cudaEventCreate(&copy_done);
  device.memcpyDeviceToHost(&h_delete_mask[0], &d_delete_mask[0],
                            nto_copy * mask_ld * sizeof(int));
  // CUDA_CHECK(cudaMemcpyAsync(&h_delete_mask[0], &d_delete_mask[0],
  //                            nto_copy * mask_ld * sizeof(int),
  //                            cudaMemcpyDeviceToHost, device.stream()));
  CUDA_CHECK(cudaEventRecord(copy_done, device.stream()));
  int offset = 0;
  std::vector<int> h_keep_sorted_list;
  std::vector<int> rmv(mask_ld, 0);
  while (offset < N) {
    const int ncopied = nto_copy;
    int next_offset = offset + ncopied;
    nto_copy = std::min(NMS_CHUNK_SIZE, N - next_offset);
    if (nto_copy > 0) {
      // CUDA_CHECK(cudaMemcpyAsync(&h_delete_mask[next_offset * mask_ld],
      //                            &d_delete_mask[next_offset * mask_ld],
      //                            nto_copy * mask_ld * sizeof(int),
      //                            cudaMemcpyDeviceToHost, device.stream()));
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
  int nboxes_to_generate = config.virtual_thread_count.x;
  int num_images = config.virtual_thread_count.y;
  CUDA_AXIS_KERNEL_LOOP(image_index, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(ibox, config.virtual_thread_count.x, X) {
      // CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index,
      // num_images){ { box_conv_index : # of the same box, but indexed in the
      // scores from the conv layer, of shape (A,H,W) the num_images dimension
      // was already removed box_conv_index = a*K + h*W + w
      const int box_conv_index = d_sorted_scores_keys[image_index * KA + ibox];

      // We want to decompose box_conv_index in (h,w,a)
      // such as box_conv_index = h*W*A + W*A + a
      // (avoiding modulos in the process)
      int remaining = box_conv_index;
      const int dH = WA;  // stride of H
      const int h = remaining / dH;
      remaining -= h * dH;
      const int dW = A;  // stride of H
      const int w = remaining / dW;
      remaining -= w * dW;
      const int a = remaining;  // dA = 1
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
      int deltas_idx = box_conv_index + image_index * KA;
      float4 deltas = d_bbox_deltas[deltas_idx];
      float dx = deltas.x;
      float dy = deltas.y;
      float dw = deltas.z;
      float dh = deltas.w;
      // printf("deltas_idx=%d dx=%f, dy=%f, dw=%f,
      // dh=%f\n",deltas_idx,dx,dy,dw,dh);
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
      const float min_size_scaled =
          min_size;  //*0.166667;
                     // min_size * d_img_info_vec[3 * image_index + 2];
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

__global__ void GeneratePreNMSUprightBoxesKernelV2(
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
  int nboxes_to_generate = config.virtual_thread_count.x;
  int num_images = config.virtual_thread_count.y;
  int num_true = 0;
  CUDA_AXIS_KERNEL_LOOP(image_index, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(ibox, config.virtual_thread_count.x, X) {
      // CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index,
      // num_images){ { box_conv_index : # of the same box, but indexed in the
      // scores from the conv layer, of shape (A,H,W) the num_images dimension
      // was already removed box_conv_index = a*K + h*W + w
      const int box_conv_index = d_sorted_scores_keys[image_index * KA + ibox];

      // We want to decompose box_conv_index in (h,w,a)
      // such as box_conv_index = h*W*A + W*A + a
      // (avoiding modulos in the process)
      int remaining = box_conv_index;
      const int dH = WA;  // stride of H
      const int h = remaining / dH;
      remaining -= h * dH;
      const int dW = A;  // stride of H
      const int w = remaining / dW;
      remaining -= w * dW;
      const int a = remaining;  // dA = 1
      // Loading the anchor a
      // float4 is a struct with float x,y,z,w
      const float4 anchor = d_anchors[box_conv_index];
      // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
      float x1 = anchor.y;
      float x2 = anchor.w;
      float y1 = anchor.x;
      float y2 = anchor.z;

      // TODO use fast math when possible

      // Deltas of shape (N,H,W,A4)
      int deltas_idx = box_conv_index + image_index * KA;
      float4 deltas = d_bbox_deltas[deltas_idx];
      float dx = deltas.y;
      float dy = deltas.x;
      float dw = deltas.w;
      float dh = deltas.z;
      // printf("deltas_idx=%d dx=%f, dy=%f, dw=%f,
      // dh=%f\n",deltas_idx,dx,dy,dw,dh);
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
      // const float y2_old=y2;
      // const float x2_old=x2;
      // const float x1_old=x1;
      // const float y1_old=y1;
      // Clipping box to image
      const float img_height = d_img_info_vec[5 * image_index + 0];
      const float img_width = d_img_info_vec[5 * image_index + 1];
      const float min_size_scaled =
          min_size * d_img_info_vec[5 * image_index + 2];
      // min_size * d_img_info_vec[3 * image_index + 2];
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
      // printf("SAMI Box is x1=%f x2=%f y1=%f y2=%f old x1=%f x2=%f y1=%f y2=%f
      // score=%f keep=%s width=%f height=%f
      // out_index=%d\n",x1,x2,y1,y2,x1_old,x2_old,y1_old,y2_old,
      // d_inout_scores[image_index * KA + ibox],
      // (keep_box?"true":"false"),width,height, out_index);

      d_boxes_keep_flags[out_index] = keep_box;
      d_out_boxes[out_index] = {x1, y1, x2, y2};
      // if(keep_box)printf("Has keep box %d\n",image_index);
      // d_inout_scores size: (num_images,KA)
      if (!keep_box)
        d_inout_scores[image_index * KA + ibox] = FLT_MIN;  // for NMS
    }
  }
}

__global__ void WriteUprightBoxesOutputV2(
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
  // printf("num_images %d KA %d\n",num_images,KA);
  CUDA_AXIS_KERNEL_LOOP(img_idx, config.virtual_thread_count.y, Y) {
    CUDA_AXIS_KERNEL_LOOP(box_idx, config.virtual_thread_count.x, X) {
      // CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {
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
                                            TensorShape({max_nms_mask_size}),
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

class ROIAlignOpV2 : public tensorflow::OpKernel {
 public:
  explicit ROIAlignOpV2(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width_));
    OP_REQUIRES_OK(context, context->GetAttr("min_level", &min_level_));
    OP_REQUIRES_OK(context, context->GetAttr("max_level", &max_level_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("canonical_scale", &canonical_scale_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("canonical_level", &canonical_level_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sampling_ratio", &sampling_ratio_));
    OP_REQUIRES_OK(context, context->GetAttr("debug", &debug_));

    is_nhwc_ = false;
    CHECK_GT(spatial_scale_, 0);
    CHECK_GT(pooled_height_, 0);
    CHECK_GT(pooled_width_, 0);
    CHECK_GE(sampling_ratio_, 0);
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const auto X = context->input(0);
    const auto RoIs = context->input(1);
    TensorShape output_shape;
    Tensor* Y = nullptr;
    int64 RoIDim0 = RoIs.dim_size(1);
    const int64 batch = X.dim_size(0);
    const int64 num_levels = X.dim_size(1);
    const int64 channels = X.dim_size(2);
    const int64 height = X.dim_size(3);
    const int64 width = X.dim_size(4);
    const int64 roi_cols = RoIs.dim_size(2);  // should be 4
    const int64 n_rois = RoIs.dim_size(1);    // num_rois,
    std::vector<int64> shape = {batch, n_rois, channels, pooled_height_,
                                pooled_width_};  // N,K,C,H,W
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(shape, &output_shape));
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &Y));
    if (RoIs.NumElements() == 0) {
      return;
    }

    const int64 total_count = Y->NumElements();
    if (total_count == 0) return;

    const GPUDevice& d = context->eigen_device<GPUDevice>();
    Tensor levels;
    Tensor scaled_boxes;
    OP_REQUIRES_OK(context, context->allocate_temp(RoIs.dtype(), RoIs.shape(),
                                                   &scaled_boxes));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataType::DT_INT32,
                                TensorShape({batch, n_rois, 1}), &levels));
    CudaLaunchConfig config1D = GetCudaLaunchConfig(batch * n_rois, d);
    VLOG(1) << "Before boxes cudaconfig numelts= "
            << config1D.virtual_thread_count << " " << name() << " block "
            << config1D.block_count << " threads=" << config1D.thread_per_block;
    Boxes2ScaledBoxesAndLevels<float>
        <<<config1D.block_count, config1D.thread_per_block, 0, d.stream()>>>(
            config1D, RoIs.flat<float>().data(), min_level_, max_level_,
            canonical_scale_, canonical_level_, (levels).flat<int32>().data(),
            (scaled_boxes).flat<float>().data(), false, debug_);
    // d.synchronize();
    VLOG(1) << "after boxes scaled_shape" << scaled_boxes.shape()
            << " levels.shape" << levels.shape() << " input shape "
            << X.shape();
    Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(
        n_rois * channels * pooled_height_ * pooled_width_, batch, d);
    VLOG(1) << "before RoiAlign forward " << name() << " X " << X.shape()
            << " boxes= " << scaled_boxes.shape()
            << " levels=" << levels.shape() << " output shape=" << Y->shape()
            << " block ( " << config.block_count.x << ","
            << config.block_count.y << "," << config.block_count.z << " ) "
            << " thread ( " << config.thread_per_block.x << ","
            << config.thread_per_block.y << "," << config.thread_per_block.z
            << " )"
            << " virt ( " << config.virtual_thread_count.x << ","
            << config.virtual_thread_count.y << ","
            << config.virtual_thread_count.z << ")";
    RoIAlignForwardV2<float>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            config, X.flat<float>().data(), spatial_scale_, num_levels,
            channels, height, width, n_rois, pooled_height_, pooled_width_,
            sampling_ratio_, (scaled_boxes).flat<float>().data(),
            (levels).flat<int32>().data(), roi_cols, (*Y).flat<float>().data(),
            debug_);
    // d.synchronize();
    VLOG(1) << "after RoiAlign forward, X= " << X.shape().DebugString()
            << " scaled_boxes=" << scaled_boxes.shape()
            << " pooled_width=" << pooled_width_ << " output=" << Y->shape();
    // CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
  }

 private:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
  int min_level_;
  int max_level_;
  float canonical_scale_;
  int canonical_level_;
  bool is_nhwc_;
  bool debug_;
};

class ROIAlignOpGradV2 : public tensorflow::OpKernel {
 public:
  explicit ROIAlignOpGradV2(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sampling_ratio", &sampling_ratio_));
    OP_REQUIRES_OK(context, context->GetAttr("min_level", &min_level_));
    OP_REQUIRES_OK(context, context->GetAttr("max_level", &max_level_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("canonical_scale", &canonical_scale_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("canonical_level", &canonical_level_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sampling_ratio", &sampling_ratio_));
    OP_REQUIRES_OK(context, context->GetAttr("debug", &debug_));
    is_nhwc_ = false;
    CHECK_GT(spatial_scale_, 0);
    CHECK_GT(pooled_height_, 0);
    CHECK_GT(pooled_width_, 0);
    CHECK_GE(sampling_ratio_, 0);
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const auto grads = context->input(0);
    const auto features = context->input(1);
    const auto RoIs = context->input(2);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, features.shape(), &output));
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    const int64 batch = features.dim_size(0);
    const int64 num_levels = features.dim_size(1);
    const int64 channels = features.dim_size(2);
    const int64 height = features.dim_size(3);
    const int64 width = features.dim_size(4);
    const int64 roi_cols = RoIs.dim_size(2);
    const int64 n_rois = RoIs.dim_size(1);
    Tensor levels;
    Tensor scaled_boxes;
    OP_REQUIRES_OK(context, context->allocate_temp(RoIs.dtype(), RoIs.shape(),
                                                   &scaled_boxes));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataType::DT_INT32,
                                TensorShape({batch, n_rois, 1}), &levels));
    CudaLaunchConfig config1D = GetCudaLaunchConfig(batch * n_rois, d);
    VLOG(1) << "Before boxes cudaconfig numelts= "
            << config1D.virtual_thread_count << " " << name();
    Boxes2ScaledBoxesAndLevels<float>
        <<<config1D.block_count, config1D.thread_per_block, 0, d.stream()>>>(
            config1D, RoIs.flat<float>().data(), min_level_, max_level_,
            canonical_scale_, canonical_level_, (levels).flat<int32>().data(),
            (scaled_boxes).flat<float>().data(), true, debug_);
    // d.synchronize();
    VLOG(1) << "after boxes scaled_shape" << scaled_boxes.shape()
            << " levels.shape" << levels.shape();
    // CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);

    Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(
        n_rois * channels * pooled_height_ * pooled_width_, batch, d);
    VLOG(1) << "before RoiAlign Backward " << name()
            << " grads=" << grads.shape() << " features=" << features.shape()
            << " RoIs" << RoIs.shape() << " block ( " << config.block_count.x
            << "," << config.block_count.y << "," << config.block_count.z
            << " ) "
            << " thread ( " << config.thread_per_block.x << ","
            << config.thread_per_block.y << "," << config.thread_per_block.z
            << " )"
            << " virt ( " << config.virtual_thread_count.x << ","
            << config.virtual_thread_count.y << ","
            << config.virtual_thread_count.z << ")";
    CudaLaunchConfig zconfig = GetCudaLaunchConfig(output->NumElements(), d);
    SetZero<<<zconfig.block_count, zconfig.thread_per_block, 0, d.stream()>>>(
        zconfig.virtual_thread_count, (*output).flat<float>().data());

    RoIAlignBackwardFeatureV2<float>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            config, grads.flat<float>().data(), spatial_scale_, num_levels,
            channels, height, width, n_rois, pooled_height_, pooled_width_,
            sampling_ratio_, roi_cols, (scaled_boxes).flat<float>().data(),
            (levels).flat<int32>().data(), (*output).flat<float>().data(),
            debug_);
    // d.synchronize();
    VLOG(1) << "after RoiAlign Backward, X.shape() "
            << features.shape().DebugString()
            << " scaled_boxes=" << scaled_boxes.shape()
            << " pooled_width=" << pooled_width_
            << "output shape=" << output->shape();
    // CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
  }

 private:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
  int min_level_;
  int max_level_;
  float canonical_scale_;
  int canonical_level_;
  bool is_nhwc_;
  bool debug_;
};

class BoxEncode : public tensorflow::OpKernel {
 public:
  explicit BoxEncode(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("weight_x", &weight_x_));
    OP_REQUIRES_OK(context, context->GetAttr("weight_y", &weight_y_));
    OP_REQUIRES_OK(context, context->GetAttr("weight_h", &weight_h_));
    OP_REQUIRES_OK(context, context->GetAttr("weight_w", &weight_w_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const auto boxes = context->input(0);
    const auto ground_truth = context->input(1);
    const auto labels = context->input(2);
    Tensor* output = nullptr;
    const int64 batch = boxes.dim_size(0);
    const int64 num_boxes = boxes.dim_size(1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, boxes.shape(), &output));
    if (boxes.NumElements() == 0) {
      return;
    }
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    CudaLaunchConfig config1D = GetCudaLaunchConfig(batch * num_boxes, d);
    VLOG(1) << "Before encode_boxes= " << config1D.virtual_thread_count << " "
            << name();
    box_encode_kernel<<<config1D.block_count, config1D.thread_per_block, 0,
                        d.stream()>>>(
        config1D, (float4*)boxes.flat<float>().data(),
        (float4*)ground_truth.flat<float>().data(), weight_x_, weight_y_,
        weight_w_, weight_h_, labels.flat<float>().data(),
        (float4*)(*output).flat<float>().data());
    // d.synchronize();
    VLOG(1) << "after encode_boxes " << name();
    // CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
  }

 private:
  float weight_x_;
  float weight_y_;
  float weight_w_;
  float weight_h_;
};

class BoxIntersectionOverUnion : public tensorflow::OpKernel {
 public:
  explicit BoxIntersectionOverUnion(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const auto boxes = context->input(0);
    const auto ground_truth = context->input(1);
    Tensor* output = nullptr;
    const int64 batch = boxes.dim_size(0);
    const int64 num_boxes = boxes.dim_size(1);
    const int64 num_gt = ground_truth.dim_size(1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({batch, num_boxes, num_gt}), &output));
    if (boxes.NumElements() == 0) {
      return;
    }
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    Cuda2DLaunchConfig config =
        GetCuda2DLaunchConfig(num_boxes * num_gt, batch, d);
    VLOG(1) << "Before encode_boxes= " << config.virtual_thread_count.x << " "
            << config.virtual_thread_count.y << " " << name();
    box_iou_kernel<<<config.block_count, config.thread_per_block, 0,
                     d.stream()>>>(config, (float4*)boxes.flat<float>().data(),
                                   (float4*)ground_truth.flat<float>().data(),
                                   num_boxes, num_gt,
                                   (*output).flat<float>().data());
    // d.synchronize();
    VLOG(1) << "after encode_boxes " << name();
    // CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
  }

 private:
  float weight_x_;
  float weight_y_;
  float weight_w_;
  float weight_h_;
};

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
    OP_REQUIRES_OK(context, context->GetAttr("correct_transform_coords",
                                             &correct_transform_coords_));
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
    // VLOG(0)<<"num_images="<<num_images<<" A="<<A<<" H="<<H<<" W="<<W;
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
    float4* f4_ptr = nullptr;
    cub::DeviceSelect::Flagged(nullptr, cub_select_temp_storage_bytes, f4_ptr,
                               char_ptr, f4_ptr, int_ptr, K * A, cuda_stream);
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
    CudaLaunchConfig zconfig =
        GetCudaLaunchConfig(dev_postnms_rois_probs.NumElements(), d);
    SetZero<<<zconfig.block_count, zconfig.thread_per_block, 0, d.stream()>>>(
        zconfig.virtual_thread_count,
        dev_postnms_rois_probs.flat<float>().data());

    for (int image_index = 0; image_index < num_images; ++image_index) {
      zconfig = GetCudaLaunchConfig(dev_nms_mask.NumElements(), d);
      SetZero<<<zconfig.block_count, zconfig.thread_per_block, 0, d.stream()>>>(
          zconfig.virtual_thread_count, d_nms_mask);
      zconfig = GetCudaLaunchConfig(dev_nms_mask.NumElements(), d);
      SetZero<<<zconfig.block_count, zconfig.thread_per_block, 0, d.stream()>>>(
          zconfig.virtual_thread_count, d_nms_mask);
      zconfig = GetCudaLaunchConfig(dev_image_boxes_keep_list.NumElements(), d);
      SetZero<<<zconfig.block_count, zconfig.thread_per_block, 0, d.stream()>>>(
          zconfig.virtual_thread_count, d_image_boxes_keep_list);

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

class GenerateBoundingBoxProposalsV2 : public tensorflow::AsyncOpKernel {
 public:
  explicit GenerateBoundingBoxProposalsV2(
      tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
    feat_stride_ = 1.0 / spatial_scale_;
    OP_REQUIRES_OK(context, context->GetAttr("pre_nms_topn", &pre_nms_topn_));
    OP_REQUIRES_OK(context, context->GetAttr("post_nms_topn", &post_nms_topn_));
    OP_REQUIRES_OK(context, context->GetAttr("nms_threshold", &nms_threshold_));
    OP_REQUIRES_OK(context, context->GetAttr("min_size", &min_size_));
    OP_REQUIRES_OK(context, context->GetAttr("debug", &debug_));
    // compatibility for detectron like networks. False for generic case
    OP_REQUIRES_OK(context, context->GetAttr("correct_transform_coords",
                                             &correct_transform_coords_));
    CHECK_GT(spatial_scale_, 0);
    CHECK_GT(pre_nms_topn_, 0);
    CHECK_GT(post_nms_topn_, 0);
    CHECK_GT(nms_threshold_, 0);
    CHECK_GE(min_size_, 0);
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
    const auto box_dim = anchors.dim_size(2) / A;
    CHECK_EQ(box_dim, 4);
    // TODO(skama): make sure that inputs are ok.
    const int K = H * W;
    // VLOG(0)<<"num_images="<<num_images<<" A="<<A<<" H="<<H<<" W="<<W;
    const int conv_layer_nboxes = K * A;
    // The following calls to CUB primitives do nothing
    // (because the first arg is nullptr)
    // except setting cub_*_temp_storage_bytes
    auto cuda_stream = GetCudaStream(context);
    size_t cub_sort_temp_storage_bytes = 0;
    float* flt_ptr = nullptr;
    int* int_ptr = nullptr;
    cudaError_t cuda_ret = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, cub_sort_temp_storage_bytes, flt_ptr, flt_ptr, int_ptr,
        int_ptr, num_images * conv_layer_nboxes, num_images, int_ptr, int_ptr,
        0, 8 * sizeof(float),  // sort all bits
        cuda_stream);
    CHECK_EQ(cuda_ret, CUDA_SUCCESS);
    // get the size of select temp buffer
    size_t cub_select_temp_storage_bytes = 0;
    char* char_ptr = nullptr;
    float4* f4_ptr = nullptr;
    cuda_ret = cub::DeviceSelect::Flagged(
        nullptr, cub_select_temp_storage_bytes, f4_ptr, char_ptr, f4_ptr,
        int_ptr, K * A, cuda_stream);
    CHECK_EQ(cuda_ret, CUDA_SUCCESS);
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
    cuda_ret = cub::DeviceSegmentedRadixSort::SortPairsDescending(
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
    CHECK_EQ(cuda_ret, CUDA_SUCCESS);
    conf2d = GetCuda2DLaunchConfig(nboxes_to_generate, num_images, d);
    GeneratePreNMSUprightBoxesKernelV2<<<
        conf2d.block_count, conf2d.thread_per_block, 0, d.stream()>>>(
        conf2d, d_sorted_conv_layer_indexes.flat<int>().data(),
        reinterpret_cast<const float4*>(bbox_deltas.flat<float>().data()),
        reinterpret_cast<const float4*>(anchors.flat<float>().data()), H, W, A,
        feat_stride_, min_size_, image_info.flat<float>().data(),
        bbox_xform_clip_default_, correct_transform_coords_,
        reinterpret_cast<float4*>(dev_boxes.flat<float>().data()),
        nboxes_to_generate, dev_sorted_scores.flat<float>().data(),
        (char*)dev_boxes_keep_flags.flat<int8>().data());
    CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
    const int nboxes_generated = nboxes_to_generate;
    const int roi_cols = box_dim;
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
    CudaLaunchConfig zconfig =
        GetCudaLaunchConfig(dev_postnms_rois_probs.NumElements(), d);
    SetZero<<<zconfig.block_count, zconfig.thread_per_block, 0, d.stream()>>>(
        zconfig.virtual_thread_count,
        dev_postnms_rois_probs.flat<float>().data());

    int* d_prenms_nboxes = dev_prenms_nboxes.flat<int>().data();
    int h_prenms_nboxes;
    char* d_boxes_keep_flags = (char*)dev_boxes_keep_flags.flat<int8>().data();
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
    Tensor* output_rois = nullptr;
    Tensor* output_roi_probs = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(
            0, TensorShape({num_images, post_nms_topn_, roi_cols}),
            &output_rois),
        done);
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(1, TensorShape({num_images, post_nms_topn_}),
                                 &output_roi_probs),
        done);
    float* d_postnms_rois = (*output_rois).flat<float>().data();
    float* d_postnms_rois_probs = (*output_roi_probs).flat<float>().data();
    for (int image_index = 0; image_index < num_images; ++image_index) {
      // Sub matrices for current image
      zconfig = GetCudaLaunchConfig(dev_nms_mask.NumElements(), d);
      SetZero<<<zconfig.block_count, zconfig.thread_per_block, 0, d.stream()>>>(
          zconfig.virtual_thread_count, d_nms_mask);
      zconfig = GetCudaLaunchConfig(dev_image_boxes_keep_list.NumElements(), d);
      SetZero<<<zconfig.block_count, zconfig.thread_per_block, 0, d.stream()>>>(
          zconfig.virtual_thread_count, d_image_boxes_keep_list);
      const float* d_image_boxes =
          &d_boxes[image_index * nboxes_generated * box_dim];
      const float* d_image_sorted_scores =
          &d_sorted_scores[image_index * K * A];
      char* d_image_boxes_keep_flags =
          &d_boxes_keep_flags[image_index * nboxes_generated];

      float* d_image_postnms_rois =
          &d_postnms_rois[image_index * roi_cols * post_nms_topn_];
      float* d_image_postnms_rois_probs =
          &d_postnms_rois_probs[image_index * post_nms_topn_];

      // Moving valid boxes (ie the ones with d_boxes_keep_flags[ibox] == true)
      // to the output tensors
      // printf("Host before flagged boxes=%d
      // ngen=%d\n",h_prenms_nboxes,nboxes_generated);
      cuda_ret = cub::DeviceSelect::Flagged(
          d_cub_select_temp_storage, cub_select_temp_storage_bytes,
          reinterpret_cast<const float4*>(d_image_boxes),
          d_image_boxes_keep_flags,
          reinterpret_cast<float4*>(d_image_prenms_boxes), d_prenms_nboxes,
          nboxes_generated, d.stream());
      CHECK_EQ(cuda_ret, CUDA_SUCCESS);
      cuda_ret = cub::DeviceSelect::Flagged(
          d_cub_select_temp_storage, cub_select_temp_storage_bytes,
          d_image_sorted_scores, d_image_boxes_keep_flags,
          d_image_prenms_scores, d_prenms_nboxes, nboxes_generated, d.stream());
      CHECK_EQ(cuda_ret, CUDA_SUCCESS);
      d.memcpyDeviceToHost(&h_prenms_nboxes, d_prenms_nboxes, sizeof(int));
      d.synchronize();

      // We know prenms_boxes <= topN_prenms, because nboxes_generated <=
      // topN_prenms. Calling NMS on the generated boxes
      const int prenms_nboxes = h_prenms_nboxes;
      // printf("Host boxes=%d ngen=%d\n",h_prenms_nboxes,nboxes_generated);
      int nkeep;
      // printf("Before nms\n");
      nms_gpu_upright(d_image_prenms_boxes, prenms_nboxes, nms_threshold_,
                      d_image_boxes_keep_list, &nkeep, d_nms_mask, h_nms_mask,
                      context);
      CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
      // printf("After nms nkeep=%d\n",nkeep);
      // All operations done after previous sort were keeping the relative order
      // of the elements the elements are still sorted keep topN <=> truncate
      // the array
      const int postnms_nboxes = std::min(nkeep, post_nms_topn_);
      // Moving the out boxes to the output tensors,
      // adding the image_index dimension on the fly
      CudaLaunchConfig config = GetCudaLaunchConfig(post_nms_topn_, d);
      // make this single kernel
      WriteUprightBoxesOutputV2<<<config.block_count, config.thread_per_block,
                                  0, d.stream()>>>(
          config, reinterpret_cast<const float4*>(d_image_prenms_boxes),
          d_image_prenms_scores, d_image_boxes_keep_list, postnms_nboxes,
          d_image_postnms_rois, d_image_postnms_rois_probs);
      nrois_in_output += postnms_nboxes;
      CHECK_EQ(cudaGetLastError(), CUDA_SUCCESS);
    }
    // Tensor* output_rois = nullptr;
    // Tensor* output_roi_probs = nullptr;
    // OP_REQUIRES_OK_ASYNC(
    //     context,
    //     context->allocate_output(0, TensorShape({nrois_in_output, roi_cols}),
    //                              &output_rois),
    //     done);
    // OP_REQUIRES_OK_ASYNC(
    //     context,
    //     context->allocate_output(1, TensorShape({nrois_in_output}),
    //                              &output_roi_probs),
    //     done);
    // d.memcpyDeviceToHost(output_rois->flat<float>().data(), d_postnms_rois,
    //                      nrois_in_output * roi_cols * sizeof(float));
    // d.memcpyDeviceToHost(output_roi_probs->flat<float>().data(),
    //                      d_postnms_rois_probs, nrois_in_output *
    //                      sizeof(float));
    done();
  }

 private:
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
  bool debug_;
};

#undef GENRPN_BOXES_PER_THREAD
#undef GENRPN_CHUNK_SIZE
}  // namespace sami
template struct functor::ROIAlignGrad<GPUDevice, float>;
template struct functor::ROIAlign<GPUDevice, float>;
// template struct functor::ROIAlignGradV2<GPUDevice, float>;
// template struct functor::ROIAlignV2<GPUDevice, float>;
REGISTER_KERNEL_BUILDER(
    Name("GenerateBoundingBoxProposals").Device(tensorflow::DEVICE_GPU),
    tensorflow::sami::GenerateBoundingBoxProposals)
REGISTER_KERNEL_BUILDER(
    Name("GenerateBoundingBoxProposalsV2").Device(tensorflow::DEVICE_GPU),
    tensorflow::sami::GenerateBoundingBoxProposalsV2)
REGISTER_KERNEL_BUILDER(Name("ROIAlignV2").Device(tensorflow::DEVICE_GPU),
                        tensorflow::sami::ROIAlignOpV2);
REGISTER_KERNEL_BUILDER(Name("ROIAlignV2Grad").Device(tensorflow::DEVICE_GPU),
                        tensorflow::sami::ROIAlignOpGradV2);
REGISTER_KERNEL_BUILDER(Name("BoxEncode").Device(tensorflow::DEVICE_GPU),
                        tensorflow::sami::BoxEncode);
REGISTER_KERNEL_BUILDER(
    Name("BoxIntersectionOverUnion").Device(tensorflow::DEVICE_GPU),
    tensorflow::sami::BoxIntersectionOverUnion);

}  // namespace tensorflow
#endif