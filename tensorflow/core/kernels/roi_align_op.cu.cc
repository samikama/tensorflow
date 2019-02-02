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
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/roi_align_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
// taken from caffe2 implementation caffe2/operators/roi_align_(gradient)?_op.*
namespace {
template <typename T>
EIGEN_DEVICE_FUNC T
bilinear_interpolate(const T* bottom_data, const int height, const int width,
                     T y, T x, const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

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
__global__ void RoIAlignForward(const int nthreads, const T* bottom_data,
                                const T spatial_scale, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                const int sampling_ratio, const T* bottom_rois,
                                int roi_cols, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
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

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

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
__global__ void RoIAlignBackwardFeature(
    const int nthreads,
    const T* top_diff,  // grads
    const int num_rois, const T spatial_scale, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int sampling_ratio,
    const T* bottom_rois,  // rois
    T* bottom_diff /* input_grad */) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;
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
            config.virtual_thread_count, X.data(), spatial_scale, channels,
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
            config.virtual_thread_count, grads.data(), num_rois,spatial_scale, channels,
            height, width, pooled_height, pooled_width, sampling_ratio, rois.data(),
            output.data());
    // clang-format on
  }
};
}  // namespace functor
template struct functor::ROIAlignGrad<GPUDevice, float>;
template struct functor::ROIAlign<GPUDevice, float>;
}  // namespace tensorflow
#endif