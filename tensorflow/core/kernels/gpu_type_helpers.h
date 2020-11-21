#ifndef TENSORFLOW_CORE_KERNELS_GPU_TYPE_HELPERS_H
#define TENSORFLOW_CORE_KERNELS_GPU_TYPE_HELPERS_H
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

// An example Op.

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
namespace GpuTypeHelpers {

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float asFloat(T);
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float asFloat<float>(float f) {
  return f;
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float asFloat<Eigen::half>(
    Eigen::half f) {
  return __half2float(f);
}
template <typename T, typename U>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T asType(U);
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float asType<float, Eigen::half>(
    Eigen::half f) {
  return __half2float(f);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Eigen::half asType<Eigen::half, float>(
    float f) {
  return __float2half(f);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float asType<float, float>(float f) {
  return f;
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Eigen::half
asType<Eigen::half, Eigen::half>(Eigen::half f) {
  return f;
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float asType<float, int>(int f) {
  return static_cast<float>(f);
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Eigen::half asType<Eigen::half, int>(
    int f) {
  return __float2half(static_cast<float>(f));
}
// template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int
// asType<int,Eigen::half>(Eigen::half f){return __half2int_rd(f);}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int asType<int, float>(float f) {
  return static_cast<int>(f);
}

}  // namespace GpuTypeHelpers
}  // namespace tensorflow
#endif
#endif