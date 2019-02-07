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

#ifndef TENSORFLOW_CORE_KERNELS_ROI_ALIGN_OP_H_
#define TENSORFLOW_CORE_KERNELS_ROI_ALIGN_OP_H_

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct ROIAlign {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor X,
                  typename TTypes<T, 2>::ConstTensor RoIs,
                  const int pooled_height, const int pooled_width,
                  const int samplig_ratio, const T spatial_scale,
                  typename TTypes<T, 4>::Tensor Y);
};

template <typename Device, typename T>
struct ROIAlignGrad {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor grads,
                  typename TTypes<T, 4>::ConstTensor inputs,
                  typename TTypes<T, 2>::ConstTensor rois,
                  const int pooled_height, const int pooled_width,
                  const int sampling_ratio, const T spatial_scale,
                  typename TTypes<T, 4>::Tensor output);
};

// ignore for now
template <typename Device, typename T>
struct NMSGPUUpright {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor boxes,
                  const int N, const float treshold, const int max_boxes);
};

template <typename Device, typename T>
struct GeneratePreNMSUprightBoxes {
  void operator()(
      const Device& d,
      typename TTypes<T, 4>::ConstTensor
          digits,  // Scores [N, A, H, W]
      typename TTypes<T, 4>::ConstTensor
          bbox_deltas,  // [N, A*4, H, W] (full, unsorted / sliced)
      typename TTypes<T, 2>::ConstTensor image_shapes,  // (N, 3 ) (h, w, scale) of images
      typename TTypes<T, 2>::ConstTensor
          anchors,  // (A,4)
      const T spatial_scale, const int pre_nms_topN, const int post_nms_topN,
      const T nms_thresh, const T min_size,const bool correct_transform_coords, 
      typename TTypes<T, 2>::Tensor conv_layer_indexes,
      typename TTypes<T, 1>::Tensor image_offset,
      typename TTypes<T, 2>::Tensor rois,
      typename TTypes<T, 1>::Tensor roi_probs );
};

}  // namespace functor
}  // namespace tensorflow

#endif
