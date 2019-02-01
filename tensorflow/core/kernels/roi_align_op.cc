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

#include "tensorflow/core/kernels/roi_align_op.h"
#include <algorithm>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// put it in sami namspace for now so that there is no conflict with other nms
// implementations
namespace sami {

class ROIAlignOp : public tensorflow::OpKernel {
 public:
  explicit ROIAlignOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height_));
    OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sampling_ratio", &sampling_ratio_));
    string data_layout;
    OP_REQUIRES_OK(context, context->GetAttr("data_layout", &data_layout));
    is_nhwc_ = (data_layout == "NHWC");
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
    int64 RoIDim0 = RoIs.dim_size(0);
    if (is_nhwc_) {
      std::vector<int64> shape = {RoIDim0, pooled_height_, pooled_width_,
                                  X.dim_size(3)};
      OP_REQUIRES_OK(context,
                     TensorShapeUtils::MakeShape(shape, &output_shape));
    } else {
        std::vector<int64> shape = {RoIDim0, X.dim_size(1),
                                    pooled_height_,
                                    pooled_width_};
        OP_REQUIRES_OK(context,
                       TensorShapeUtils::MakeShape(shape, &output_shape));
    }
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &Y));
    if (RoIs.NumElements() == 0) {
      return;
    }
    tensorflow::GPUDevice d = context->eigen_gpu_device();
    typename TTypes<float, 4>::ConstTensor x(X.tensor<float, 4>());
    typename TTypes<float, 2>::ConstTensor rois(RoIs.tensor<float, 2>());
    TTypes<float, 4>::Tensor y(Y->tensor<float, 4>());
    functor::ROIAlign<GPUDevice, float>()(d, x, rois, pooled_height_,
                                          pooled_width_, sampling_ratio_,
                                          spatial_scale_, y);
  }

 private:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
  bool is_nhwc_;
};

}  // namespace sami
// REGISTER_KERNEL_BUILDER(Name("ROIAlign").Device(tensorflow::DEVICE_GPU),
//                         tensorflow::sami::ROIAlignOp);
}  // namespace tensorflow
#endif