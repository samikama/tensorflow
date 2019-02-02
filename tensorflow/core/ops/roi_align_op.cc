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


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"


namespace tensorflow{
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeAndType;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("ROIAlign")
    .Input("input: float")
    .Input("rois: float")
    .Output("output: float")
    .Attr("spatial_scale: float = 1.0")
    .Attr("pooled_height: int = 1")
    .Attr("pooled_width: int = 1")
    .Attr("sampling_ratio: int = -1")
    .SetShapeFn([](InferenceContext* c)->Status{
      // 4D feature inputs [N,C,H,W]
      ShapeHandle features;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &features));
      // 2D roi boxes [R,4 or 5] [[0, N-1], x1, y1, x2, y2] where N is the batch index
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &boxes));

      auto input_shape=c->input(0);
      auto roi_shape=c->input(1);
      int pooled_h;
      TF_RETURN_IF_ERROR(c->GetAttr("pooled_height",&pooled_h));
      int pooled_w;
      TF_RETURN_IF_ERROR(c->GetAttr("pooled_width",&pooled_w));
      auto Rdim=c->Dim(roi_shape,0); // Num boxes
      auto Cdim=c->Dim(input_shape,1); // Num channels
      auto output_shape=c->MakeShape({Rdim,Cdim,pooled_h,pooled_w});
      c->set_output(0,output_shape);
      return Status::OK();
    });

REGISTER_OP("ROIAlignGrad")
    .Input("grads: float")
    .Input("input: float")
    .Input("rois: float")
    .Output("output: float")
    .Attr("spatial_scale: float = 1.0")
    .Attr("pooled_height: int = 1")
    .Attr("pooled_width: int = 1")
    .Attr("sampling_ratio: int = -1")
    .SetShapeFn([](InferenceContext* c)->Status{
      c->set_output(0,c->input(1));
      return Status::OK();
    });

}