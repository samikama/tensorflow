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

REGISTER_OP("GenerateBoundingBoxProposals")
    .Input("scores: float")
    .Input("bbox_deltas: float")
    .Input("image_info: float")
    .Input("anchors: float")
    .Output("rois: float")
    .Output("roi_probabilities: float")
    .Attr("spatial_scale: float = 0.0625")
    .Attr("pre_nms_topn: int = 6000")
    .Attr("post_nms_topn: int = 300")
    .Attr("nms_threshold: float = 0.7")
    .Attr("min_size: float = 16")
    .Attr("correct_transform_coords: bool = true")
    .SetShapeFn([](InferenceContext* c)->Status{
      // make sure input tensors have are correct rank
      ShapeHandle scores,images,bounding_boxes,anchors;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &scores));  //(N, H, W, A)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &bounding_boxes));  //(N,H,W,A4)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &images)); // (N,2)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &anchors)); // (A,4)
      // TODO(skama): verify that the inputs are compatible
      auto roi_shape=c->MakeShape({InferenceContext::kUnknownDim,5}); //(N,5)
      auto prob_shape=c->MakeShape({InferenceContext::kUnknownDim}); // (N)
      c->set_output(0,roi_shape);
      c->set_output(1,prob_shape);
      return Status::OK();
    });

}