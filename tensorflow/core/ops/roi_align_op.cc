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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

namespace tensorflow {
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
    .SetShapeFn([](InferenceContext* c) -> Status {
      // 4D feature inputs [N,C,H,W]
      ShapeHandle features;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &features));
      // 2D roi boxes [R,4 or 5] [[0, N-1], x1, y1, x2, y2] where N is the batch
      // index
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &boxes));

      auto input_shape = c->input(0);
      auto roi_shape = c->input(1);
      int pooled_h;
      TF_RETURN_IF_ERROR(c->GetAttr("pooled_height", &pooled_h));
      int pooled_w;
      TF_RETURN_IF_ERROR(c->GetAttr("pooled_width", &pooled_w));
      auto Rdim = c->Dim(roi_shape, 0);    // Num boxes
      auto Cdim = c->Dim(input_shape, 1);  // Num channels
      auto output_shape = c->MakeShape({Rdim, Cdim, pooled_h, pooled_w});
      c->set_output(0, output_shape);
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
    .SetShapeFn([](InferenceContext* c) -> Status {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("ROIAlignV2")
    .Input("input: float")
    .Input("rois: float")
    .Output("output: float")
    .Attr("spatial_scale: float = 1.0")
    .Attr("pooled_height: int = 1")
    .Attr("pooled_width: int = 1")
    .Attr("sampling_ratio: int = -1")
    .Attr("min_level: int = 2")
    .Attr("max_level: int = 5")
    .Attr("canonical_scale: float = 224.0")
    .Attr("canonical_level: int = 4")
    .SetShapeFn([](InferenceContext* c) -> Status {
      // 5D feature inputs [N,L,C,H,W]
      ShapeHandle features;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &features));
      // 3D roi boxes [N,R,4] [ y1, x1, y2, x2]
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &boxes));

      auto input_shape = c->input(0);
      auto roi_shape = c->input(1);
      int pooled_h;
      TF_RETURN_IF_ERROR(c->GetAttr("pooled_height", &pooled_h));
      int pooled_w;
      TF_RETURN_IF_ERROR(c->GetAttr("pooled_width", &pooled_w));
      auto Rdim = c->Dim(roi_shape, 1);    // Num boxes i.e K
      auto Cdim = c->Dim(input_shape, 2);  // Num channels = C
      auto output_shape = c->MakeShape({c->Dim(input_shape, 0), Rdim, Cdim,
                                        pooled_h, pooled_w});  // N, K, C, H, W
      //std::cerr<<"SAMI SHAPE ="<<c->DebugString(output_shape)<<" input="<<c->DebugString(input_shape)<<" roi="<<c->DebugString(roi_shape)<<std::endl;
      c->set_output(0, output_shape);
      return Status::OK();
    });

REGISTER_OP("ROIAlignV2Grad")
    .Input("grads: float")
    .Input("input: float")
    .Input("rois: float")
    .Output("output: float")
    .Attr("spatial_scale: float = 1.0")
    .Attr("pooled_height: int = 1")
    .Attr("pooled_width: int = 1")
    .Attr("sampling_ratio: int = -1")
    .Attr("min_level: int = 2")
    .Attr("max_level: int = 5")
    .Attr("canonical_scale: float = 224.0")
    .Attr("canonical_level: int = 4")
    .SetShapeFn([](InferenceContext* c) -> Status {
      c->set_output(0, c->input(1));
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
    .SetShapeFn([](InferenceContext* c) -> Status {
      // make sure input tensors have are correct rank
      ShapeHandle scores, images, bounding_boxes, anchors;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &scores));  //(N, H, W, A)
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(1), 4, &bounding_boxes));          //(N,H,W,A4)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &images));   // (N,2)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &anchors));  // (A,4)
      // TODO(skama): verify that the inputs are compatible
      auto roi_shape =
          c->MakeShape({InferenceContext::kUnknownDim, 5});             //(K,5)
      auto prob_shape = c->MakeShape({InferenceContext::kUnknownDim});  // (K)
      c->set_output(0, roi_shape);
      c->set_output(1, prob_shape);
      return Status::OK();
    });

REGISTER_OP("GenerateBoundingBoxProposalsV2")
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
    .SetShapeFn([](InferenceContext* c) -> Status {
      // make sure input tensors have are correct rank
      ShapeHandle scores, images, bounding_boxes, anchors;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &scores));  //(N, H, W, A)
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(1), 4, &bounding_boxes));         //(N,H,W,A4)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &images));  // (N,5)
      auto im_info = c->Dim(images, 1);
      TF_RETURN_IF_ERROR(c->WithValue(im_info, 5, &im_info));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &anchors));  // (A4)
      // TODO(skama): verify that the inputs are compatible
      int post_nms_top_n;
      TF_RETURN_IF_ERROR(c->GetAttr("post_nms_topn", &post_nms_top_n));
      auto roi_shape = c->MakeShape(
          {c->Dim(scores, 0), post_nms_top_n, 4});  //(N,post_nms_top_n,4)
      auto prob_shape = c->MakeShape(
          {c->Dim(scores, 0), post_nms_top_n});  // (N,post_nms_top_n)
      c->set_output(0, roi_shape);
      c->set_output(1, prob_shape);
      return Status::OK();
    });

// New pythorch ops!
REGISTER_OP("MatchRPNProposals") // see add_class_assignments
    .Input("iou: float")  // match quality matrix -> iou matrix BxMxN
    .Input("scaled_gt_boxes: float")  // scaled_ground_truth BxNx4
    .Input("ground_truth_labels: int32")  // ground_truth labels -> BxNx1
    .Output("max_boxes: float") // 
    .Output("max_classes: int32") // see _add_class_assignments!
    .Output("max_overlap: float") // 
    .Output("argmax_iou: int32") // 
    .Attr("allow_low_quality: bool = false")
    .Attr("low_threshold: float = 1")
    .Attr("high_threshold: float = 1")
    .SetShapeFn([](InferenceContext* c) -> Status {
      // TODO fix
      // 4D feature inputs [N,C,H,W]
      ShapeHandle match_quality_matrix;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(0), 3, &match_quality_matrix));  // BxMxN ?

      auto input_shape = c->input(0);
      auto Cdim = c->Dim(input_shape, 1);  // Num channels = C
      auto output_shape = c->MakeShape(
          {c->Dim(input_shape, 0), InferenceContext::kUnknownDim});  // N, ?
      c->set_output(0, output_shape);
      return Status::OK();
    });

REGISTER_OP("ComputeFlow")
    .Input("input: float")  // iou matrix
    .Output("output: float")
    .Attr("height: float = 1")
    .Attr("width: float = 1")
    .SetShapeFn([](InferenceContext* c) -> Status {
      // TODO fix
      // 4D feature inputs [N,C,H,W]
      ShapeHandle match_quality_matrix;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(0), 3, &match_quality_matrix));  // N,X,Y ?

      auto input_shape = c->input(0);
      auto Cdim = c->Dim(input_shape, 1);  // Num channels = C
      auto output_shape = c->MakeShape(
          {c->Dim(input_shape, 0), InferenceContext::kUnknownDim});  // N, ?
      c->set_output(0, output_shape);
      return Status::OK();
    });

REGISTER_OP("BoxIntersectionOverUnion") // see bbox_overlap, caveat -1! 
    .Input("boxes: float")  // Box1 proposals BxMx4 [x1,y1,x2,y2]
    .Input("ground_truth: float")  // Box2 ground truth BxNx4 [x1,y1,x2,y2]
    .Output("output: float") // MxN tensor 
    .SetShapeFn([](InferenceContext* c) -> Status {
      // TODO fix
      // 
      ShapeHandle boxes1,boxes2;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(0), 3, &boxes1));  // BxMx4 ?
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(0), 3, &boxes2));  // BxNx4 ?
      auto input_shape = c->input(0);
      auto output_shape = c->MakeShape(
          {c->Dim(boxes1, 0), c->Dim(boxes1,1),c->Dim(boxes2,1) });  // N, ?
      c->set_output(0, output_shape);
      return Status::OK();
    });

REGISTER_OP("BoxEncode") //convert x1,y1,x2,y2 to xcenter,yc,h,w  see encode_box_targets! 
    .Input("boxes: float")    // BxKx4[y1,x1,y2,x2]
    .Input("anchors: float")  // BxKx4[y1,x1,y2,x2]
    .Input("anchor_labels: float") // BxK
    .Attr("weight_x: float = 1")
    .Attr("weight_y: float = 1")
    .Attr("weight_w: float = 1")
    .Attr("weight_h: float = 1")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) -> Status {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow

// encode_box_targets -> encode_boxes 
// proposal_labels -> boxiou+match_proposals 
// 