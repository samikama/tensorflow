/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/tensorrt/convert/trt_optimization_pass.h"
#include "tensorflow/contrib/tensorrt/convert/convert_graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"

using tensorflow::strings::StrAppend;
#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {
namespace convert {
tensorflow::Status TRTOptimizationPass::Optimize(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item, GraphDef* optimized_graph) {
  return tensorflow::Status::OK();
}
void TRTOptimizationPass::Feedback(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item,
    const GraphDef& optimized_graph, double result) {}
// tensorflow::Status TRTOptimizationPass::Run(const
// tensorflow::GraphOptimizationPassOptions &op){
//     VLOG(1)<<"Called TRTOptimization Pass "<<pass_;
//     VLOG(1)<<"SessionHandle="<<op.session_handle;
//     VLOG(1)<<"op.SessionOptions "<<op.session_options;
//     if(op.session_options){
//         VLOG(1)<<" target="<<op.session_options->target;
//         auto cp=op.session_options->config;
//         for (auto d:cp.device_count()){
//             VLOG(1)<<" Device "<<d.first<<" "<<d.second;
//         }
//         VLOG(1)<<" intra_op_par= "<<cp.intra_op_parallelism_threads();
//         VLOG(1)<<" inter_op_par= "<<cp.inter_op_parallelism_threads();
//         VLOG(1)<<" per_sess_thr= "<<cp.use_per_session_threads();
//         for (auto ds:cp.device_filters()){
//             VLOG(1)<<" device_filters= "<<ds;
//         }
//         VLOG(1)<<" allow_soft_placement="<<cp.allow_soft_placement();
//         auto go=cp.graph_options();
//         VLOG(1)<<" GraphOptions:";
//         VLOG(1)<<"    build_cost_model       = "<<go.build_cost_model();
//         VLOG(1)<<"    build_cost_model_after =
//         "<<go.build_cost_model_after(); VLOG(1)<<"    infer shapes
//         = "<<go.infer_shapes(); VLOG(1)<<"    place_pruned_graph     =
//         "<<go.place_pruned_graph(); VLOG(1)<<"    OptimizerOptions:"; auto
//         oo=go.optimizer_options(); VLOG(1)<<"        do_common_subexp_elim =
//         "<<oo.do_common_subexpression_elimination(); VLOG(1)<<"
//         do_constant_folding   = "<<oo.do_constant_folding(); VLOG(1)<<"
//         max_folded_const_in_b = "<<oo.max_folded_constant_in_bytes();
//         VLOG(1)<<"        do_function_inlining  =
//         "<<oo.do_function_inlining(); VLOG(1)<<"        optimization level
//         = "<<oo.opt_level(); VLOG(1)<<"        GlobalJitLevel        =
//         "<<oo.global_jit_level();
//     }
//     VLOG(1)<<"op.graph="<<op.graph;
//     if(op.partition_graphs){
//         auto pg=op.partition_graphs;
//         VLOG(1)<<"op.partition_graphs:";
//         for(const auto& k:*pg){
//             VLOG(1)<<" '"<<k.first<<"' : "<<k.second.get();
//             //VLOG(1)<<k.second->ToGraphDefDebug().DebugString();
//         }
//     }
//     if(op.cost_model){
//         auto dumpShape=[](const tensorflow::TensorShapeProto& sh)->string{
//             string s("(");
//             for(auto d:sh.dim()){
//                 StrAppend(&s,",",d.size());
//             }
//             StrAppend(&s,")");
//             return s;
//         };
//         auto cm=op.cost_model;
//         if(op.partition_graphs){
//             auto pgraphs=op.partition_graphs;
//             for(const auto &pg:*pgraphs){
//                 auto &g=pg.second;
//                 for(const auto &n:g->op_nodes()){
//                     string nname=n->name();
//                     for(int output=0;output<n->num_outputs();output++){
//                         auto shape=cm->MaxMemoryShape(n,output);
//                         StrAppend(&nname," ",output,":",dumpShape(shape));
//                     }
//                     VLOG(1)<<"  "<<pg.first<<" "<<nname;
//                 }
//             }
//         }else{
//             auto g=op.graph;
//             for(const auto n:g->get()->op_nodes()){
//                 string nname=n->name();
//                 for(int output=0;output<n->num_outputs();output++){
//                     auto shape=cm->MaxMemoryShape(n,output);
//                     StrAppend(&nname," ",output,":",dumpShape(shape));
//                 }
//                 VLOG(1)<<"  "<<nname;
//             }
//         }
//     }
//     return tensorflow::Status::OK();
// }
}  // namespace convert
}  // namespace tensorrt
// REGISTER_OPTIMIZATION(tensorflow::OptimizationPassRegistry::POST_PARTITIONING, 9999, \
    // tensorflow::tensorrt::convert::TRTOptimizationPass(tensorflow::OptimizationPassRegistry::POST_PARTITIONING));
// REGISTER_OPTIMIZATION(tensorflow::OptimizationPassRegistry::PRE_PLACEMENT, 9999, \
    // tensorflow::tensorrt::convert::TRTOptimizationPass(tensorflow::OptimizationPassRegistry::PRE_PLACEMENT));
// REGISTER_OPTIMIZATION(tensorflow::OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 9999, \
    // tensorflow::tensorrt::convert::TRTOptimizationPass(tensorflow::OptimizationPassRegistry::POST_REWRITE_FOR_EXEC));
// REGISTER_OPTIMIZATION(tensorflow::OptimizationPassRegistry::POST_PLACEMENT, 9999, \
    // tensorflow::tensorrt::convert::TRTOptimizationPass(tensorflow::OptimizationPassRegistry::POST_PLACEMENT));

}  // namespace tensorflow

#endif
#endif