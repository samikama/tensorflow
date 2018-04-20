/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
1;4804;0c
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
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"

using tensorflow::str_util::Uppercase;
using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;
#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {
namespace convert {
tensorflow::Status TRTOptimizationPass::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  VLOG(0) << "Called INIT for " << m_name_ << " with config = " << config;
  if (config == nullptr) {
    return tensorflow::Status::OK();
  }
  return tensorflow::Status::OK();
};

tensorflow::Status TRTOptimizationPass::Optimize(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item, GraphDef* optimized_graph) {
  VLOG(0) << "Called TRTOptimization Pass " << m_name_;
  VLOG(0) << "Cluster = " << cluster;
  string offset("  ");
  string offset2 = StrCat(offset, offset);
  string offset3 = StrCat(offset2, offset);
  string offset4 = StrCat(offset2, offset2);
  if (cluster) {
    VLOG(0) << offset << "type             = " << cluster->type();
    VLOG(0) << offset << "num warmup steps = " << cluster->NumWarmupSteps();
    const auto devNames = cluster->GetDeviceNames();
    if (devNames.size()) {
      VLOG(0) << offset << " Device names:";
      for (const auto s : devNames) {
        VLOG(0) << offset2 << s;
      }
    }
    std::unordered_map<string, uint64> peak_mem;
    auto status = cluster->GetPeakMemoryUsage(&peak_mem);
    if (status == tensorflow::Status::OK()) {
      VLOG(0) << offset << "Peak Memory Usage :";
      for (auto s : peak_mem) {
        VLOG(0) << offset2 << s.first << " = " << s.second;
      }
    }

    const auto dev_props = cluster->GetDevices();
    if (dev_props.size()) {
      VLOG(0) << offset << "Device properties:";
      for (auto k : dev_props) {
        VLOG(0) << offset2 << k.first;
        const auto& dt = k.second;
        VLOG(0) << offset3 << "type          = " << dt.type();
        VLOG(0) << offset3 << "vendor        = " << dt.vendor();
        VLOG(0) << offset3 << "model         = " << dt.model();
        VLOG(0) << offset3 << "frequency     = " << dt.frequency();
        VLOG(0) << offset3 << "num cores     = " << dt.num_cores();
        VLOG(0) << offset3 << "num registers = " << dt.num_registers();
        VLOG(0) << offset3 << "L1 cache size = " << dt.l1_cache_size();
        VLOG(0) << offset3 << "L2 cache size = " << dt.l2_cache_size();
        VLOG(0) << offset3 << "L3 cache size = " << dt.l3_cache_size();
        VLOG(0) << offset3 << "SHMem per SMP = "
                << dt.shared_memory_size_per_multiprocessor();
        VLOG(0) << offset3 << "memory size   = " << dt.memory_size();
        VLOG(0) << offset3 << "bandwidth     = " << dt.bandwidth();
        if (dt.environment_size()) {
          VLOG(0) << offset3 << "environment   :";
          for (const auto e : dt.environment()) {
            VLOG(0) << offset4 << e.first << " = " << e.second;
          }
        }
      }
    }
  }
  VLOG(0) << "item: " << item.id;
  int max_dim = -1;
  if (item.feed.size()) {
    VLOG(0) << offset << "Feeds  :";
    for (const auto& f : item.feed) {
      const auto& shape = f.second.shape();
      if (shape.dims() > 0) {
        if (shape.dim_size(0) > max_dim) max_dim = shape.dim_size(0);
      }
      VLOG(0) << offset2 << f.first << " = shaped "
              << f.second.shape().DebugString();
    }
  } else {
    VLOG(0) << offset << "No Feeds";
  }
  if (item.fetch.size()) {
    VLOG(0) << offset << "Fetches  :";
    for (const auto& f : item.fetch) {
      VLOG(0) << offset2 << f;
    }
  } else {
    VLOG(0) << offset << "No Fetches";
  }

  if (item.init_ops.size()) {
    VLOG(0) << offset << "init ops  :";
    for (const auto& f : item.init_ops) {
      VLOG(0) << offset2 << f;
    }
  } else {
    VLOG(0) << offset << "No init ops";
  }
  VLOG(0) << "Save Op = " << item.save_op;
  VLOG(0) << "Restore Op = " << item.restore_op;
  VLOG(0) << "save_restore_loc_tensor = " << item.save_restore_loc_tensor;
  if (item.keep_ops.size()) {
    VLOG(0) << offset << "keep ops  :";
    for (const auto& f : item.keep_ops) {
      VLOG(0) << offset2 << f;
    }
  } else {
    VLOG(0) << offset << "No keep ops";
  }
  VLOG(1) << item.graph.DebugString();
  *optimized_graph=item.graph;
  return tensorflow::Status::OK();
}

void TRTOptimizationPass::Feedback(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item,
    const GraphDef& optimized_graph, double result) {}

using tensorflow::grappler::CustomGraphOptimizerRegistrar;
namespace {

class samiReg : public CustomGraphOptimizerRegistrar {
 public:
  samiReg(const tensorflow::grappler::CustomGraphOptimizerRegistry::Creator& cr,
          const string& name)
      : CustomGraphOptimizerRegistrar(cr, name) {
    VLOG(1) << "Constructing a CustomOptimizationPass registration object for "
            << name;
  }
};
// static CustomGraphOptimizerRegistrar TRTOptimizationPass_Registrar([]() {
static samiReg TRTOptimizationPass_Registrar(
    []() {
      VLOG(1)
          << "Instantiating CustomOptimizationPass object TensorRTOptimizer";
      return new tensorflow::tensorrt::convert::TRTOptimizationPass(
          "TensorRTOptimizer");
    },
    ("TensorRTOptimizer"));
}  // namespace

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif
#endif
