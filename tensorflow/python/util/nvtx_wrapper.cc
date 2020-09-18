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

#include <string>

#include "pybind11/pybind11.h"
#include "tensorflow/core/util/env_var.h"
#ifdef GOOGLE_CUDA

#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#else
typedef int nvtxRangeId_t;
#endif
namespace tensorflow {
namespace NVTX_UTILS {
inline bool is_nvtx_on() {
  static bool enabled = []() {
    bool b;
    TF_CHECK_OK(ReadBoolFromEnvVar("ENABLE_NVTX_MARKERS", true, &b));
    return b;
  }();
  return enabled;
}
// DJBHash
inline unsigned djb_hash(const char* c) {
  unsigned hash = 5381;
  unsigned s;
  while (s = *c++) {
    hash = ((hash << 5) + hash) + s;
  }
  return hash;
}

inline uint32_t GetColorForType(const char* c) {
  // colors from colorbrewer2.org
  // https://colorbrewer2.org/?type=qualitative&scheme=Accent&n=8 and
  // https://colorbrewer2.org/?type=qualitative&scheme=Paired&n=8
  static const uint32_t colors[] = {0x7fc97f, 0xbeaed4, 0xfdc086, 0xffff99,
                                    0x386cb0, 0xf0027f, 0xbf5b17, 0x666666,
                                    0xa6cee3, 0x1f78b4, 0xb2df8a, 0x33a02c,
                                    0xfb9a99, 0xe31a1c, 0xfdbf6f, 0xff7f00};
  return colors[djb_hash(c) & 15];
};

#ifdef GOOGLE_CUDA
inline nvtxRangeId_t StartNvtxRange(const char* record_msg, const char* op_type,
                                    uint32_t* color) {
  nvtxEventAttributes_t attrs = {};
  attrs.version = NVTX_VERSION;
  attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attrs.colorType = NVTX_COLOR_ARGB;
  attrs.color = ((color) ? *color : GetColorForType(op_type));
  attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attrs.message.ascii = record_msg;
  attrs.category = 0;
  return ::nvtxRangeStartEx(&attrs);
}
inline void EndNvtxRange(nvtxRangeId_t& range) { ::nvtxRangeEnd(range); }

#else
inline nvtxRangeId_t StartNvtxRange(const char* record_msg, const char* op_type,
                                    uint32_t* color) {
  return nvtxRangeId_t();
}
inline void EndNvtxRange(nvtxRangeId_t& range) {}
#endif



struct PyNVTXToken {
  nvtxRangeId_t range;
};
}  // namespace NVTX_UTILS
}  // namespace tensorflow
namespace py = pybind11;
PYBIND11_MODULE(_pywrap_nvtx, m) {
  py::class_<tensorflow::NVTX_UTILS::PyNVTXToken>(m, "NvtxToken");
  m.def("push",
        [](const std::string& message,
           const std::string& type) -> tensorflow::NVTX_UTILS::PyNVTXToken* {
          auto token = new tensorflow::NVTX_UTILS::PyNVTXToken();
          if (tensorflow::NVTX_UTILS::is_nvtx_on()) {
            token->range = tensorflow::NVTX_UTILS::StartNvtxRange(
                message.c_str(), type.c_str(), nullptr);
          }
          return token;
        });
  m.def("pop", [](tensorflow::NVTX_UTILS::PyNVTXToken* token) {
    if (tensorflow::NVTX_UTILS::is_nvtx_on() && token) {
      tensorflow::NVTX_UTILS::EndNvtxRange(token->range);
    }
  });
}
