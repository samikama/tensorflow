#ifndef TENSORFLOW_CORE_FRAMEWORK_NVTX_HELPER_H_
#define TENSORFLOW_CORE_FRAMEWORK_NVTX_HELPER_H_
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/env_var.h"
#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#else
typedef int nvtxRangeId_t;
#endif

namespace tensorflow {
namespace nvtx_helper {
inline bool is_nvtx_on() {
  static bool enabled = []() {
    bool b;
    TF_CHECK_OK(ReadBoolFromEnvVar("ENABLE_NVTX_MARKERS", false, &b));
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
inline nvtxRangeId_t StartNvtxRange(const char* record_msg,
                                    const char* op_type) {
  nvtxEventAttributes_t attrs = {};
  attrs.version = NVTX_VERSION;
  attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attrs.colorType = NVTX_COLOR_ARGB;
  attrs.color = GetColorForType(op_type);
  attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attrs.message.ascii = record_msg;
  attrs.category = 0;
  return ::nvtxRangeStartEx(&attrs);
}
inline void EndNvtxRange(nvtxRangeId_t t) { ::nvtxRangeEnd(t); }
#else
inline nvtxRangeId_t StartNvtxRange(const char* record_msg,
                                    const char* op_type) {
  return nvtxRangeId_t(0);
}
inline void EndNvtxRange(nvtxRangeId_t t) {}

#endif
}  // namespace nvtx_helper
}  // namespace tensorflow
#endif
