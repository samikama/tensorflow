#ifndef TENSORFLOW_CORE_KERNELS_BATCHED_NON_MAX_SUPPRESSION_OP_H_
#define TENSORFLOW_CORE_KERNELS_BATCHED_NON_MAX_SUPPRESSION_OP_H_
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <sstream>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_device_functions.h"
namespace tensorflow {
Status DoNMSBatchedGPU(OpKernelContext* context, const Tensor& boxes,
                       const Tensor& scores, const Tensor& box_counts_tensor,
                       const int max_output_size, const float iou_threshold_val,
                       const float score_threshold, bool pad_to_max_output,
                       int* num_saved_outputs, Tensor** output_indices,
                       int kernel = -1, bool pre_sorted_inputs = false);

Status DoNMSBatchedGPUJagged(
    OpKernelContext* context, const Tensor& boxes, const Tensor& scores,
    const Tensor& box_counts_tensor, const int max_output_size,
    const float iou_threshold_val, const float score_threshold,
    bool pad_to_max_output, int* num_saved_outputs, Tensor** output_indices,
    int kernel = 0, bool pre_sorted_inputs = false);
namespace BNMS {
template <typename Index>
__device__ __forceinline__ void SelectHelper(const Index i_selected,
                                             const Index i_original) {}

template <typename Index, typename T, typename... Args>
__device__ __forceinline__ void SelectHelper(const Index i_selected,
                                             const Index i_original,
                                             const T* original, T* selected,
                                             Args... args) {
  selected[i_selected] = original[i_original];
  BNMS::SelectHelper(i_selected, i_original, args...);
}

// Batch version of IndexMultiSelect, num_elemets contains number of elements in
// each entry offsets is the offsets of batch entries,
template <typename Index, typename T, typename... Args>
__global__ void BatchedIndexMultiSelect(const int* num_elements,
                                        const int* input_strides,
                                        const int* output_strides,
                                        int batch_size, const Index* indices,
                                        const T* original, T* selected,
                                        Args... args) {
  for (const int y : GpuGridRangeY(batch_size)) {
    int istride = input_strides[y];
    int ostride = output_strides[y];
    for (const int idx : GpuGridRangeX(num_elements[y])) {
      BNMS::SelectHelper(idx + ostride, istride + indices[idx + istride],
                         original, selected, args...);
    }
  }
}

template <typename T>
std::string dumpDeviceTensor(const std::string& Name, const Tensor& t,
                             OpKernelContext* context) {
  std::vector<T> storage(t.NumElements(), 0);
  auto d = context->eigen_gpu_device();
  d.memcpyDeviceToHost(storage.data(), t.flat<T>().data(),
                       t.NumElements() * sizeof(T));
  d.synchronize();
  std::stringstream oss;
  oss << "Dumping " << Name << " NumElements=" << t.NumElements() << std::endl;
  for (size_t l = 0; l < storage.size() / 32; ++l) {
    oss << "l= " << l << " [ " << l * 32 << " ] = ";
    for (size_t k = 0; k < 32; ++k) {
      oss << storage[l * 32 + k] << " ";
    }
    oss << std::endl;
  }
  if ((storage.size() / 32) * 32 < storage.size()) {
    oss << "l= " << (storage.size() / 32) << " [ " << (storage.size() / 32) * 32
        << " ] = ";
    for (int k = (storage.size() / 32) * 32; k < storage.size(); ++k) {
      oss << storage[k] << " ";
    }
    oss << std::endl;
  }
  return oss.str();
}

template <typename T>
std::string dumpDeviceTensor(const std::string& Name, const int* t,
                             const int num_elements, OpKernelContext* context) {
  std::vector<int> storage(num_elements, 0);
  auto d = context->eigen_gpu_device();
  d.memcpyDeviceToHost(storage.data(), t, num_elements * sizeof(T));
  d.synchronize();
  std::stringstream oss;
  oss << "Dumping " << Name << " NumElements=" << num_elements << std::endl;
  for (size_t l = 0; l < storage.size() / 32; ++l) {
    oss << "l= " << l << " [ " << l * 32 << " ] = ";
    for (size_t k = 0; k < 32; ++k) {
      oss << storage[l * 32 + k] << " ";
    }
    oss << std::endl;
  }
  if ((storage.size() / 32) * 32 < storage.size()) {
    oss << "l= " << (storage.size() / 32) << " [ " << (storage.size() / 32) * 32
        << " ] = ";
    for (int k = (storage.size() / 32) * 32; k < storage.size(); ++k) {
      oss << storage[k] << " ";
    }
    oss << std::endl;
  }
  return oss.str();
}
inline void CheckKernel(OpKernelContext* ctx, const std::string& msg) {
  auto d = ctx->eigen_gpu_device();
  d.synchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(FATAL) << "Kernel Failed!" << msg << " err=" << cudaGetErrorString(err);
  } else {
    if (VLOG_IS_ON(3)) {
      LOG(INFO) << "Pass " << msg;
      // }else{
      //   LOG(INFO)<<"VLog is off but pass "<<msg;
    }
  }
}
}  // namespace BNMS
}  // namespace tensorflow
#endif
#endif