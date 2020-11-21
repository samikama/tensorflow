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

Status DoNMSBatchedGPUJagged(
    OpKernelContext* context, const Tensor& boxes, const Tensor& scores,
    const Tensor& box_counts_tensor, const int max_output_size,
    const Eigen::half iou_threshold_val, const Eigen::half score_threshold,
    bool pad_to_max_output, int* num_saved_outputs, Tensor** output_indices,
    int kernel = 0, bool pre_sorted_inputs = false);
namespace BatchedNMS {
struct __align__(16) Box {
  float x1, y1, x2, y2;
};

struct __align__(8) HalfBox {
  Eigen::half x1;
  Eigen::half y1;
  Eigen::half x2;
  Eigen::half y2;
};

template <typename T>
DataType BoxDataType();
template <>
DataType BoxDataType<Box>() {
  return DataType::DT_FLOAT;
}
template <>
DataType BoxDataType<HalfBox>() {
  return DataType::DT_HALF;
}
template <typename T>
struct ToCubType;
template <>
struct ToCubType<Eigen::half> {
  typedef __half Type;
};

template <>
struct ToCubType<float> {
  typedef float Type;
};

template <typename Index>
__device__ __forceinline__ void SelectHelper(const Index i_selected,
                                             const Index i_original) {}

template <typename Index, typename T, typename... Args>
__device__ __forceinline__ void SelectHelper(const Index i_selected,
                                             const Index i_original,
                                             const T* original, T* selected,
                                             Args... args) {
  selected[i_selected] = original[i_original];
  BatchedNMS::SelectHelper(i_selected, i_original, args...);
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
    // if(threadIdx.x==0){
    //   printf("b=%d is=%d os=%d %s\n",y,istride,ostride,__PRETTY_FUNCTION__);
    // }
    for (const int idx : GpuGridRangeX(num_elements[y])) {
      BatchedNMS::SelectHelper(idx + ostride, istride + indices[idx + istride],
                               original, selected, args...);
    }
  }
}

template <typename T>
std::string dumpDeviceTensor(const std::string& Name, const Tensor& t,
                             OpKernelContext* context, size_t count = 0) {
  std::vector<T> storage(t.NumElements(), 0);
  auto d = context->eigen_gpu_device();
  d.memcpyDeviceToHost(storage.data(), t.flat<T>().data(),
                       t.NumElements() * sizeof(T));
  d.synchronize();
  std::stringstream oss;
  oss << context->op_kernel().name() << ": Dumping " << Name
      << " NumElements=" << t.NumElements() ;
  size_t num_to_print = storage.size();
  if(count > 0){
    num_to_print=min(count, storage.size());
    oss<<", asked="<<count<<", dumping="<<num_to_print;
  }
  oss<< std::endl;
  for (size_t l = 0; l < num_to_print / 32; ++l) {
    oss << "l= " << l << " [ " << l * 32 << " ] = ";
    for (size_t k = 0; k < 32; ++k) {
      oss << storage[l * 32 + k] << " ";
    }
    oss << std::endl;
  }
  if ((num_to_print / 32) * 32 < num_to_print) {
    oss << "l= " << (num_to_print / 32) << " [ " << (num_to_print / 32) * 32
        << " ] = ";
    for (int k = (num_to_print / 32) * 32; k < num_to_print; ++k) {
      oss << storage[k] << " ";
    }
    oss << std::endl;
  }
  return oss.str();
}

template <typename T>
std::string dumpDeviceTensor(const std::string& Name, const int* t,
                             const int num_elements, OpKernelContext* context,
                             size_t count = 0) {
  std::vector<int> storage(num_elements, 0);
  auto d = context->eigen_gpu_device();
  d.memcpyDeviceToHost(storage.data(), t, num_elements * sizeof(T));
  d.synchronize();
  std::stringstream oss;
  oss << context->op_kernel().name() << ": Dumping " << Name
      << " NumElements=" << num_elements ;
  size_t num_to_print = storage.size();
  if(count > 0){
    num_to_print=min(count, storage.size());
    oss<<", asked="<<count<<", dumping="<<num_to_print;
  }
  oss<< std::endl;
  for (size_t l = 0; l < num_to_print / 32; ++l) {
    oss << "l= " << l << " [ " << l * 32 << " ] = ";
    for (size_t k = 0; k < 32; ++k) {
      oss << storage[l * 32 + k] << " ";
    }
    oss << std::endl;
  }
  if ((num_to_print / 32) * 32 < num_to_print) {
    oss << "l= " << (num_to_print / 32) << " [ " << (num_to_print / 32) * 32
        << " ] = ";
    for (int k = (num_to_print / 32) * 32; k < num_to_print; ++k) {
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
}  // namespace BatchedNMS
}  // namespace tensorflow
#endif
#endif