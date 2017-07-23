#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// 析构函数
SyncedMemory::~SyncedMemory() {
  // 释放cpu内存
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

  // 释放gpu内存
#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
#endif  // CPU_ONLY
}

// 将数据转到cpu中
inline void SyncedMemory::to_cpu() {
  switch (head_) {
  // 未分配内存
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
// 数据在GPU中
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    // 从GPU复制到cpu中
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  // 数据已经同步了或者在cpu上有数据，就不需要进行任何操作
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

// 将数据转到gpu中
inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  // 未分配内存
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  // 数据在cpu中
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  // 数据已经同步了或者在gpu上有数据，就不需要进行任何操作
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

// 返回不可修改的cpu内存指针
const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

// 设置cpu数据
void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  // 如果当前cpu指针已经有指向了，将其释放
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  // 直接将指针指向新的位置，不对数据进行深度复制
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

// 返回不可修改的gpu内存指针
const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

// 设置gpu数据
void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

// 返回可以修改的指针，head状态会发生改变，因为另一端的数据已经不再是同步了
void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

}  // namespace caffe

