#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  // 分配GPU内存
  if (Caffe::mode() == Caffe::GPU) {
    // cudaMallocHost是CUDA的api
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
  // 分配CPU内存
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

// 释放内存函数
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  // 释放GPU内存
  if (use_cuda) {
    // cudaFreeHost是CUDA的api
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  // 释放CPU内存
  free(ptr);
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
// 管理内存分配和同步的类
class SyncedMemory {
 public:
  // 构造函数
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}

  // 析构函数
  ~SyncedMemory();

  // 返回指向cpu数据的void类型的指针，用void类型可以管理任意类型的数据，对于具体类型数据只需制定指针类型即可
  // 另外指针时const的，不能通过该指针改变数据
  const void* cpu_data();

  // 将当前cpu_ptr_指向data指向的数据，并将其原来指向的数据（如果存在）释放
  void set_cpu_data(void* data);

  // GPU数据相关函数，与CPU函数性质类似
  const void* gpu_data();
  void set_gpu_data(void* data);

  // 获取可变的数据指针，所以用mutable标明
  void* mutable_cpu_data();
  void* mutable_gpu_data();

  // 枚举类
  // head的状态，前三个分别是，没有初始化，在cpu，在gpu，最后一个表示同步了，说名数据刚从cpu转到gpu，或gpu到cpu
  // 下面的函数要很据这些状态来判断是否同步，怎样同步
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  // 返回head状态
  SyncedHead head() { return head_; }
  // 返回大小
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;

  // 数据的长度
  size_t size_;

  // head状态
  SyncedHead head_;


  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  
  // gpu设备编号
  int gpu_device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
