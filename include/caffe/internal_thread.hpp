#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
// 为caffe内部代码的多线程提供支持
// 封装了boost thread
// 派生类可以通过实现InternalThreadEntry来达到多线程的能力
class InternalThread {
 public:
  // 构造函数
  InternalThread() : thread_() {}
  // 虚析构函数
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  // caffe的线程局部状态将会使用当前线程值来进行初始化，当前的线程的值有设备id，solver的编号、随机数种子等 
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  // 线程停止方法，是否知道线程退出才返回
  void StopInternalThread();
  
  // 判断线程是否已经起来了
  bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  // 定义了一个虚函数，要求继承该类的必须要实现之
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. */
  // 在当请求退出的时候应该调用该函数 
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);
  
  // 内部指针，指向boost::thread
  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
