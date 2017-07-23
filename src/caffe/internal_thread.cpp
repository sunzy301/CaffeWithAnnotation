#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// 析构函数，调用停止内部线程函数  
InternalThread::~InternalThread() {
  StopInternalThread();
}

// 判断线程是否启动了
bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

// 测试线程是否起来
bool InternalThread::must_stop() {
  // 首先thread_指针不能为空，然后该线程是可等待的（joinable）
  return thread_ && thread_->interruption_requested();
}

// 初始化工作
void InternalThread::StartInternalThread() {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  bool root_solver = Caffe::root_solver();

  // 实例化一个thread对象给thread_指针，该线程的执行的是entry函数
  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
          rand_seed, solver_count, root_solver));
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

// 线程所要执行的函数  
void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, bool root_solver) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_root_solver(root_solver);

  // 线程需要实际执行的操作
  InternalThreadEntry();
}

// 停止线程 
void InternalThread::StopInternalThread() {
  // 如果线程已经开始  
  if (is_started()) {
    // 打断线程
    thread_->interrupt();
    try {
      // 等待线程结束
      thread_->join();
    } 
    // 线程自己要结束，就不记录任何错误信息
    catch (boost::thread_interrupted&) {
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();
    }
  }
}

}  // namespace caffe
