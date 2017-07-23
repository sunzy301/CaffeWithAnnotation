#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */
/* 
从共享的资源读取数据然后排队输入到数据层，每个资源创建单个线程，即便是使用多个GPU在并行任务中求解。
这就保证对于频繁读取数据库，并且每个求解的线程使用的子数据是不同的。
数据成功设计就是这样使在求解时数据保持一种循环地并行训练。 
*/  
class DataReader {
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();

  inline BlockingQueue<Datum*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<Datum*>& full() const {
    return queue_pair_->full_;
  }

 protected:
  // Queue pairs are shared between a body and its readers
  class QueuePair {
   public:
    explicit QueuePair(int size);
    ~QueuePair();

    BlockingQueue<Datum*> free_;
    BlockingQueue<Datum*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  // A single body is created per source
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
    void InternalThreadEntry();
    void read_one(db::Cursor* cursor, QueuePair* qp);

    const LayerParameter param_;
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

    friend class DataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  // 队列
  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<DataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
