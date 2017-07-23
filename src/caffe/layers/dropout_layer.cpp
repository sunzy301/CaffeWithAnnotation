// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 调用基类的LayerSetUp
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  // 设置drop概率
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  // 成绩倍率
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

// forward前向计算
template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // bottom blob
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // top blob
  Dtype* top_data = top[0]->mutable_cpu_data();
  // 随机mask数据指针
  // 把随机数值保存在矩阵中
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  // 数据数量
  const int count = bottom[0]->count();
  // dropout只用在train过程中，test中不用
  if (this->phase_ == TRAIN) {
    // Create random numbers
    // 产生随机数
    // 二项分布，大于threshold就是1，不然就是0
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

// backward反向梯度计算
template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    // top blob diff
    const Dtype* top_diff = top[0]->cpu_diff();
    // bottom blob diff
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      // mask
      // 读取forward计算时产生的随机数
      const unsigned int* mask = rand_vec_.cpu_data();
      // 数据数量
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        // 计算梯度
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
