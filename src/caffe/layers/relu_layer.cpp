#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {
// relu的cpu版本实现，gpu版本在cu文件中
// forward前向计算
template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // 读取bottom blob数据指针
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // 读取top blob数据可修改指针
  Dtype* top_data = top[0]->mutable_cpu_data();
  // 数据数量
  const int count = bottom[0]->count();
  // 负的斜率
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // 对于每个数据点，计算激活值
  // 使用了址内计算，节约内存
  for (int i = 0; i < count; ++i) {
    // 具体计算公式
    // if x >= 0 then
    //   y = x
    // else
    //   y = negative_slope * x
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

// backward反向梯度计算
template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    // 获取bottom blob data数据指针
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // 获取top blob diff数据指针
    const Dtype* top_diff = top[0]->cpu_diff();
    // 获取bottom blob diff可修改数据指针
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // 数据数量
    const int count = bottom[0]->count();
    // 负的斜率
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      // if x >= 0 then
      //   dl/dx = dl/dy * 1
      // else 
      // dl/dx = dl/dy * negative_slope
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
