#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// layer的设置参数
template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // 设置sigmoid层的bottom和top blob
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  // 初始化sigmoid层
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // loss layer的reshape
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  // sigmoid layer的reshape
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  // sigmoid的前向计算
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  // bottom blob[0]是输出数据，因为sigmoid是址内计算
  const Dtype* input_data = bottom[0]->cpu_data();
  // bottom blob[1]是已知标记
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  // 每个样本统计loss
  // 公式看上和原始的交叉熵不完全一样，但是这种计算方式更加稳定
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
  // 设置平均的loss值作为输出
  top[0]->mutable_cpu_data()[0] = loss / num;
}

// 反向计算梯度值
template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    // sigmoid输出
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    // 已知样本值
    const Dtype* target = bottom[1]->cpu_data();
    // 每个样本的反向梯度
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // 计算梯度
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
