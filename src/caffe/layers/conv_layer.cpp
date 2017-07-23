#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

// 具体实现Layer类中的Forward与Backward方法
// 每个方法都有cpu与gpu的不同版本


template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 获取权重
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    // bottom blob指针
    const Dtype* bottom_data = bottom[i]->cpu_data();
    // top blob可修改指针
    Dtype* top_data = top[i]->mutable_cpu_data();
    // 对于batch中每个样本进行计算
    for (int n = 0; n < this->num_; ++n) {
      // 调用具体的计算函数
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      // 如果需要，就计算偏置
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // 权重参数
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // 获取权重blob的diff,也就是梯度的数据可修改指针
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    // 计算偏置的梯度
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    // 计算权重的梯度
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      // 对于batch中每一个样本计算
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        // 计算权重梯度，每个样本的梯度会被累加起来
        // 保存在weight_diff中
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        
        // gradient w.r.t. bottom data, if necessary.
        // 如果需要，计算反向bottom的梯度用于继续向下传播
        // 保存在bottom blob的diff中
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
