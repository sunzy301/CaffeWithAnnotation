#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
// 卷积层与反卷积层的基类，直接集成自Layer类 
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  // 构造函数
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  // 实际进行初始化函数    
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  // cpu的相关操作
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);
  
  // gpu的相关操作
#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  // 卷积操作的参数
  /// @brief The spatial dimensions of a filter kernel.
  // kernel的每个维度大小
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  // 每个维度stride大小，就是kernel在每个维度上隔多少像素进行一次卷积操作
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  // 每个维度stride大小，卷积之后每个维度需要补充的像素
  Blob<int> pad_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  // 空间维度的数量，空间维度指的是xy这样的空间坐标系
  int num_spatial_axes_;
  // bottom blob中一个样本数据长度，是channel*h*w的数值
  int bottom_dim_;
  // top blob中一个样本数据长度，是channel*h*w的数值
  int top_dim_;
 
  // 输入的blob中channel所在的维度
  int channel_axis_;
  // 一个batch中有多少数据 batchsize
  int num_;
  // 输入的channel维度数量
  int channels_;
  // 卷积组的大小
  int group_;
  // 输出空间维度 = 卷积之后的图像长*卷积之后图像的宽
  int out_spatial_dim_;
  int weight_offset_;
  // 输出的channel数量
  int num_output_;
  // 是否启用偏置 
  bool bias_term_;
  // 是否是1*1卷积标志
  bool is_1x1_;
  // 强制使用n维通用卷积 
  bool force_nd_im2col_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  // 一系列包装器，可以不再需要记忆很长的参数列表
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          data);
    }
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  // 卷积输出channel数量
  int conv_out_channels_;
  // 卷积输入channel数量
  int conv_in_channels_;
  // 卷积的输出的空间维度 = 卷积后图像h*卷积后图像w
  int conv_out_spatial_dim_;
  // 卷积核的维度 = 输入图像的维度*卷积核的h*卷积核的w 
  int kernel_dim_;
  int col_offset_;
  int output_offset_;
  // im2col的时候使用的存储空间 
  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
