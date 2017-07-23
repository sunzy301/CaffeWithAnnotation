#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

// 参数设置
template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // pool的设置参数
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  // global pooling设置参数
  global_pooling_ = pool_param.global_pooling();
  // 设置kernel的长宽
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  // 设置pad参数
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  // 设置stride
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  // 检查global pooling情况
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  // 检查pooling模式，有ave和max两种
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // 输入特征维度数量
  channels_ = bottom[0]->channels();
  // 输入高度
  height_ = bottom[0]->height();
  // 输入宽度
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  // 输出高度
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  // 输出宽度
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  // reshape top blob
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  //max pooling 反向求导时要用到取最大值的位置，max_idx_就是记录pooling过程中取max value 的index ，  
    //它是一个int型的blob 和输出top具有相同的shape  
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    // 初始化随机数矩阵，需要记录随机的是哪一个
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

// forward操作
// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom blob data
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // top blob data可修改
  Dtype* top_data = top[0]->mutable_cpu_data();
  // 输出数据数量
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  // 如果top blob向量中有超过一个blob，就在第二个blob中记录mask
  // 不然就在层中记录mask信息
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  // max 最大卷积
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            // pooling窗口位置
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            // 循环变量
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            // 输出index
            const int pool_index = ph * pooled_width_ + pw;
            // 检索pooling窗口
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                // 检索到的位置在输入图像中index
                const int index = h * width_ + w;
                // 寻找最大的最为输出
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        // 计算偏移量，进入下一张图的index起始地址
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  // ave 平均卷积
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            // pooling窗口位置
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            // pooling窗口大小
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            // pooling窗口中输入数值求和
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            // pooling 进行平均操作
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        // 计算偏移量，进入下一张图的index起始地址
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

// backward操作
template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  // top blob diff
  const Dtype* top_diff = top[0]->cpu_diff();
  // bottom blob diff可修改指针
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  // 是否使用top blob中的mask
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  // max 梯度计算
  // 从哪个数据选取的，梯度就反传过去到那个数据点，别的数据不变
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    // 取得mask
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            // 输出图像index
            const int index = ph * pooled_width_ + pw;
            // bottom 对应的index，从mask中读取那时候记录的位置数据
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            // 累加diff
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        // 计算偏移量，进入下一张图的index起始地址
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  // AVE 梯度计算
  // 对应pooling窗口平均分摊反向的梯度
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            // pooling窗口大小
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            // 对应pooling窗口平均分摊反向的梯度
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        // 计算偏移量，进入下一张图的index起始地址
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
