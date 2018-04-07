#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReductionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  op_ = this->layer_param_.reduction_param().operation();
  group_ = this->layer_param_.reduction_param().group();
  CHECK(group_!=0) << "group should not be 0!";
}

template <typename Dtype>
void ReductionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (op_ == ReductionParameter_ReductionOp_PROD) {
    if (group_>0)
      CHECK_EQ(bottom[0]->shape(1)%group_, 0) << "input columns is not divisible by group!";

    axis_ = 1; //TODO: read from proto

    CHECK_EQ(bottom[0]->num_axes(), 2) << "input must be a (NxM) matrix!";
    vector<int> top_shape(2);
    top_shape[0]=bottom[0]->shape(0);
    top_shape[1]= group_<0 ? 1 : bottom[0]->shape(1)/group_;
    top[0]->Reshape(top_shape);
    return;
  }

  axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.reduction_param().axis());
  // In the output, we'll keep all axes up to the reduction axis, but
  // throw away any after that.
  // Note: currently reducing along non-tail axes is not supported; otherwise,
  // we'd need to also copy any axes following an "end_axis".
  vector<int> top_shape(bottom[0]->shape().begin(),
                        bottom[0]->shape().begin() + axis_);
  top[0]->Reshape(top_shape);
  num_ = bottom[0]->count(0, axis_);
  dim_ = bottom[0]->count(axis_);
  CHECK_EQ(num_, top[0]->count());
  if (op_ == ReductionParameter_ReductionOp_SUM ||
      op_ == ReductionParameter_ReductionOp_MEAN) {
    vector<int> sum_mult_shape(1, dim_);
    sum_multiplier_.Reshape(sum_mult_shape);
    caffe_set(dim_, Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
  coeff_ = this->layer_param().reduction_param().coeff();
  if (op_ == ReductionParameter_ReductionOp_MEAN) {
    coeff_ /= dim_;
  }
}

template <typename Dtype>
void ReductionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* mult_data = NULL;
  if (sum_multiplier_.count() > 0) {
    mult_data = sum_multiplier_.cpu_data();
  }
  Dtype* top_data = top[0]->mutable_cpu_data();

  if (op_ == ReductionParameter_ReductionOp_PROD) {
    const int N = bottom[0]->shape(0);
    const int M = bottom[0]->shape(1);
    const int K = top[0]->shape(1);
    const int G = M/K;

    for(int i=0; i<N; ++i) {
      for(int j=0; j<K; ++j) {
        top_data[i*K+j]=(Dtype)1.0;
        for(int g=0; g<G; ++g) {
          top_data[i*K+j] *= bottom_data[i*M+j*G+g];
        }
      }
    }

    return;
  }

  for (int i = 0; i < num_; ++i) {
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      *top_data = caffe_cpu_dot(dim_, mult_data, bottom_data);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      *top_data = caffe_cpu_asum(dim_, bottom_data);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      *top_data = caffe_cpu_dot(dim_, bottom_data, bottom_data);
      break;
    default:
      LOG(FATAL) << "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
    }
    bottom_data += dim_;
    ++top_data;
  }
  if (coeff_ != Dtype(1)) {
    // Reset the top_data pointer.
    top_data = top[0]->mutable_cpu_data();
    caffe_scal(num_, coeff_, top_data);
  }
}

template <typename Dtype>
void ReductionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  // Get bottom_data, if needed.
  const Dtype* bottom_data = NULL;
  switch (op_) {
  // Operations that don't need bottom_data
  case ReductionParameter_ReductionOp_SUM:
  case ReductionParameter_ReductionOp_MEAN:
    break;
  // Operations that need bottom_data
  case ReductionParameter_ReductionOp_ASUM:
  case ReductionParameter_ReductionOp_SUMSQ:
  case ReductionParameter_ReductionOp_PROD:
    bottom_data = bottom[0]->cpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  if (op_ == ReductionParameter_ReductionOp_PROD) {
    const Dtype* top_data = top[0]->cpu_data();

    const int N = bottom[0]->shape(0);
    const int M = bottom[0]->shape(1);
    const int K = top[0]->shape(1);
    const int G = M/K;

    for(int i=0; i<N; ++i) {
      for(int j=0; j<M; ++j) {
        const int src = i*M+j;
        const int dst = i*K+j/G;
        bottom_diff[src]=
          top_data[dst]==0 ?
            (Dtype)0.0 :
            top_diff[dst]*top_data[dst]/bottom_data[src];
      }
    }

    return;
  }

  for (int i = 0; i < num_; ++i) {
    const Dtype bottom_coeff = (*top_diff) * coeff_;
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      caffe_set(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      caffe_cpu_sign(dim_, bottom_data, bottom_diff);
      caffe_scal(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      caffe_cpu_scale(dim_, 2 * bottom_coeff, bottom_data, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown reduction op: "
          << ReductionParameter_ReductionOp_Name(op_);
    }
    bottom_data += dim_;
    bottom_diff += dim_;
    ++top_diff;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ReductionLayer);
#endif

INSTANTIATE_CLASS(ReductionLayer);
REGISTER_LAYER_CLASS(Reduction);

}  // namespace caffe
