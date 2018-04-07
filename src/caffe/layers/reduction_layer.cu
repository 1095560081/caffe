#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void prod_reduction_forward(
    const int N, const int K, const int M, const int G,
    const Dtype* const X,
    Dtype* const Y)
{
  const int NK = N*K;
  CUDA_KERNEL_LOOP(ij, NK) //for each entry of the matrix Y
  {
    const int i = ij/K;
    const int j = ij%K;
    Y[ij]=(Dtype)1.0;
    for(int g=0; g<G; ++g) {
      Y[ij] *= X[i*M+j*G+g];
    }
  }//for ij
}

template <typename Dtype>
__global__ void prod_reduction_backward(
    const int N, const int M, const int K, const int G,
    const Dtype* const X,
    const Dtype* const Y,
    const Dtype* const dY,
    Dtype* const dX)
{
  const int NM = N*M;
  CUDA_KERNEL_LOOP(ij, NM) //for each entry of the matrix dX
  {
    const int i = ij/M;
    const int j = ij%M;
    const int dst = i*K+j/G;
    dX[ij]=
      Y[dst]==0 ?
        (Dtype)0.0 :
        dY[dst]*Y[dst]/X[ij];
  }//for ij
}


template <typename Dtype>
void ReductionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* mult_data = NULL;
  if (sum_multiplier_.count() > 0) {
    mult_data = sum_multiplier_.gpu_data();
  }
  Dtype* top_data = top[0]->mutable_cpu_data();

  if (op_ == ReductionParameter_ReductionOp_PROD) {
    const int N = bottom[0]->shape(0);
    const int M = bottom[0]->shape(1);
    const int K = top[0]->shape(1);
    const int G = M/K;

    Dtype* top_data_gpu = top[0]->mutable_gpu_data();

    const int NK = N*K;
    prod_reduction_forward<Dtype><<<CAFFE_GET_BLOCKS(NK), CAFFE_CUDA_NUM_THREADS>>>(
        N,K,M,G,
        bottom_data,
        top_data_gpu
    );
    CUDA_POST_KERNEL_CHECK;

    return;
  }

  for (int i = 0; i < num_; ++i) {
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      caffe_gpu_dot(dim_, mult_data, bottom_data, top_data);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      caffe_gpu_asum(dim_, bottom_data, top_data);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      caffe_gpu_dot(dim_, bottom_data, bottom_data, top_data);
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
    top_data = top[0]->mutable_gpu_data();
    caffe_gpu_scal(num_, coeff_, top_data);
  }
}

template <typename Dtype>
void ReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
    bottom_data = bottom[0]->gpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (op_ == ReductionParameter_ReductionOp_PROD) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff_gpu = top[0]->gpu_diff();

    const int N = bottom[0]->shape(0);
    const int M = bottom[0]->shape(1);
    const int K = top[0]->shape(1);
    const int G = M/K;

    const int NM = N*M;
    prod_reduction_backward<Dtype><<<CAFFE_GET_BLOCKS(NM), CAFFE_CUDA_NUM_THREADS>>>(
        N,M,K,G,
        bottom_data,
        top_data,
        top_diff_gpu,
        bottom_diff
    );
    CUDA_POST_KERNEL_CHECK;

    return;
  }

  for (int i = 0; i < num_; ++i) {
    const Dtype bottom_coeff = (*top_diff) * coeff_;
    switch (op_) {
    case ReductionParameter_ReductionOp_SUM:
    case ReductionParameter_ReductionOp_MEAN:
      caffe_gpu_set(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_ASUM:
      caffe_gpu_sign(dim_, bottom_data, bottom_diff);
      caffe_gpu_scal(dim_, bottom_coeff, bottom_diff);
      break;
    case ReductionParameter_ReductionOp_SUMSQ:
      caffe_gpu_scale(dim_, 2 * bottom_coeff, bottom_data, bottom_diff);
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

INSTANTIATE_LAYER_GPU_FUNCS(ReductionLayer);

}  // namespace caffe
