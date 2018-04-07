/*
 * l2_normalization_layer.cu
 *
 *  Created on: Oct 20, 2017
 *      Author: cfeng
 */

#include <algorithm>
#include <vector>

#include "caffe/layers/l2_normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
__global__ void l2_normalization_forward(
    const Dtype* const X,
    const int out_num,
    const int dim,
    const int in_num,
    const int axis_len,
    const Dtype eps,
    Dtype* const Y)
{
  const int OI = out_num * in_num;
  CUDA_KERNEL_LOOP(oi, OI)
  {
    const int o = oi / in_num;
    const int i = oi % in_num;
    const int offset = o * dim + i;
    const Dtype* x = X + offset;
    Dtype* y = Y + offset;
    Dtype m(0.0);
    for(int k=0; k<axis_len; ++k) {
      const Dtype xk = x[k*in_num];
      m += xk*xk;
    }
    if (m<=eps) {
      for(int k=0; k<axis_len; ++k) {
        y[k*in_num] = x[k*in_num];
      }
      continue;
    }
    const Dtype n = (Dtype) 1.0 / (sqrt(m)); //TODO: use sqrtf for float
    for(int k=0; k<axis_len; ++k) {
      y[k*in_num] = x[k*in_num] * n;
    }
  }
}


template<typename Dtype>
void L2NormalizationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const int N = bottom[0]->count();
  const Dtype* X = bottom[0]->gpu_data();
  Dtype* Y = top[0]->mutable_gpu_data();

  if (this->is_global_) {
    Dtype m(0.0);
    caffe_gpu_dot(N, X, X, &m);
    this->too_small_ = m <= this->eps_;
    if (this->too_small_) {
      caffe_copy(N, X, Y);
      return; //too small, just copy
    }
    this->m_ = m; //squared L2 norm
    Dtype n = (Dtype)1.0/(std::sqrt(m));
    this->n_ = n; //1 over L2 norm
    caffe_gpu_scale(N, n, X, Y);
  } else {
    const int out_num = bottom[0]->count(0, this->axis_);
    const int dim = N / out_num;
    const int in_num = bottom[0]->count(this->axis_+1);
    const int axis_len = bottom[0]->shape(this->axis_);

    const int OI = out_num * in_num;
    l2_normalization_forward<Dtype><<<CAFFE_GET_BLOCKS(OI), CAFFE_CUDA_NUM_THREADS>>>(
        X,
        out_num,
        dim,
        in_num,
        axis_len,
        this->eps_,
        Y
    );

    CUDA_POST_KERNEL_CHECK;
  }//is_global_
}

template<typename Dtype>
__global__ void l2_normalization_backward(
    const Dtype* const X,
    const Dtype* const dY,
    const int out_num,
    const int dim,
    const int in_num,
    const int axis_len,
    const Dtype eps,
    Dtype* const dX)
{
  const int OI = out_num * in_num;
  CUDA_KERNEL_LOOP(oi, OI)
  {
    const int o = oi / in_num;
    const int i = oi % in_num;
    const int offset = o * dim + i;
    const Dtype* x = X + offset;
    const Dtype* dy = dY + offset;
    Dtype* dx = dX + offset;
    Dtype m(0.0);
    for(int k=0; k<axis_len; ++k) {
      const Dtype xk = x[k*in_num];
      m += xk*xk;
    }
    if (m<=eps) {
      for(int k=0; k<axis_len; ++k) {
        dx[k*in_num] = dy[k*in_num];
      }
      continue;
    }

    const Dtype n = (Dtype) 1.0 / (sqrt(m)); //TODO: use sqrtf for float
    //dl/dy=b^T
    Dtype bTx(0.0);
    for(int k=0; k<axis_len; ++k) {
      const int k_offset = k*in_num;
      bTx += dy[k_offset] * x[k_offset];
    }
    for(int k=0; k<axis_len; ++k) {
      const int k_offset = k*in_num;
      dx[k_offset] = n*dy[k_offset] -n*bTx/m * x[k_offset]; //dX <- n*b^T -n*b^Tx/m x^T
    }
  }
}


template<typename Dtype>
void L2NormalizationLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
  if (!propagate_down[0]) return;

  const int N = bottom[0]->count();
  const Dtype* X = bottom[0]->gpu_data();
  const Dtype* dY = top[0]->gpu_diff();
  Dtype* dX = bottom[0]->mutable_gpu_diff();

  if (this->is_global_) {
    if (this->too_small_) {
      caffe_copy(N, dY, dX);
      return;
    }
    //dl/dy=b^T
    Dtype bTx(0.0);
    caffe_gpu_dot(N, dY, X, &bTx);
    caffe_gpu_axpby(N, -this->n_*bTx/this->m_, X,
                    Dtype(0.0), dX);   //dX <- -n*b^Tx/m x^T
    caffe_gpu_axpy(N, this->n_, dY, dX); //dX <- n*b^T + dX
  } else {
    const int out_num = bottom[0]->count(0, this->axis_);
    const int dim = N / out_num;
    const int in_num = bottom[0]->count(this->axis_+1);
    const int axis_len = bottom[0]->shape(this->axis_);

    const int OI = out_num * in_num;
    l2_normalization_backward<Dtype><<<CAFFE_GET_BLOCKS(OI), CAFFE_CUDA_NUM_THREADS>>>(
        X,
        dY,
        out_num,
        dim,
        in_num,
        axis_len,
        this->eps_,
        dX
    );

    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L2NormalizationLayer);

}//caffe
