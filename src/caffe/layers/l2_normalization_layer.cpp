/*
 * l2_normalization_layer.cpp
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
void L2NormalizationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  this->is_global_ = this->layer_param().l2n_param().is_global();
  if (!this->is_global_)
    this->axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param().l2n_param().axis());
  this->eps_ = std::abs(this->layer_param().l2n_param().eps());
}

template<typename Dtype>
void L2NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top)
{
  if (bottom.size() != 1)
  {
    LOG(ERROR)<< "number of bottoms to L2NormalizationLayer should be 1, instead of "
    << bottom.size() << "!";
  }

  top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void L2NormalizationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const int N = bottom[0]->count();
  const Dtype* X = bottom[0]->cpu_data();
  Dtype* Y = top[0]->mutable_cpu_data();

  if (this->is_global_) {
    const Dtype m = caffe_cpu_dot(N, X, X);
    this->too_small_ = m <= this->eps_;
    if (this->too_small_) {
      caffe_copy(N, X, Y);
      return; //too small, just copy
    }
    this->m_ = m; //squared L2 norm
    Dtype n = (Dtype)1.0/(std::sqrt(m));
    this->n_ = n; //1 over L2 norm
    caffe_cpu_scale(N, n, X, Y);
  } else {
    const int out_num = bottom[0]->count(0, this->axis_);
    const int dim = N / out_num;
    const int in_num = bottom[0]->count(this->axis_+1);
    const int axis_len = bottom[0]->shape(this->axis_);

    for(int o=0; o<out_num; ++o) {
      for(int i=0; i<in_num; ++i) {
        const int offset = o * dim + i;
        const Dtype* x = X + offset;
        Dtype* y = Y + offset;
        const Dtype m = caffe_cpu_strided_dot(axis_len, x, in_num, x, in_num);
        if (m <= this->eps_) {
          caffe_cpu_strided_copy(axis_len, x, in_num, y, in_num);
          continue;
        }
        const Dtype n = (Dtype) 1.0 / (std::sqrt(m));
        caffe_cpu_strided_scale(axis_len, n, x, in_num, y, in_num);
      }//stride
    }//out_num
  }//is_global_
}

template<typename Dtype>
void L2NormalizationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
  if (!propagate_down[0]) return;

  const int N = bottom[0]->count();
  const Dtype* X = bottom[0]->cpu_data();
  const Dtype* dY = top[0]->cpu_diff();
  Dtype* dX = bottom[0]->mutable_cpu_diff();

  if (this->is_global_) {
    if (this->too_small_) {
      caffe_copy(N, dY, dX);
      return;
    }
    //dl/dy=b^T
    Dtype bTx = caffe_cpu_dot(N, dY, X);
    caffe_cpu_axpby(N, -this->n_*bTx/this->m_, X,
                    Dtype(0.0), dX);   //dX <- -n*b^Tx/m x^T
    caffe_axpy(N, this->n_, dY, dX); //dX <- n*b^T + dX
  } else {
    const int out_num = bottom[0]->count(0, this->axis_);
    const int dim = N / out_num;
    const int in_num = bottom[0]->count(this->axis_+1);
    const int axis_len = bottom[0]->shape(this->axis_);

    for(int o=0; o<out_num; ++o) {
      for(int i=0; i<in_num; ++i) {
        const int offset = o * dim + i;
        const Dtype* x = X + offset;
        const Dtype m = caffe_cpu_strided_dot(axis_len, x, in_num, x, in_num);
        const Dtype* dy = dY + offset;
        Dtype* dx = dX + offset;
        if (m <= this->eps_) {
          caffe_cpu_strided_copy(axis_len, dy, in_num, dx, in_num);
          continue;
        }

        const Dtype n = (Dtype) 1.0 / (std::sqrt(m));
        //dl/dy=b^T
        Dtype bTx = caffe_cpu_strided_dot(axis_len, dy, in_num, x, in_num);
        caffe_cpu_strided_axpby(axis_len, -n*bTx/m, x, in_num,
                                Dtype(0.0), dx, in_num);      //dX <- -n*b^Tx/m x^T
        caffe_strided_axpy(axis_len, n, dy, in_num, dx, in_num); //dX <- n*b^T + dX
      }//stride
    }//out_num
  }
}

#ifdef CPU_ONLY
STUB_GPU(L2NormalizationLayer);
#endif

INSTANTIATE_CLASS(L2NormalizationLayer);
REGISTER_LAYER_CLASS(L2Normalization);

}//caffe
