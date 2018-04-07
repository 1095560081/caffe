/*
 * kernel_correlation_layer.cpp
 *
 *  Created on: Oct 16, 2017
 *      Author: cfeng
 */

#include <vector>
#include <stdio.h>
#include <cstring>
#include <assert.h>

#include "caffe/filler.hpp"
#include "caffe/layers/kernel_correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

template<typename Dtype>
void KernelCorrelationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  // disable weight_decay
  ::caffe::ParamSpec* the_param(0);
  if (this->layer_param_.param_size()==0) {
    the_param = this->layer_param_.add_param();
  } else {
    the_param = this->layer_param_.mutable_param(0);
    LOG(INFO) << "force weight decay of KernelCorrelationLayer to be 0!";
  }
  the_param->set_decay_mult(0);

  //check P
  CHECK_EQ(bottom[0]->num_axes(), 2)<< "bottom[0] should always have shape (n,d)!";
  this->d_ = bottom[0]->shape(1);
  CHECK_GT(d_, 0)
  << "incorrect dimension d_!";

  //l: number of output/kernels
  num_output_ = this->layer_param_.kernel_correlation_param().num_output();
  //m: number of points per kernel
  num_points_per_kernel_ = this->layer_param_.kernel_correlation_param().num_points_per_kernel();
  sigma_ = this->layer_param_.kernel_correlation_param().sigma();

  CHECK_GT(num_output_, 0)
  << "num_output_ should be larger than 0!";
  CHECK_GT(num_points_per_kernel_, 0)
  << "num_points_per_kernel_ should be larger than 0!";
  CHECK_GT(sigma_, 0)
  << "sigma_ should be larger than 0!";

  if (this->blobs_.size() > 0)
  {
    LOG(INFO)<< "Skipping parameter initialization";
  }
  else
  {
    this->blobs_.resize(1);

    //initialize K, all kernels
    vector<int> kernel_shape(3);
    kernel_shape[0]=num_output_;
    kernel_shape[1]=num_points_per_kernel_;
    kernel_shape[2]=d_;
    this->blobs_[0].reset(new Blob<Dtype>(kernel_shape));

    shared_ptr<Filler<Dtype> > kernel_filler(GetFiller<Dtype>(
            this->layer_param_.kernel_correlation_param().kernel_filler()));
    kernel_filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  Reshape(bottom, top);
}  // setup

template<typename Dtype>
void KernelCorrelationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top)
{
  if (bottom.size() != 3)
  {
    LOG(ERROR)<< "number of bottoms to KernelCorrelationLayer should be 3, instead of "
    << bottom.size() << "!";
  }

  CHECK_EQ(d_, bottom[0]->shape(1))
  << "dimension d_ does not match with bottom[0]!";

  //check P
  CHECK_EQ(bottom[0]->num_axes(), 2)
  << "bottom[0] should always have shape (n,d)!";
  //check indptr
  CHECK_EQ(bottom[1]->num_axes(), 1)
  << "bottom[1] should always be an vector!";
  CHECK_EQ(bottom[1]->shape(0), bottom[0]->shape(0)+1)
  << "bottom[1] (indptr) does not match bottom[0]!";
  //check indices
  CHECK_EQ(bottom[2]->num_axes(), 1)
  << "bottom[2] (indices) should always be an vector!";

  vector<int> shape(2);
  shape[0] = bottom[0]->shape(0);// n
  shape[1] = this->num_output_;// l
  top[0]->Reshape(shape);

  vector<int> shape4(4);
  shape4[0] = this->num_output_;
  shape4[1] = this->num_points_per_kernel_;
  shape4[2] = this->d_;
  shape4[3] = bottom[0]->shape(0);  // number of points
  this->tmp_dk_i.Reshape(shape4); //[L,M,D,N]

  shape[0] = bottom[0]->shape(0);  // number of points
  shape[1] = 1;
  this->tmp_multiplier.Reshape(shape); //[N,1]
  caffe_set(shape[0], Dtype(1), tmp_multiplier.mutable_cpu_data());
}  // reshape

template<typename Dtype>
void KernelCorrelationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const int N = bottom[0]->shape(0);  // number of points
  const int D = this->d_;             // dimension of point
  const int L = this->num_output_;    // number of output channels
  const int M = this->num_points_per_kernel_;

  const Dtype* P = bottom[0]->cpu_data();         //[N,D]
  const Dtype* indptr = bottom[1]->cpu_data();    //[N+1,]
  const Dtype* indices = bottom[2]->cpu_data();   //[nnz,]
  const Dtype* K = this->blobs_[0]->cpu_data();   //[L,M,D]
  Dtype* C = top[0]->mutable_cpu_data();          //[N,L]

  for (int s = 0; s < N; ++s)  //each point
  {
    const Dtype* q = P + s * D;
    const int ind_begin = (int) indptr[s];
    const int ind_end = (int) indptr[s + 1];
    const int n_nbs = ind_end - ind_begin;

    for (int l = 0; l < L; ++l)  //each kernel
    {
      const Dtype* k = K + l * M * D;

      Dtype c(0.0);
      for (int j = ind_begin; j < ind_end; ++j)  //each neighbor point
      {
        const Dtype* p = P + static_cast<int>(indices[j]) * D;

        for (int i = 0; i < M; ++i)  //each kernel point
        {
          const Dtype* k_i = k + i * D;

          Dtype dij(0.0);
          for (int d = 0; d < D; ++d)
          {
            Dtype vij = k_i[d] + q[d] - p[d];
            dij += vij * vij;
          }
          c += std::exp(-dij / sigma_);
        }  //i
      }  //j
      //write operation
      if (n_nbs > 0)
        C[s * L + l] = c / n_nbs;  //average by neighborhood size //TODO: what about average by M*n_nbs
      else
        C[s * L + l] = Dtype(0.0);
    }  //l
  }  //s
}  // forward_cpu

template<typename Dtype>
void KernelCorrelationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
  CHECK_EQ(propagate_down[0], false)<<"Backward to bottom[0] is not supported currently!";
  CHECK_EQ(propagate_down[1], false)<<"Backward to bottom[1] is not supported currently!";
  CHECK_EQ(propagate_down[2], false)<<"Backward to bottom[2] is not supported currently!";

  if (!this->param_propagate_down_[0]) return;

  const int N = bottom[0]->shape(0);  // number of points
  const int D = this->d_;// dimension of point
  const int L = this->num_output_;// number of output channels
  const int M = this->num_points_per_kernel_;

  const Dtype* P = bottom[0]->cpu_data();//[N,D]
  const Dtype* indptr = bottom[1]->cpu_data();//[N+1,]
  const Dtype* indices = bottom[2]->cpu_data();//[nnz,]
  const Dtype* K = this->blobs_[0]->cpu_data();//[L,M,D]

  const Dtype* dC = top[0]->cpu_diff();//[N,L]
  Dtype* dK = this->blobs_[0]->mutable_cpu_diff();//[L,M,D]

  for(int l=0; l<L; ++l)//each kernel
  {
    Dtype* dk = dK+l*M*D;    //[M,D]
    const Dtype* k = K+l*M*D;//[M,D]

    for(int i=0; i<M; ++i)//each kernel point
    {
      Dtype* dk_i = dk+i*D;    //[D,]
      const Dtype* k_i = k+i*D;//[D,]

      for(int d=0; d<D; ++d)
      {
        for(int s=0; s<N; ++s)  //each output point
        {
          //loss' derivative w.r.t. s-th point to l-th kernel's correlation
          const Dtype dl_dcs = dC[s*L+l];
          const Dtype* q = P+s*D;//[D,]
          const int ind_begin = (int) indptr[s];
          const int ind_end = (int) indptr[s + 1];
          const int n_nbs = ind_end - ind_begin;

          Dtype dcs_dki(0.0);
          for(int j=ind_begin; j<ind_end; ++j)//each neighbor point
          {
            const Dtype* p = P+static_cast<int>(indices[j])*D;

            Dtype dij(0.0);
            for(int dd=0; dd<D; ++dd)
            {
              Dtype vij = k_i[dd] + q[dd] - p[dd];
              dij += vij*vij;
            }
            Dtype fij=std::exp(-dij/sigma_);

            Dtype vij = k_i[d] + q[d] - p[d];
            dcs_dki += fij*vij;
          }  //j
          if (n_nbs>0)
          { //write operation
            dcs_dki *= (Dtype)(-2.0)/(sigma_*n_nbs); //TODO: what about average by M*n_nbs
            dk_i[d] += dl_dcs*dcs_dki;
          }
        }  //s
      }  //d
    }  //i
  }  //l
}  // backward_cpu

#ifdef CPU_ONLY
STUB_GPU(KernelCorrelationLayer);
#endif

INSTANTIATE_CLASS(KernelCorrelationLayer);
REGISTER_LAYER_CLASS(KernelCorrelation);

}  // namespace caffe
