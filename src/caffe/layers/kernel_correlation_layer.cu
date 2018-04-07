/*
 * kernel_correlation_layer.cu
 *
 *  Created on: Oct 16, 2017
 *      Author: cfeng
 */

#include <vector>
#include <stdio.h>
#include <cstring>
#include <assert.h>

#include "caffe/layers/kernel_correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

template<typename Dtype>
__global__ void kernel_correlation_forward(const Dtype* const P,
                                           const Dtype* const indptr,
                                           const Dtype* const indices,
                                           const Dtype* const K,
                                           const Dtype sigma, const int N,
                                           const int D, const int L,
                                           const int M, Dtype* const C)
{
  const int NL = N * L;
  CUDA_KERNEL_LOOP(sl, NL)  //for each entry of the matrix C
  {
    const int s = sl / L;
    const int l = sl % L;
    const Dtype* q = P + s * D;
    const int ind_begin = (int) indptr[s];
    const int ind_end = (int) indptr[s + 1];
    const int n_nbs = ind_end - ind_begin;

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
        c += expf(-dij / sigma);  //TODO: specialize for double!
      }  //i
    }  //j
    //write operation
    if (n_nbs > 0)
      C[s * L + l] = c / n_nbs;  //average by neighborhood size //TODO: what about average by M*n_nbs
    else
      C[s * L + l] = Dtype(0.0);
  }
}

template<typename Dtype>
void KernelCorrelationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const int N = bottom[0]->shape(0);  // number of points
  const int D = this->d_;             // dimension of point
  const int L = this->num_output_;    // number of output channels
  const int M = this->num_points_per_kernel_;

  const Dtype* P = bottom[0]->gpu_data();         //[N,D]
  const Dtype* indptr = bottom[1]->gpu_data();    //[N+1,]
  const Dtype* indices = bottom[2]->gpu_data();   //[nnz,]
  const Dtype* K = this->blobs_[0]->gpu_data();    //[L,M,D]
  Dtype* C = top[0]->mutable_gpu_data();          //[N,L]

  const int NL = N * L;
  kernel_correlation_forward<Dtype><<<CAFFE_GET_BLOCKS(NL), CAFFE_CUDA_NUM_THREADS>>>(
      P,
      indptr, indices,
      K,
      sigma_,
      N,D,L,M,
      C
  );

  CUDA_POST_KERNEL_CHECK
  ;
}  // forward_gpu

template<typename Dtype>
__global__ void kernel_correlation_backward(const Dtype* const P,
                                            const Dtype* const indptr,
                                            const Dtype* const indices,
                                            const Dtype* const K,
                                            const Dtype* const dC,
                                            const Dtype sigma, const int N,
                                            const int D, const int L,
                                            const int M, Dtype* const dK)
{
  const int LMD = L * M * D;
  const int MD = M * D;
  CUDA_KERNEL_LOOP(lid, LMD)  //for each entry of the matrix dK
  {
    const int l = lid / MD;
    Dtype* dk = dK + l * MD;       //[M,D]
    const Dtype* k = K + l * MD;   //[M,D]

    const int id = lid % MD;
    const int i = id / D;
    const int d = id % D;
    Dtype* dk_i = dk + i * D;       //[D,]
    const Dtype* k_i = k + i * D;   //[D,]

    for (int s = 0; s < N; ++s)  //each output point
    {
      //loss' derivative w.r.t. s-th point to l-th kernel's correlation
      const Dtype dl_dcs = dC[s * L + l];
      const Dtype* q = P + s * D;  //[D,]
      const int ind_begin = (int) indptr[s];
      const int ind_end = (int) indptr[s + 1];
      const int n_nbs = ind_end - ind_begin;

      Dtype dcs_dki(0.0);
      for (int j = ind_begin; j < ind_end; ++j)  //each neighbor point
      {
        const Dtype* p = P + static_cast<int>(indices[j]) * D;

        Dtype dij(0.0);
        for (int dd = 0; dd < D; ++dd)
        {
          Dtype vij = k_i[dd] + q[dd] - p[dd];
          dij += vij * vij;
        }
        Dtype fij = expf(-dij / sigma);  //TODO: specialize for double!

        Dtype vij = k_i[d] + q[d] - p[d];
        dcs_dki += fij * vij;
      }  //j
      dcs_dki *= (Dtype) (-2.0) / (sigma * n_nbs); //TODO: what about average by M*n_nbs
      dk_i[d] += dl_dcs * dcs_dki;  //write operation
    }  //s
  }  //lid
}

template<typename Dtype>
__global__ void kernel_correlation_backward_core(
    const int  L, const int M, const int D, const int N,
    const Dtype sigma,
    const Dtype* P,//[N,D]
    const Dtype* K,//[L,M,D]
    const Dtype* dC,//[N,L]
    const Dtype* indptr, const Dtype* indices,
    Dtype* tmp//[L,M,D,N]
    )
{
  const int DN = D*N;
  const int MDN = M*DN;
  const int MD = M*D;
  const int LMDN = L*MDN;
  CUDA_KERNEL_LOOP(lids, LMDN)  //
  {
    const int l = lids/MDN;
    const int ids = lids%MDN;
    const Dtype* k = K+l*MD;//[M,D]
    Dtype* tmp_l = tmp + l*MDN; //[M,D,N]

    const int i = ids/DN;
    const int ds = ids%DN;
    const Dtype* k_i = k+i*D;//[D,]
    Dtype* tmp_l_i = tmp_l + i*DN; //[D,N]

    const int d = ds/N;
    const int s = ds%N;

    Dtype* tmp_d_s = tmp_l_i + d*N + s; //[D,N]
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
      Dtype fij=expf(-dij/sigma); //TODO: add support for double

      Dtype vij = k_i[d] + q[d] - p[d];
      dcs_dki += fij*vij;
    }  //j
    if (n_nbs>0)
    { //write operation
      dcs_dki *= (Dtype)(-2.0)/(sigma*n_nbs); //TODO: what about average by M*n_nbs
      *tmp_d_s = dl_dcs*dcs_dki;
    }
    else
    {
      *tmp_d_s = Dtype(0.0);
    }
  }//ds
}

template<typename Dtype>
void KernelCorrelationLayer<Dtype>::Backward_gpu(
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

  const Dtype* P = bottom[0]->gpu_data();//[N,D]
  const Dtype* indptr = bottom[1]->gpu_data();//[N+1,]
  const Dtype* indices = bottom[2]->gpu_data();//[nnz,]
  const Dtype* K = this->blobs_[0]->gpu_data();//[L,M,D]

  const Dtype* dC = top[0]->gpu_diff();//[N,L]
  Dtype* dK = this->blobs_[0]->mutable_gpu_diff();//[L,M,D]

  Dtype* tmp = this->tmp_dk_i.mutable_gpu_data(); //[L,M,D,N]

  const int ND = N*D;
  const int LMDN = L*M*ND;
  kernel_correlation_backward_core<Dtype><<<CAFFE_GET_BLOCKS(LMDN), CAFFE_CUDA_NUM_THREADS>>>(
      L,M,D,N,
      sigma_,
      P, K,
      dC, indptr, indices,
      tmp
  );
  CUDA_POST_KERNEL_CHECK;

  const Dtype* tmp_mult = tmp_multiplier.gpu_data();
  for(int l=0; l<L; ++l)//each kernel
  {
    Dtype* dk = dK+l*M*D;         //[M,D]
    Dtype* tmp_l = tmp + l*M*ND;  //[M,D,N]

    for(int i=0; i<M; ++i)//each kernel point
    {
      Dtype* dk_i = dk+i*D;    //[D,]
      Dtype* tmp_l_i = tmp_l + i*ND; //[D,N]

      //dk_i = tmp * 1_N' ([DxN]*[Nx1]=[Dx1])
      caffe_gpu_gemv<Dtype>(
          CblasNoTrans, D, N,
          (Dtype)1., tmp_l_i, tmp_mult, (Dtype)1.,
          dk_i);
    }  //i
  }  //l
}  // backward_gpu

INSTANTIATE_LAYER_GPU_FUNCS(KernelCorrelationLayer);

}  // namespace caffe
