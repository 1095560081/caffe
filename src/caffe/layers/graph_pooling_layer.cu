#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/graph_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
/////////////////////////////////////////////// Forward
template <typename Dtype>
__global__ void graph_max_pooling(
    const Dtype* const X,
    const Dtype* const indptr,
    const Dtype* const indices,
    const int N, const int D,
    Dtype* const Y,
    int* idx)
{
  const int ND = N*D;
  CUDA_KERNEL_LOOP(ij, ND) //for each entry of the matrix Y
  {
    const int i = ij/D;
    const int j = ij%D;
    const int ind_begin = (int)indptr[i]; //TODO: use reinterpret_cast<int>
    const int ind_end = (int)indptr[i+1]; //TODO: use reinterpret_cast<int>
    Y[ij] = X[ij]; //always start with the signal itself
    idx[ij] = i;
    for(int kk=ind_begin; kk<ind_end; ++kk) //for each neighbor node index
    {
      const int k = (int)indices[kk]; //TODO: use reinterpret_cast<int>
      const int kj = k*D+j;
      if(Y[ij]<X[kj]) {
        Y[ij]=X[kj];
        idx[ij] = k;
      }
    }//for kk
  }//for ij
}

template <typename Dtype>
__global__ void batched_global_max_pooling(
    const Dtype* const X,
    const Dtype* const n_offset,
    const int B, const int D,
    Dtype* const Y,
    int* idx)
{
  const int BD=B*D;
  CUDA_KERNEL_LOOP(bj, BD) //for each dimension of the signal
  {
    const int b = bj/D;
    const int j = bj%D;
    const int ind_begin = b==0 ? 0 : (int)n_offset[b-1]; //TODO: use reinterpret_cast<int>
    const int ind_end = (int)n_offset[b]; //TODO: use reinterpret_cast<int>
    for(int k=ind_begin; k<ind_end; ++k) //for all node index in this data
    {
      const int kj = k*D+j;
      if(k==ind_begin || Y[bj]<X[kj]) {
        Y[bj]=X[kj];
        idx[bj] = k;
      }
    }//for k
  }//for bj
}

template <typename Dtype>
__global__ void batched_global_mean_pooling(
    const Dtype* const X,
    const Dtype* const n_offset,
    const int B, const int D,
    Dtype* const Y)
{
  const int BD=B*D;
  CUDA_KERNEL_LOOP(bj, BD) //for each dimension of the signal
  {
    const int b = bj/D;
    const int j = bj%D;
    const int ind_begin = b==0 ? 0 : (int)n_offset[b-1]; //TODO: use reinterpret_cast<int>
    const int ind_end = (int)n_offset[b]; //TODO: use reinterpret_cast<int>
    Y[bj]=(Dtype)0.0;
    for(int k=ind_begin; k<ind_end; ++k) //for all node index in this data
    {
      const int kj = k*D+j;
      Y[bj]+=X[kj];
    }//for k
    const int nk=ind_end-ind_begin;
    if(nk>0)
      Y[bj]/=(Dtype)nk;
  }//for bj
}

template <typename Dtype>
__global__ void global_max_pooling(
    const Dtype* const X,
    const int N, const int D,
    Dtype* const Y,
    int* idx)
{
  CUDA_KERNEL_LOOP(j, D) //for each dimension of the signal
  {
    for(int i=0; i<N; ++i) //for each node
    {
      const int ij = i*D+j;
      if (i==0 || Y[j]<X[ij]) {
        Y[j] = X[ij];
        idx[j] = i;
      }
    }//for i<N
  }//for j<D
}

template <typename Dtype>
__global__ void global_mean_pooling(
    const Dtype* const X,
    const int N, const int D,
    Dtype* const Y)
{
  CUDA_KERNEL_LOOP(j, D) //for each dimension of the signal
  {
    Y[j]=(Dtype)0.0;
    for(int i=0; i<N; ++i) //for each node
    {
      const int ij = i*D+j;
      Y[j]+=X[ij];
    }//for i<N
    Y[j]/=(Dtype)N;
  }//for j<D
}

template <typename Dtype>
void GraphPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* X = bottom[0]->gpu_data();
  Dtype* Y = top[0]->mutable_gpu_data();

  const int N = bottom[0]->shape(0); //number of nodes for X/Y
  const int D = bottom[0]->shape(1); //dimension of signal
  const int n_bottom = bottom.size();

  if (this->mode_ == GraphPoolingParameter_Mode_MAX)
  {
    int* idx = this->idx_.mutable_gpu_data();

    if(n_bottom==4) //graph max pooling
    {
      const Dtype* indptr = bottom[1]->gpu_data();
      const Dtype* indices = bottom[2]->gpu_data();

      const int ND = N*D;
      graph_max_pooling<Dtype><<<CAFFE_GET_BLOCKS(ND), CAFFE_CUDA_NUM_THREADS>>>(
          X, indptr, indices,
          N, D,
          Y, idx
      );
    }
    else if(n_bottom==2) //batched global max pooling
    {
      const Dtype* n_offset = bottom[1]->gpu_data();
      const int B = bottom[1]->count(); //number of data in the batch

      const int BD = B*D;
      batched_global_max_pooling<Dtype><<<CAFFE_GET_BLOCKS(BD), CAFFE_CUDA_NUM_THREADS>>>(
          X, n_offset,
          B, D,
          Y, idx
      );
    }
    else //global max pooling
    {
      global_max_pooling<Dtype><<<CAFFE_GET_BLOCKS(D), CAFFE_CUDA_NUM_THREADS>>>(
          X,
          N, D,
          Y, idx
      );
    }
  }
  else //average pooling, Y=AX
  {
    if(n_bottom==4)
    {
    const Dtype* indptr = bottom[1]->cpu_data();
    const Dtype* indices = bottom[2]->cpu_data();
    const Dtype* data = bottom[3]->cpu_data();

      const int M = bottom[1]->count()-1; //rows for Y

      CHECK_LT(M, 2147483647)
        << "Current GPU implementation only support number of output nodes up to 2^31-1 due to line 172 of this file!"
           "Reference: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities";
      CHECK_LT(D, 65535)
        << "Current GPU implementation only support signal dimension up to 65535 due to line 172 of this file!"
           "Reference: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities";

    caffe_gpu_csr_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        M, D, N,
      (Dtype)1.0,
      data, indices, indptr,
      X,
      (Dtype)0.0,
      Y,
      CblasRowMajor,
      bottom[3]->count()
    );
  }
    else if(n_bottom==2) //batched global mean pooling
    {
      const Dtype* n_offset = bottom[1]->gpu_data();
      const int B = bottom[1]->count(); //number of data in the batch

      const int BD = B*D;
      batched_global_mean_pooling<Dtype><<<CAFFE_GET_BLOCKS(BD), CAFFE_CUDA_NUM_THREADS>>>(
          X, n_offset,
          B, D,
          Y
      );
    }
    else //global mean pooling
    {
      global_mean_pooling<Dtype><<<CAFFE_GET_BLOCKS(D), CAFFE_CUDA_NUM_THREADS>>>(
          X,
          N, D,
          Y
      );
    }
  }
  CUDA_POST_KERNEL_CHECK;
}

/////////////////////////////////////////////// Backward
template <typename Dtype>
__global__ void graph_max_pooling_backward(
    const Dtype* const dY,
    const Dtype* const indptr,
    const Dtype* const indices,
    const int N, const int D,
    Dtype* const dX,
    const int* const idx)
{
  const int ND=N*D;
  CUDA_KERNEL_LOOP(ij, ND) //for each entry of the signal matrix
  {
    const int i = ij/D;
    const int j = ij%D;
    const int ind_begin = (int)indptr[i]; //TODO: when converting float to int, should we use (int)(float+0.5) for safety?
    const int ind_end = (int)indptr[i+1];
    Dtype gradient = idx[ij]==i ? dY[ij] : (Dtype)0.;
    for(int kk=ind_begin; kk<ind_end; ++kk) //for each neighbor node index
    {
      const int k = (int)indices[kk];
      const int kj = k*D+j;
      if(idx[kj]==i) { //TODO: this only works for undirected graph
        gradient += dY[kj];
      }
    }//for kk
    dX[ij]=gradient;
  }//for ij
}

template <typename Dtype>
__global__ void batched_global_max_pooling_backward(
    const Dtype* const dY,
    const Dtype* const n_offset,
    const int B, const int D,
    Dtype* const dX,
    const int* const idx)
{
  const int BD=B*D;
  CUDA_KERNEL_LOOP(bj, BD) //for each dimension of the signal
  {
    const int b = bj/D;
    const int j = bj%D;
    const int ind_begin = b==0 ? 0 : (int)n_offset[b-1]; //TODO: use reinterpret_cast<int>
    const int ind_end = (int)n_offset[b]; //TODO: use reinterpret_cast<int>
    for(int k=ind_begin; k<ind_end; ++k) //for all node index in this data
    {
      const int kj = k*D+j;
      if(idx[bj]==k) {
        dX[kj] = dY[bj];
        //TODO: break?
      }
    }//for k
  }//for bj
}

template <typename Dtype>
__global__ void batched_global_mean_pooling_backward(
    const Dtype* const dY,
    const Dtype* const n_offset,
    const int B, const int D,
    Dtype* const dX)
{
  const int BD=B*D;
  CUDA_KERNEL_LOOP(bj, BD) //for each dimension of the signal
  {
    const int b = bj/D;
    const int j = bj%D;
    const int ind_begin = b==0 ? 0 : (int)n_offset[b-1]; //TODO: use reinterpret_cast<int>
    const int ind_end = (int)n_offset[b]; //TODO: use reinterpret_cast<int>
    const int nk=ind_end-ind_begin;
    for(int k=ind_begin; k<ind_end; ++k) //for all node index in this data
    {
      const int kj = k*D+j;
      dX[kj] = (Dtype)(1.0/nk)*dY[bj];
    }//for k
  }//for bj
}

template <typename Dtype>
__global__ void global_max_pooling_backward(
    const Dtype* const dY,
    const int D,
    Dtype* const dX,
    const int* const idx)
{
  for(int j=0; j<D; ++j) //for each dimension of the signal
  {
    const int src = j;
    const int dst = idx[src]*D+j;
    dX[dst] += dY[src];
  }
}

template <typename Dtype>
__global__ void global_mean_pooling_backward(
    const Dtype* const dY,
    const int N, const int D,
    Dtype* const dX)
{
  const int ND=N*D;
  CUDA_KERNEL_LOOP(ij, ND)
  {
    const int j=ij%D;
    dX[ij] = (Dtype)(1.0/N) * dY[j];
  }
}

template <typename Dtype>
void GraphPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) return;

  const Dtype* dY = top[0]->gpu_diff();
  Dtype* dX = bottom[0]->mutable_gpu_diff();

  const int N = bottom[0]->shape(0); //number of nodes for X/Y
  const int D = bottom[0]->shape(1); //dimension of signal
  const int n_bottom = bottom.size();

  if (this->mode_ == GraphPoolingParameter_Mode_MAX)
  {
    const int* idx = this->idx_.gpu_data();

    if(n_bottom==4) //graph max pooling
    {
      const Dtype* indptr = bottom[1]->gpu_data();
      const Dtype* indices = bottom[2]->gpu_data();

      const int ND = N*D;
      graph_max_pooling_backward<Dtype><<<CAFFE_GET_BLOCKS(ND), CAFFE_CUDA_NUM_THREADS>>>(
          dY, indptr, indices,
          N, D,
          dX, idx
      );
    }
    else if(n_bottom==2) //batched global max pooling
    {
      caffe_gpu_set(bottom[0]->count(), (Dtype)0., dX);

      const Dtype* n_offset = bottom[1]->gpu_data();
      const int B = bottom[1]->count(); //number of data in the batch

      const int BD = B*D;
      batched_global_max_pooling_backward<Dtype><<<CAFFE_GET_BLOCKS(BD), CAFFE_CUDA_NUM_THREADS>>>(
          dY, n_offset,
          B, D,
          dX, idx
      );
    }
    else //global max pooling
    {
      caffe_gpu_set(bottom[0]->count(), (Dtype)0., dX);

      global_max_pooling_backward<Dtype><<<1, 1>>>(
          dY, D,
          dX, idx
      );
    }
  }
  else
  {
    if(n_bottom==4)
    {
    const Dtype* indptr = bottom[1]->cpu_data();
    const Dtype* indices = bottom[2]->cpu_data();
    const Dtype* data = bottom[3]->cpu_data();

      const int M = bottom[1]->count()-1; //rows for Y
    	

    caffe_gpu_csr_gemm<Dtype>(CblasTrans, CblasNoTrans,
        N, D, M,
      (Dtype)1.0,
      data, indices, indptr,
      dY,
      (Dtype)0.0,
      dX,
      CblasRowMajor,
      bottom[3]->count()
    );
  }
    else if(n_bottom==2) //batched global mean pooling
    {
      const Dtype* n_offset = bottom[1]->gpu_data();
      const int B = bottom[1]->count(); //number of data in the batch

      const int BD = B*D;
      batched_global_mean_pooling_backward<Dtype><<<CAFFE_GET_BLOCKS(BD), CAFFE_CUDA_NUM_THREADS>>>(
          dY, n_offset,
          B, D,
          dX
      );
    }
    else //global max pooling
    {
      const int ND=N*D;
      global_mean_pooling_backward<Dtype><<<CAFFE_GET_BLOCKS(ND), CAFFE_CUDA_NUM_THREADS>>>(
          dY, N, D,
          dX
      );
    }
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(GraphPoolingLayer);


}  // namespace caffe
