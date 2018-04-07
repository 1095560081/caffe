#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matrix_transpose_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//////////////////////////////////////////////////////////////////////////////

//A<NxM>, B=A^T <MxN>
template<typename Dtype>
void caffe_gpu_geam(const int M, const int N, const Dtype* A, Dtype* B);

template<>
void caffe_gpu_geam<float>(const int M, const int N, const float* A, float* B)
{
  //cublas follows fortran order/column-major
  int lda=M;
  int ldb=N;
  cublasOperation_t cuTransA = CUBLAS_OP_T;
  cublasOperation_t cuTransB = CUBLAS_OP_N;
  const float alpha = 1.0;
  const float beta = 0.0;
  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), cuTransA, cuTransB,
    N, M,
    &alpha,
    A, lda,
    &beta,
    B, ldb,
    B, ldb));
}

template<>
void caffe_gpu_geam<double>(const int M, const int N, const double* A, double* B)
{
  //cublas follows fortran order/column-major
  int lda=M;
  int ldb=N;
  cublasOperation_t cuTransA = CUBLAS_OP_T;
  cublasOperation_t cuTransB = CUBLAS_OP_N;
  const double alpha = 1.0;
  const double beta = 0.0;
  CUBLAS_CHECK(cublasDgeam(Caffe::cublas_handle(), cuTransA, cuTransB,
    N, M,
    &alpha,
    A, lda,
    &beta,
    B, ldb,
    B, ldb));
}

//////////////////////////////////////////////////////////////////////////////

template<typename Dtype>
__global__ void small_matrix_transpose(
    const Dtype* const SRC,
    const int B, const int Msrc, const int Nsrc,
    Dtype* const DST)
{
  const int MN = Msrc*Nsrc;
  CUDA_KERNEL_LOOP(b, B)  //for each batch
  {
    const Dtype* src = SRC + b * MN;
    Dtype* dst = DST + b * MN;

    for (int m=0; m<Msrc; ++m) {
      for(int n=0; n<Nsrc; ++n) {
        dst[n*Msrc+m] = src[m*Nsrc+n];
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void MatrixTransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X_data = bottom[0]->gpu_data();
  Dtype* Z_data = top[0]->mutable_gpu_data();

  const bool X_hasbatch = (bottom[0]->num_axes()>2);
  const bool Z_hasbatch = (top[0]->num_axes()>2);
  const int B = Z_hasbatch ? top[0]->count(0, top[0]->num_axes()-2) : 1;
  const int X_stride = M_ * N_;
  const int Z_stride = N_ * M_;
  if (B > X_stride) {
    small_matrix_transpose<Dtype><<<CAFFE_GET_BLOCKS(B), CAFFE_CUDA_NUM_THREADS>>>(
        X_data,
        B, M_, N_,
        Z_data
    );

    CUDA_POST_KERNEL_CHECK;
  } else {
    for(int b=0; b<B; ++b) {
      caffe_gpu_geam<Dtype>(
        N_, M_,
        X_data+b*X_stride*int(X_hasbatch),
        Z_data+b*Z_stride*int(Z_hasbatch));
    }
  }//if
}

template <typename Dtype>
void MatrixTransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) return;

  const Dtype* Z_diff = top[0]->gpu_diff();
  Dtype* X_diff = bottom[0]->mutable_gpu_diff();

  const bool X_hasbatch = (bottom[0]->num_axes()>2);
  const bool Z_hasbatch = (top[0]->num_axes()>2);
  const int B = Z_hasbatch ? top[0]->count(0, top[0]->num_axes()-2) : 1;
  const int X_stride = M_ * N_;
  const int Z_stride = N_ * M_;
  if (B>X_stride) {
    small_matrix_transpose<Dtype><<<CAFFE_GET_BLOCKS(B), CAFFE_CUDA_NUM_THREADS>>>(
        Z_diff,
        B, N_, M_,
        X_diff
    );

    CUDA_POST_KERNEL_CHECK;
  } else {
    for(int b=0; b<B; ++b) {
      caffe_gpu_geam<Dtype>(
        M_, N_,
        Z_diff+b*Z_stride*int(Z_hasbatch),
        X_diff+b*X_stride*int(X_hasbatch));
    }
  }//if
}

INSTANTIATE_LAYER_GPU_FUNCS(MatrixTransposeLayer);

}  // namespace caffe
