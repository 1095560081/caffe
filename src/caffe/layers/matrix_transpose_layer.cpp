#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matrix_transpose_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//////////////////////////////////////////////////////////////////////////////

#define MATRIX_TRANSPOSE_USE_OPENMP 0

//template<typename Dtype>
//void caffe_cpu_geam_inplace(const int M, const int N, Dtype* A)

//A<NxM>, B=A^T <MxN>
template<typename Dtype>
void caffe_cpu_geam(const int M, const int N, const Dtype* A, Dtype* B)
{
  CHECK(A!=B) << "MatrixTranspose layer does not support in-place operation yet!";
//  if (A==B) {
//    caffe_cpu_geam_inplace(M, N, B);
//    return;
//  }

  const int MN=M*N;
#if MATRIX_TRANSPOSE_USE_OPENMP
#pragma omp parallel for
#endif
  for(int ij=0; ij<MN; ++ij)
  {
    const int i=ij/N;
    const int j=ij%N;
    const int ji=j*M+i;
    B[ij]=A[ji];
  }
}

//////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void MatrixTransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]!=top[0]) << "MatrixTranspose layer does not support in-place operation yet!";
  Reshape(bottom, top);
}

template <typename Dtype>
void MatrixTransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->num_axes()>=2)
    << "X blob must be of shape (B[,B2,...,Bn],M,N) or (M,N)!";

  const int row_axis = bottom[0]->CanonicalAxisIndex(-2);
  const int col_axis = bottom[0]->CanonicalAxisIndex(-1);
  const int Rx = bottom[0]->shape(row_axis);
  const int Cx = bottom[0]->shape(col_axis);
  M_ = Rx;
  N_ = Cx;

  vector<int> top_shape = bottom[0]->shape();
  top_shape[top_shape.size()-1] = M_;
  top_shape[top_shape.size()-2] = N_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MatrixTransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X_data = bottom[0]->cpu_data();
  Dtype* Z_data = top[0]->mutable_cpu_data();

  const bool X_hasbatch = (bottom[0]->num_axes()>2);
  const bool Z_hasbatch = (top[0]->num_axes()>2);
  const int B = Z_hasbatch ? top[0]->count(0, top[0]->num_axes()-2) : 1;
  const int X_stride = M_ * N_;
  const int Z_stride = N_ * M_;
  for(int b=0; b<B; ++b) {
    caffe_cpu_geam<Dtype>(
      N_, M_,
      X_data+b*X_stride*int(X_hasbatch),
      Z_data+b*Z_stride*int(Z_hasbatch));
  }
}

template <typename Dtype>
void MatrixTransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) return;

  const Dtype* Z_diff = top[0]->cpu_diff();
  Dtype* X_diff = bottom[0]->mutable_cpu_diff();

  const bool X_hasbatch = (bottom[0]->num_axes()>2);
  const bool Z_hasbatch = (top[0]->num_axes()>2);
  const int B = Z_hasbatch ? top[0]->count(0, top[0]->num_axes()-2) : 1;
  const int X_stride = M_ * N_;
  const int Z_stride = N_ * M_;
  for(int b=0; b<B; ++b) {
    caffe_cpu_geam<Dtype>(
      M_, N_,
      Z_diff+b*Z_stride*int(Z_hasbatch),
      X_diff+b*X_stride*int(X_hasbatch));
  }
}

#ifdef CPU_ONLY
STUB_GPU(MatrixTransposeLayer);
#endif

INSTANTIATE_CLASS(MatrixTransposeLayer);
REGISTER_LAYER_CLASS(MatrixTranspose);

}  // namespace caffe
