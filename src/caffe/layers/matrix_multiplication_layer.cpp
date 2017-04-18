#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matrix_multiplication_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Reshape(bottom, top);
}

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(3, bottom[0]->num_axes())
    << "X blob must be of shape (B,M,K)!";
  CHECK_EQ(3, bottom[1]->num_axes())
    << "Y blob must be of shape (B,K,N)!";
  const int Bx = bottom[0]->shape(0);
  const int By = bottom[1]->shape(0);
  CHECK_EQ(Bx, By) //FixMe: Bx need not be equal to By, since sometimes we want to broadcast
    << "Input batch size not equal ("<<Bx<<"!="<<By<<").";

  const int Rx = bottom[0]->shape(1);
  const int Cx = bottom[0]->shape(2);
  const int Ry = bottom[1]->shape(1);
  const int Cy = bottom[1]->shape(2);
  CHECK_EQ(Cx, Ry)
    << "Input X and Y have incompatible dimensions ("<<Rx<<"x"<<Cx<<" vs. "<<Ry<<"x"<<Cy<<").";
  M_ = Rx;
  K_ = Cx;
  N_ = Cy;
  
  vector<int> top_shape(3);
  top_shape[0]=Bx;
  top_shape[1]=M_;
  top_shape[2]=N_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X_data = bottom[0]->cpu_data();
  const Dtype* Y_data = bottom[1]->cpu_data();
  Dtype* Z_data = top[0]->mutable_cpu_data();

  const int B = bottom[0]->shape(0);
  const int X_stride = M_ * K_;
  const int Y_stride = K_ * N_;
  const int Z_stride = M_ * N_;
  for(int b=0; b<B; ++b) {//TODO: parfor by OpenMP?
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      M_, N_, K_,
      (Dtype)1.,
      X_data+b*X_stride, Y_data+b*Y_stride,
      (Dtype)0.,
      Z_data+b*Z_stride);
  }
}

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* Z_diff = top[0]->cpu_diff();
  const Dtype* Y_data = bottom[1]->cpu_data();
  Dtype* X_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* X_data = bottom[0]->cpu_data();
  Dtype* Y_diff = bottom[1]->mutable_cpu_diff();

  const int B = bottom[0]->shape(0);
  const int X_stride = M_ * K_;
  const int Y_stride = K_ * N_;
  const int Z_stride = M_ * N_;
  for(int b=0; b<B; ++b) {//TODO: parfor by OpenMP?
    if (propagate_down[0]) {
      // dl/dX' = dl/d(XY)' * Y', i.e., bottom[0].diff = top[0].diff * bottom[1].data'
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        M_, K_, N_,
        (Dtype)1.,
        Z_diff+b*Z_stride, Y_data+b*Y_stride,
        (Dtype)0.,
        X_diff+b*X_stride);
    }
    if (propagate_down[1]) {
      // dl/dY' = X' * dl/d(XY)', i.e., bottom[1].diff = bottom[0].data' * top[0].diff
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        K_, N_, M_,
        (Dtype)1.,
        X_data+b*X_stride, Z_diff+b*Z_stride,
        (Dtype)0.,
        Y_diff+b*Y_stride);
    }
  }//for b
}

#ifdef CPU_ONLY
STUB_GPU(MatrixMultiplicationLayer);
#endif

INSTANTIATE_CLASS(MatrixMultiplicationLayer);
REGISTER_LAYER_CLASS(MatrixMultiplication);

}  // namespace caffe
