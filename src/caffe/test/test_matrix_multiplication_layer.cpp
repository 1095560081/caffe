#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/matrix_multiplication_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class MatrixMultiplicationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MatrixMultiplicationLayerTest()
      : blob_bottom_X_(new Blob<Dtype>()),
        blob_bottom_Y_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    // reshape
    vector<int> shape(3);
    shape[0]=2; shape[1]=3; shape[2]=4;
    blob_bottom_X_->Reshape(shape);
    shape[0]=2; shape[1]=4; shape[2]=5;
    blob_bottom_Y_->Reshape(shape);
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_X_);
    filler.Fill(this->blob_bottom_Y_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MatrixMultiplicationLayerTest() {
    delete blob_bottom_X_;
    delete blob_bottom_Y_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_X_;
  Blob<Dtype>* const blob_bottom_Y_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MatrixMultiplicationLayerTest, TestDtypesAndDevices);

TYPED_TEST(MatrixMultiplicationLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_Y_);
  LayerParameter layer_param;
  shared_ptr<MatrixMultiplicationLayer<Dtype> > layer(
      new MatrixMultiplicationLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 5);
}

template <typename Dtype>
void print_mat(
  const int M, const int N,
  const Dtype* Z, const char* name)
{
  std::cout<<name<<"("<<M<<"x"<<N<<"):"<<std::endl;
  for(int m=0; m<M; ++m) {
    for(int n=0; n<N; ++n) {
      std::cout<<Z[m*N+n]<<" ";
    }
    std::cout<<std::endl;
  }
}

template <typename Dtype>
void check_mat_mul(
  const int M, const int N, const int K,
  const Dtype* X, const Dtype* Y, const Dtype* Z)
{
  print_mat<Dtype>(M,K,X,"X");
  print_mat<Dtype>(K,N,Y,"Y");
  print_mat<Dtype>(M,N,Z,"Z");
  std::cout<<"-----------"<<std::endl;
  for(int m=0; m<M; ++m) {
    for(int n=0; n<N; ++n) {
      const Dtype z=Z[m*N+n];
      Dtype expected_z=0;
      for(int k=0; k<K; ++k) {
        expected_z += X[m*K+k]*Y[k*N+n];
      }
      EXPECT_FLOAT_EQ(z, expected_z);
    }
  }
}

TYPED_TEST(MatrixMultiplicationLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_Y_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    shared_ptr<MatrixMultiplicationLayer<Dtype> > layer(
        new MatrixMultiplicationLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* X_data = this->blob_bottom_X_->cpu_data();
    const Dtype* Y_data = this->blob_bottom_Y_->cpu_data();
    const Dtype* Z_data = this->blob_top_->cpu_data();
    const int B = this->blob_bottom_X_->shape(0);
    const int M = this->blob_top_->shape(1);
    const int N = this->blob_top_->shape(2);
    const int K = this->blob_bottom_X_->shape(2);
    const int X_stride = this->blob_bottom_X_->count(1);
    const int Y_stride = this->blob_bottom_Y_->count(1);
    const int Z_stride = this->blob_top_->count(1);
    for (int b = 0; b < B; ++b) {
      check_mat_mul<Dtype>(M,N,K,
        X_data+b*X_stride,
        Y_data+b*Y_stride,
        Z_data+b*Z_stride);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(MatrixMultiplicationLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_Y_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    MatrixMultiplicationLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
