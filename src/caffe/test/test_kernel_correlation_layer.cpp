/*
 * test_kernel_correlation_layer.cpp
 *
 *  Created on: Oct 17, 2017
 *      Author: cfeng
 */

#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/kernel_correlation_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe
{

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template<typename TypeParam>
class KernelCorrelationLayerTest : public MultiDeviceTest<TypeParam>
{
  typedef typename TypeParam::Dtype Dtype;
 protected:
  KernelCorrelationLayerTest()
      : blob_X(new Blob<Dtype>()),
        blob_G_indptr(new Blob<Dtype>()),
        blob_G_indices(new Blob<Dtype>()),
        blob_Y(new Blob<Dtype>()),
        blob_K(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>())
  {
    // X
    vector<int> shape(2);
    shape[0] = 7;
    shape[1] = 3;
    blob_X->Reshape(shape);

    Dtype* X = blob_X->mutable_cpu_data();
    const Dtype x[7][3] =
    {
    { 0, 0, 0 },
    { 1, 0, 0 },
    { -1, 0, 0 },
    { 0, 1, 0 },
    { 0, -1, 0 },
    { 0, 0, 1 },
    { 0, 0, -1 } };
    for (int i = 0, cnt = 0; i < 7; ++i)
    {
      for (int j = 0; j < 3; ++j, ++cnt)
        X[cnt] = x[i][j];
    }

    // G
    vector<int> shape0(1);
    shape0[0] = 8;
    blob_G_indptr->Reshape(shape0);
    shape0[0] = 43;
    blob_G_indices->Reshape(shape0);
    const Dtype gindptr[8] =
    { 0, 7, 13, 19, 25, 31, 37, 43 };
    const Dtype gindices[43] =
    { 3, 6, 0, 4, 1, 2, 5, 3, 6, 5, 1, 4, 0, 6, 5, 2, 3, 0, 4, 0, 3, 6, 2, 5, 1,
      0, 4, 1, 5, 2, 6, 4, 1, 3, 0, 5, 2, 6, 3, 4, 0, 1, 2 };
    Dtype* G_indptr = blob_G_indptr->mutable_cpu_data();
    Dtype* G_indices = blob_G_indices->mutable_cpu_data();
    for (int i = 0; i < 8; ++i)
      G_indptr[i] = gindptr[i];
    for (int i = 0; i < 43; ++i)
      G_indices[i] = gindices[i];

    // K
    vector<int> shape1(3);
    shape1[0] = 5;
    shape1[1] = 3;
    shape1[2] = 3;
    blob_K->Reshape(shape1);
    const Dtype k[5][3][3] =
    {
    {
    { 0, 0, 0 },
    { 1, 0, 0 },
    { -1, 0, 0 } },
    {
    { 0, 0, 0 },
    { 0, 1, 0 },
    { 0, -1, 0 } },
    {
    { 0, 0, 0 },
    { 0, 0, 1 },
    { 0, 0, -1 } },
    {
    { -1, 0, 0 },
    { 0, -1, 0 },
    { 0, 0, 1 } },
    {
    { 0, 1, 0 },
    { 0, -1, 0 },
    { 0, 0, -1 } } };
    Dtype* K = blob_K->mutable_cpu_data();
    for (int i = 0, cnt = 0; i < 5; ++i)
      for (int j = 0; j < 3; ++j)
        for (int kk = 0; kk < 3; ++kk, ++cnt)
          K[cnt] = k[i][j][kk];

    // Y
    shape[0] = 7;
    shape[1] = 5;
    blob_Y->Reshape(shape);
    Dtype* Y = blob_Y->mutable_cpu_data();

    const Dtype y[7][5] =
//    {
//    { 7.06234884, 7.06234884, 7.06234884, 5.78260851, 5.78260851 },
//    { 5.16176462, 3.86403298, 3.86403298, 4.79420948, 2.93221879 },
//    { 5.16176462, 3.86403298, 3.86403298, 2.36795926, 2.93221879 },
//    { 3.86403298, 5.16176462, 3.86403298, 4.79420948, 4.22995043 },
//    { 3.86403298, 5.16176462, 3.86403298, 2.36795926, 4.22995043 },
//    { 3.86403298, 3.86403298, 5.16176462, 2.36795926, 4.79420948 },
//    { 3.86403298, 3.86403298, 5.16176462, 4.79420948, 2.36795926 } };
        {
        { 1.00890696, 1.00890696, 1.00890696, 0.82608694, 0.82608694 },
        { 0.8602941, 0.64400554, 0.64400554, 0.79903495, 0.48870313 },
        { 0.8602941, 0.64400554, 0.64400554, 0.39465991, 0.48870313 },
        { 0.64400554, 0.8602941, 0.64400554, 0.79903495, 0.7049917 },
        { 0.64400554, 0.8602941, 0.64400554, 0.39465991, 0.7049917 },
        { 0.64400554, 0.64400554, 0.8602941, 0.39465991, 0.79903495 },
        { 0.64400554, 0.64400554, 0.8602941, 0.79903495, 0.39465991 } };
    for (int i = 0, cnt = 0; i < 7; ++i)
    {
      for (int j = 0; j < 5; ++j, ++cnt)
        Y[cnt] = y[i][j];
    }

    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~KernelCorrelationLayerTest()
  {
    delete blob_X;
    delete blob_G_indptr;
    delete blob_G_indices;
    delete blob_Y;
    delete blob_K;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_X;
  Blob<Dtype>* const blob_G_indptr;
  Blob<Dtype>* const blob_G_indices;
  Blob<Dtype>* const blob_Y;
  Blob<Dtype>* const blob_K;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(KernelCorrelationLayerTest, TestDtypesAndDevices);

TYPED_TEST(KernelCorrelationLayerTest, TestSetUp){
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
layer_param.mutable_kernel_correlation_param()->set_num_output(5);
layer_param.mutable_kernel_correlation_param()->set_num_points_per_kernel(3);
layer_param.mutable_kernel_correlation_param()->set_sigma(1);
shared_ptr<KernelCorrelationLayer<Dtype> > layer(
    new KernelCorrelationLayer<Dtype>(layer_param));

this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
this->blob_bottom_vec_.push_back(this->blob_G_indptr);
this->blob_bottom_vec_.push_back(this->blob_G_indices);
layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
EXPECT_EQ(this->blob_X->num_axes(), 2);
EXPECT_EQ(this->blob_G_indptr->num_axes(), 1);
EXPECT_EQ(this->blob_G_indices->num_axes(), 1);
EXPECT_EQ(this->blob_X->shape(0)+1, this->blob_G_indptr->shape(0));
EXPECT_EQ(this->blob_top_->shape(1), 5);

}

template<typename Dtype>
static void print_mat(const int M, const int N, const Dtype* Z,
                      const char* name)
{
  std::cout << name << "(" << M << "x" << N << "):" << std::endl;
  for (int m = 0; m < M; ++m)
  {
    for (int n = 0; n < N; ++n)
    {
      std::cout << Z[m * N + n] << " ";
    }
    std::cout << std::endl << std::flush;
  }
}

template<typename Dtype>
static void check_all_eq(const int N, const Dtype* calc, const Dtype* expect,
                         const Dtype threshold = Dtype(1e-5))
{
  for (int i = 0; i < N; ++i)
  {
    EXPECT_NEAR(calc[i], expect[i], threshold);
  }
}

TYPED_TEST(KernelCorrelationLayerTest, TestForward){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
this->blob_bottom_vec_.push_back(this->blob_G_indptr);
this->blob_bottom_vec_.push_back(this->blob_G_indices);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_kernel_correlation_param()->set_num_output(5);
  layer_param.mutable_kernel_correlation_param()->set_num_points_per_kernel(3);
  layer_param.mutable_kernel_correlation_param()->set_sigma(1);

  shared_ptr<KernelCorrelationLayer<Dtype> > layer(
      new KernelCorrelationLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->blobs()[0]->CopyFrom(*this->blob_K);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* Y_calc = this->blob_top_->cpu_data();
  const Dtype* Y = this->blob_Y->cpu_data();
  check_all_eq(this->blob_Y->count(), Y_calc, Y);
  //print_mat(7,5,Y,"Y");
  //print_mat(7,5,Y_calc,"Y_calc");
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(KernelCorrelationLayerTest, TestGradient){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
this->blob_bottom_vec_.push_back(this->blob_G_indptr);
this->blob_bottom_vec_.push_back(this->blob_G_indices);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_kernel_correlation_param()->set_num_output(5);
  layer_param.mutable_kernel_correlation_param()->set_num_points_per_kernel(3);
  layer_param.mutable_kernel_correlation_param()->set_sigma(1);
  layer_param.mutable_kernel_correlation_param()->mutable_kernel_filler()->set_type("uniform");
  layer_param.mutable_kernel_correlation_param()->mutable_kernel_filler()->set_min(-0.5);
  layer_param.mutable_kernel_correlation_param()->mutable_kernel_filler()->set_max(0.5);

  KernelCorrelationLayer<Dtype> layer(layer_param);
  //layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  //layer.blobs()[0]->CopyFrom(*this->blob_K);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, -2);//note the forth argument is set to -2, meaning we only check gradient w.r.t. to params

  print_mat(5,9,layer.blobs()[0]->cpu_data(),"K");
  print_mat(5,9,layer.blobs()[0]->cpu_diff(),"dK");
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

}
  // namespace caffe
