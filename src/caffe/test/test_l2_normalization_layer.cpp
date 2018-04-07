/*
 * test_l2_normalization_layer.cpp
 *
 *  Created on: Oct 17, 2017
 *      Author: cfeng
 */

#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/l2_normalization_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe
{

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template<typename TypeParam>
class L2NormalizationLayerTest : public MultiDeviceTest<TypeParam>
{
  typedef typename TypeParam::Dtype Dtype;
 protected:
  L2NormalizationLayerTest()
      : blob_X(new Blob<Dtype>()),
        blob_Y(new Blob<Dtype>()),
        blob_y0(new Blob<Dtype>()),
        blob_y1(new Blob<Dtype>()),
        blob_y2(new Blob<Dtype>()),
        blob_x(new Blob<Dtype>()),
        blob_z(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>())
  {
    // X
    vector<int> shape(3);
    shape[0] = 2;
    shape[1] = 3;
    shape[2] = 4;
    blob_X->Reshape(shape);
    // Y, global l2 normalize
    blob_Y->Reshape(shape);
    // yi, l2 normalize along axis i
    blob_y0->Reshape(shape);
    blob_y1->Reshape(shape);
    blob_y2->Reshape(shape);

    Dtype* X = blob_X->mutable_cpu_data();
    Dtype* Y = blob_Y->mutable_cpu_data();
    Dtype* Y0 = blob_y0->mutable_cpu_data();
    Dtype* Y1 = blob_y1->mutable_cpu_data();
    Dtype* Y2 = blob_y2->mutable_cpu_data();
    const Dtype x[2][3][4] =
    {
        {
            {1,2,3,4},
            {5,0,7,8},
            {9,10,11,12}
        },
        {
            {0,0,0,0},
            {0,0,2,1},
            {0,-1,2,3}
        }
    };
    const Dtype y[2][3][4] =
    {
        {
            {0.03974643,  0.07949287,  0.1192393 ,  0.15898573},
            {0.19873217,  0.        ,  0.27822503,  0.31797147},
            {0.3577179 ,  0.39746433,  0.43721077,  0.4769572}
        },
        {
            {0,0,0,0},
            {0.        ,  0.        ,  0.07949287,  0.03974643},
            {0.        ,  -0.03974643,  0.07949287,  0.1192393}
        }
    };
    const Dtype y0[2][3][4] =
    {
        {
            {1.        ,  1.        ,  1.        ,  1.        },
            {1.        ,           0,  0.96152395,  0.99227786},
            {1.        ,  0.9950372 ,  0.98386991,  0.97014248}
        },
        {
            {0,0,0,0},
            {0.        ,           0,  0.27472112,  0.12403473},
            {0.        , -0.09950373,  0.17888544,  0.24253562}
        }
    };
    const Dtype y1[2][3][4] =
    {
        {
            {0.09667365,  0.19611613,  0.22423053,  0.26726124},
            {0.48336828,  0.        ,  0.52320457,  0.53452247},
            {0.87006289,  0.98058069,  0.8221786 ,  0.80178368}
        },
        {
            {0,0,0,0},
            {0,0.        ,  0.70710677,  0.31622776},
            {0,-1.        ,  0.70710677,  0.94868326}
        }
    };
    const Dtype y2[2][3][4] =
    {
        {
            {0.18257418,  0.36514837,  0.54772252,  0.73029673},
            {0.42562827,  0.        ,  0.59587955,  0.68100524},
            {0.42616236,  0.47351375,  0.52086508,  0.5682165}
        },
        {
            {0,0,0,0},
            {0.        ,  0.        ,  0.89442718,  0.44721359},
            {0.        , -0.26726124,  0.53452247,  0.80178368}
        }
    };
    for (int i = 0, cnt = 0; i < 2; ++i)
    {
      for (int j = 0; j < 3; ++j)
        for(int k=0; k<4; ++k, ++cnt) {
          X[cnt] = x[i][j][k];
          Y[cnt] = y[i][j][k];
          Y0[cnt] = y0[i][j][k];
          Y1[cnt] = y1[i][j][k];
          Y2[cnt] = y2[i][j][k];
        }
    }

    //x,z
    vector<int> shape2(1);
    shape2[0] = 2;
    blob_x->Reshape(shape2);
    blob_z->Reshape(shape2);
    blob_x->mutable_cpu_data()[0]=0;
    blob_x->mutable_cpu_data()[1]=0;
    blob_z->mutable_cpu_data()[0]=0;
    blob_z->mutable_cpu_data()[1]=0;

    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~L2NormalizationLayerTest()
  {
    delete blob_X;
    delete blob_Y;
    delete blob_y0;
    delete blob_y1;
    delete blob_y2;
    delete blob_x;
    delete blob_z;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_X;
  Blob<Dtype>* const blob_Y;
  Blob<Dtype>* const blob_y0;
  Blob<Dtype>* const blob_y1;
  Blob<Dtype>* const blob_y2;
  Blob<Dtype>* const blob_x;
  Blob<Dtype>* const blob_z;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(L2NormalizationLayerTest, TestDtypesAndDevices);

TYPED_TEST(L2NormalizationLayerTest, TestSetUp){
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
layer_param.mutable_l2n_param()->set_is_global(true);
shared_ptr<L2NormalizationLayer<Dtype> > layer(
    new L2NormalizationLayer<Dtype>(layer_param));

this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
EXPECT_EQ(this->blob_X->num_axes(), 3);
EXPECT_EQ(this->blob_top_->num_axes(), 3);
EXPECT_EQ(this->blob_top_->shape(0), 2);
EXPECT_EQ(this->blob_top_->shape(1), 3);
EXPECT_EQ(this->blob_top_->shape(2), 4);

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

TYPED_TEST(L2NormalizationLayerTest, TestForwardGlobal){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(true);

  shared_ptr<L2NormalizationLayer<Dtype> > layer(
      new L2NormalizationLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* Y_calc = this->blob_top_->cpu_data();
  const Dtype* Y = this->blob_Y->cpu_data();
  check_all_eq(this->blob_Y->count(), Y_calc, Y);
  print_mat(2,12,Y,"Y");
  print_mat(2,12,Y_calc,"Y_calc");
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(L2NormalizationLayerTest, TestForwardGlobal2){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_x);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(true);

  shared_ptr<L2NormalizationLayer<Dtype> > layer(
      new L2NormalizationLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* z_calc = this->blob_top_->cpu_data();
  const Dtype* z = this->blob_z->cpu_data();
  check_all_eq(this->blob_z->count(), z_calc, z);
  print_mat(1,2,z,"z");
  print_mat(1,2,z_calc,"z_calc");
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(L2NormalizationLayerTest, TestForwardAxis0){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(false);
  layer_param.mutable_l2n_param()->set_axis(0);

  shared_ptr<L2NormalizationLayer<Dtype> > layer(
      new L2NormalizationLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* Y_calc = this->blob_top_->cpu_data();
  const Dtype* Y = this->blob_y0->cpu_data();
  check_all_eq(this->blob_Y->count(), Y_calc, Y);
  print_mat(2,12,Y,"Y0");
  print_mat(2,12,Y_calc,"Y_calc");
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(L2NormalizationLayerTest, TestForwardAxis1){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(false);
  layer_param.mutable_l2n_param()->set_axis(1);

  shared_ptr<L2NormalizationLayer<Dtype> > layer(
      new L2NormalizationLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* Y_calc = this->blob_top_->cpu_data();
  const Dtype* Y = this->blob_y1->cpu_data();
  check_all_eq(this->blob_Y->count(), Y_calc, Y);
  print_mat(2,12,Y,"Y1");
  print_mat(2,12,Y_calc,"Y_calc");
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(L2NormalizationLayerTest, TestForwardAxis2){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(false);
  layer_param.mutable_l2n_param()->set_axis(2);

  shared_ptr<L2NormalizationLayer<Dtype> > layer(
      new L2NormalizationLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* Y_calc = this->blob_top_->cpu_data();
  const Dtype* Y = this->blob_y2->cpu_data();
  check_all_eq(this->blob_Y->count(), Y_calc, Y);
  print_mat(2,12,Y,"Y2");
  print_mat(2,12,Y_calc,"Y_calc");
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(L2NormalizationLayerTest, TestGradientGlobal){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(true);

  L2NormalizationLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(L2NormalizationLayerTest, TestGradientGlobal2){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_x);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(true);

  L2NormalizationLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0, 0.001); //layer is not smooth when bottom[0] magnitude is zero, thus ignore
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(L2NormalizationLayerTest, TestGradientAxis0){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(false);
  layer_param.mutable_l2n_param()->set_axis(0);

  L2NormalizationLayer<Dtype> layer(layer_param);
  //layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  //layer.blobs()[0]->CopyFrom(*this->blob_K);

  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0, 0.001);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(L2NormalizationLayerTest, TestGradientAxis1){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(false);
  layer_param.mutable_l2n_param()->set_axis(1);

  L2NormalizationLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0, 0.001);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

TYPED_TEST(L2NormalizationLayerTest, TestGradientAxis2){
typedef typename TypeParam::Dtype Dtype;
this->blob_bottom_vec_.clear();
this->blob_bottom_vec_.push_back(this->blob_X);
bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
if (Caffe::mode() == Caffe::CPU ||
    sizeof(Dtype) == 4 || IS_VALID_CUDA)
{
  LayerParameter layer_param;
  layer_param.mutable_l2n_param()->set_is_global(false);
  layer_param.mutable_l2n_param()->set_axis(2);

  L2NormalizationLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0, 0.001);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
else
{
  LOG(ERROR) << "Skipping test due to old architecture.";
}
}

}
  // namespace caffe
