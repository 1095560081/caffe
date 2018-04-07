/*
 * test_tensor_transpose_layer.cpp
 *
 *  Created on: Nov 11, 2017
 *      Author: cfeng
 */

#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/tensor_transpose_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class TensorTransposeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  TensorTransposeLayerTest()
      : blob_bottom_X_(new Blob<Dtype>()),
        blob_bottom_Y201_(new Blob<Dtype>()),
        blob_bottom_Y210_(new Blob<Dtype>()),
        blob_bottom_Y012_(new Blob<Dtype>()),
        blob_bottom_Y021_(new Blob<Dtype>()),
        blob_bottom_Y102_(new Blob<Dtype>()),
        blob_bottom_Y120_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    // reshape
    vector<int> shape(3);
    shape[0]=2; shape[1]=3; shape[2]=4;
    blob_bottom_X_->Reshape(shape);

    shape[0]=4; shape[1]=2; shape[2]=3;
    blob_bottom_Y201_->Reshape(shape);
    shape[0]=4; shape[1]=3; shape[2]=2;
    blob_bottom_Y210_->Reshape(shape);

    shape[0]=2; shape[1]=3; shape[2]=4;
    blob_bottom_Y012_->Reshape(shape);
    shape[0]=2; shape[1]=4; shape[2]=3;
    blob_bottom_Y021_->Reshape(shape);

    shape[0]=3; shape[1]=2; shape[2]=4;
    blob_bottom_Y102_->Reshape(shape);
    shape[0]=3; shape[1]=4; shape[2]=2;
    blob_bottom_Y120_->Reshape(shape);

    Dtype* x=blob_bottom_X_->mutable_cpu_data();
    Dtype* y201=blob_bottom_Y201_->mutable_cpu_data();
    Dtype* y210=blob_bottom_Y210_->mutable_cpu_data();
    Dtype* y012=blob_bottom_Y012_->mutable_cpu_data();
    Dtype* y021=blob_bottom_Y021_->mutable_cpu_data();
    Dtype* y102=blob_bottom_Y102_->mutable_cpu_data();
    Dtype* y120=blob_bottom_Y120_->mutable_cpu_data();

    Dtype X[2][3][4]={
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9, 10, 11},
        },
        {
            {12, 13, 14, 15},
            {16, 17, 18, 19},
            {20, 21, 22, 23}
        }
    };
    Dtype Y201[4][2][3]={{{ 0,  4,  8},
        {12, 16, 20}},
       {{ 1,  5,  9},
        {13, 17, 21}},
       {{ 2,  6, 10},
        {14, 18, 22}},
       {{ 3,  7, 11},
        {15, 19, 23}}};
    Dtype Y210[4][3][2]={{{ 0, 12},
        { 4, 16},
        { 8, 20}},
       {{ 1, 13},
        { 5, 17},
        { 9, 21}},
       {{ 2, 14},
        { 6, 18},
        {10, 22}},
       {{ 3, 15},
        { 7, 19},
        {11, 23}}};
    Dtype Y021[2][4][3]={{{ 0,  4,  8},
        { 1,  5,  9},
        { 2,  6, 10},
        { 3,  7, 11}},
       {{12, 16, 20},
        {13, 17, 21},
        {14, 18, 22},
        {15, 19, 23}}};
    Dtype Y102[3][2][4]={{{ 0,  1,  2,  3},
        {12, 13, 14, 15}},
       {{ 4,  5,  6,  7},
        {16, 17, 18, 19}},
       {{ 8,  9, 10, 11},
        {20, 21, 22, 23}}};
    Dtype Y120[3][4][2]={{{ 0, 12},
        { 1, 13},
        { 2, 14},
        { 3, 15}},
       {{ 4, 16},
        { 5, 17},
        { 6, 18},
        { 7, 19}},
       {{ 8, 20},
        { 9, 21},
        {10, 22},
        {11, 23}}};

    caffe_copy(2*3*4, &X[0][0][0], x);
    caffe_copy(2*3*4, &Y201[0][0][0], y201);
    caffe_copy(2*3*4, &Y210[0][0][0], y210);
    caffe_copy(2*3*4, &X[0][0][0],    y012);
    caffe_copy(2*3*4, &Y021[0][0][0], y021);
    caffe_copy(2*3*4, &Y102[0][0][0], y102);
    caffe_copy(2*3*4, &Y120[0][0][0], y120);


    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~TensorTransposeLayerTest() {
    delete blob_bottom_X_;
    delete blob_bottom_Y201_;
    delete blob_bottom_Y210_;
    delete blob_bottom_Y012_;
    delete blob_bottom_Y021_;
    delete blob_bottom_Y102_;
    delete blob_bottom_Y120_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_X_;
  Blob<Dtype>* const blob_bottom_Y201_;
  Blob<Dtype>* const blob_bottom_Y210_;
  Blob<Dtype>* const blob_bottom_Y012_;
  Blob<Dtype>* const blob_bottom_Y021_;
  Blob<Dtype>* const blob_bottom_Y102_;
  Blob<Dtype>* const blob_bottom_Y120_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TensorTransposeLayerTest, TestDtypesAndDevices);

TYPED_TEST(TensorTransposeLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  {
    LayerParameter layer_param;
    layer_param.mutable_tensor_transpose_param()->add_order(2);
    layer_param.mutable_tensor_transpose_param()->add_order(0);
    layer_param.mutable_tensor_transpose_param()->add_order(1);
    shared_ptr<TensorTransposeLayer<Dtype> > layer(
        new TensorTransposeLayer<Dtype>(layer_param));

    //201
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num_axes(), 3);
    EXPECT_EQ(this->blob_top_->shape(0), 4);
    EXPECT_EQ(this->blob_top_->shape(1), 2);
    EXPECT_EQ(this->blob_top_->shape(2), 3);
  }
  {
    LayerParameter layer_param;
    layer_param.mutable_tensor_transpose_param()->add_order(2);
    layer_param.mutable_tensor_transpose_param()->add_order(1);
    layer_param.mutable_tensor_transpose_param()->add_order(0);
    shared_ptr<TensorTransposeLayer<Dtype> > layer(
        new TensorTransposeLayer<Dtype>(layer_param));

    //210
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num_axes(), 3);
    EXPECT_EQ(this->blob_top_->shape(0), 4);
    EXPECT_EQ(this->blob_top_->shape(1), 3);
    EXPECT_EQ(this->blob_top_->shape(2), 2);
  }
  {
    LayerParameter layer_param;
    layer_param.mutable_tensor_transpose_param()->add_order(0);
    layer_param.mutable_tensor_transpose_param()->add_order(1);
    layer_param.mutable_tensor_transpose_param()->add_order(2);
    shared_ptr<TensorTransposeLayer<Dtype> > layer(
        new TensorTransposeLayer<Dtype>(layer_param));

    //012
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num_axes(), 3);
    EXPECT_EQ(this->blob_top_->shape(0), 2);
    EXPECT_EQ(this->blob_top_->shape(1), 3);
    EXPECT_EQ(this->blob_top_->shape(2), 4);
  }
  {
    LayerParameter layer_param;
    layer_param.mutable_tensor_transpose_param()->add_order(0);
    layer_param.mutable_tensor_transpose_param()->add_order(2);
    layer_param.mutable_tensor_transpose_param()->add_order(1);
    shared_ptr<TensorTransposeLayer<Dtype> > layer(
        new TensorTransposeLayer<Dtype>(layer_param));

    //021
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num_axes(), 3);
    EXPECT_EQ(this->blob_top_->shape(0), 2);
    EXPECT_EQ(this->blob_top_->shape(1), 4);
    EXPECT_EQ(this->blob_top_->shape(2), 3);
  }
  {
    LayerParameter layer_param;
    layer_param.mutable_tensor_transpose_param()->add_order(1);
    layer_param.mutable_tensor_transpose_param()->add_order(0);
    layer_param.mutable_tensor_transpose_param()->add_order(2);
    shared_ptr<TensorTransposeLayer<Dtype> > layer(
        new TensorTransposeLayer<Dtype>(layer_param));

    //102
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num_axes(), 3);
    EXPECT_EQ(this->blob_top_->shape(0), 3);
    EXPECT_EQ(this->blob_top_->shape(1), 2);
    EXPECT_EQ(this->blob_top_->shape(2), 4);
  }
  {
    LayerParameter layer_param;
    layer_param.mutable_tensor_transpose_param()->add_order(1);
    layer_param.mutable_tensor_transpose_param()->add_order(2);
    layer_param.mutable_tensor_transpose_param()->add_order(0);
    shared_ptr<TensorTransposeLayer<Dtype> > layer(
        new TensorTransposeLayer<Dtype>(layer_param));

    //120
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num_axes(), 3);
    EXPECT_EQ(this->blob_top_->shape(0), 3);
    EXPECT_EQ(this->blob_top_->shape(1), 4);
    EXPECT_EQ(this->blob_top_->shape(2), 2);
  }
}

template <typename Dtype>
static void print_mat(
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

//TYPED_TEST(TensorTransposeLayerTest, TestForward201) {
//  typedef typename TypeParam::Dtype Dtype;
//  this->blob_bottom_vec_.clear();
//  this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
//  bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//    layer_param.mutable_tensor_transpose_param()->add_order(2);
//    layer_param.mutable_tensor_transpose_param()->add_order(0);
//    layer_param.mutable_tensor_transpose_param()->add_order(1);
//    shared_ptr<TensorTransposeLayer<Dtype> > layer(
//        new TensorTransposeLayer<Dtype>(layer_param));
//    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//    const Dtype* Y = this->blob_bottom_Y201_->cpu_data();
//    const Dtype* Y_calc = this->blob_top_->cpu_data();
//    const int N = this->blob_top_->count();
//    for (int i = 0; i < N; ++i) {
//      EXPECT_FLOAT_EQ(Y[i], Y_calc[i]);
//    }
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//}
//
//TYPED_TEST(TensorTransposeLayerTest, TestGradient201) {
//  typedef typename TypeParam::Dtype Dtype;
//  this->blob_bottom_vec_.clear();
//  this->blob_bottom_vec_.push_back(this->blob_bottom_X_);
//  bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//    layer_param.mutable_tensor_transpose_param()->add_order(2);
//    layer_param.mutable_tensor_transpose_param()->add_order(0);
//    layer_param.mutable_tensor_transpose_param()->add_order(1);
//    TensorTransposeLayer<Dtype> layer(layer_param);
//    GradientChecker<Dtype> checker(1e-2, 1e-3);
//    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//        this->blob_top_vec_);
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//}

#ifndef CPU_ONLY

#define MY_TEST_GROUP(o1,o2,o3) \
    TYPED_TEST(TensorTransposeLayerTest, TestForward##o1##o2##o3) {\
      typedef typename TypeParam::Dtype Dtype;\
      this->blob_bottom_vec_.clear();\
      this->blob_bottom_vec_.push_back(this->blob_bottom_X_);\
      bool IS_VALID_CUDA = false;\
      IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;\
      if (Caffe::mode() == Caffe::CPU ||\
          sizeof(Dtype) == 4 || IS_VALID_CUDA) {\
        LayerParameter layer_param;\
        layer_param.mutable_tensor_transpose_param()->add_order(o1);\
        layer_param.mutable_tensor_transpose_param()->add_order(o2);\
        layer_param.mutable_tensor_transpose_param()->add_order(o3);\
        shared_ptr<TensorTransposeLayer<Dtype> > layer(\
            new TensorTransposeLayer<Dtype>(layer_param));\
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);\
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);\
        const Dtype* Y = this->blob_bottom_Y##o1##o2##o3##_->cpu_data();\
        const Dtype* Y_calc = this->blob_top_->cpu_data();\
        const int N = this->blob_top_->count();\
        for (int i = 0; i < N; ++i) {\
          EXPECT_FLOAT_EQ(Y[i], Y_calc[i]);\
        }\
      } else {\
        LOG(ERROR) << "Skipping test due to old architecture.";\
      }\
    }\
    TYPED_TEST(TensorTransposeLayerTest, TestGradient##o1##o2##o3) {\
      typedef typename TypeParam::Dtype Dtype;\
      this->blob_bottom_vec_.clear();\
      this->blob_bottom_vec_.push_back(this->blob_bottom_X_);\
      bool IS_VALID_CUDA = false;\
      IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;\
      if (Caffe::mode() == Caffe::CPU ||\
          sizeof(Dtype) == 4 || IS_VALID_CUDA) {\
        LayerParameter layer_param;\
        layer_param.mutable_tensor_transpose_param()->add_order(o1);\
        layer_param.mutable_tensor_transpose_param()->add_order(o2);\
        layer_param.mutable_tensor_transpose_param()->add_order(o3);\
        TensorTransposeLayer<Dtype> layer(layer_param);\
        GradientChecker<Dtype> checker(1e-2, 1e-3);\
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,\
            this->blob_top_vec_);\
      } else {\
        LOG(ERROR) << "Skipping test due to old architecture.";\
      }\
    }
#else

#define MY_TEST_GROUP(o1,o2,o3) \
    TYPED_TEST(TensorTransposeLayerTest, TestForward##o1##o2##o3) {\
      typedef typename TypeParam::Dtype Dtype;\
      this->blob_bottom_vec_.clear();\
      this->blob_bottom_vec_.push_back(this->blob_bottom_X_);\
      bool IS_VALID_CUDA = false;\
      if (Caffe::mode() == Caffe::CPU ||\
          sizeof(Dtype) == 4 || IS_VALID_CUDA) {\
        LayerParameter layer_param;\
        layer_param.mutable_tensor_transpose_param()->add_order(o1);\
        layer_param.mutable_tensor_transpose_param()->add_order(o2);\
        layer_param.mutable_tensor_transpose_param()->add_order(o3);\
        shared_ptr<TensorTransposeLayer<Dtype> > layer(\
            new TensorTransposeLayer<Dtype>(layer_param));\
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);\
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);\
        const Dtype* Y = this->blob_bottom_Y##o1##o2##o3_->cpu_data();\
        const Dtype* Y_calc = this->blob_top_->cpu_data();\
        const int N = this->blob_top_->count();\
        for (int i = 0; i < N; ++i) {\
          EXPECT_FLOAT_EQ(Y[i], Y_calc[i]);\
        }\
      } else {\
        LOG(ERROR) << "Skipping test due to old architecture.";\
      }\
    }\
    TYPED_TEST(TensorTransposeLayerTest, TestGradient##o1##o2##o3) {\
      typedef typename TypeParam::Dtype Dtype;\
      this->blob_bottom_vec_.clear();\
      this->blob_bottom_vec_.push_back(this->blob_bottom_X_);\
      bool IS_VALID_CUDA = false;\
      if (Caffe::mode() == Caffe::CPU ||\
          sizeof(Dtype) == 4 || IS_VALID_CUDA) {\
        LayerParameter layer_param;\
        layer_param.mutable_tensor_transpose_param()->add_order(o1);\
        layer_param.mutable_tensor_transpose_param()->add_order(o2);\
        layer_param.mutable_tensor_transpose_param()->add_order(o3);\
        TensorTransposeLayer<Dtype> layer(layer_param);\
        GradientChecker<Dtype> checker(1e-2, 1e-3);\
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,\
            this->blob_top_vec_);\
      } else {\
        LOG(ERROR) << "Skipping test due to old architecture.";\
      }\
    }
#endif


MY_TEST_GROUP(2,0,1)
MY_TEST_GROUP(2,1,0)
MY_TEST_GROUP(0,1,2)
MY_TEST_GROUP(0,2,1)
MY_TEST_GROUP(1,0,2)
MY_TEST_GROUP(1,2,0)

}  // namespace caffe
