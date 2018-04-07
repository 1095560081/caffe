#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/graph_pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class GraphPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  GraphPoolingLayerTest()
      :
    blob_X1(new Blob<Dtype>()),
    blob_X2(new Blob<Dtype>()),
    blob_X3(new Blob<Dtype>()),
    blob_G1_indptr(new Blob<Dtype>()),
    blob_G1_indices(new Blob<Dtype>()),
    blob_G1_data(new Blob<Dtype>()),
    blob_G2_indptr(new Blob<Dtype>()),
    blob_G2_indices(new Blob<Dtype>()),
    blob_G2_data(new Blob<Dtype>()),
    blob_G3_indptr(new Blob<Dtype>()),
    blob_G3_indices(new Blob<Dtype>()),
    blob_G3_data(new Blob<Dtype>()),
    //for MxN type of graph matrix where M!=N
    blob_G4_indptr(new Blob<Dtype>()),
    blob_G4_indices(new Blob<Dtype>()),
    blob_G4_data(new Blob<Dtype>()),

    blob_Xall(new Blob<Dtype>()),
    blob_Gall_indptr(new Blob<Dtype>()),
    blob_Gall_indices(new Blob<Dtype>()),
    blob_Gall_data(new Blob<Dtype>()),
    blob_n_offset(new Blob<Dtype>()),

    //Y=AX
    blob_Y1_ave(new Blob<Dtype>()),
    blob_Y2_ave(new Blob<Dtype>()),
    blob_Y3_ave(new Blob<Dtype>()),
    blob_Y4_ave(new Blob<Dtype>()), //for G4
    //Y=op(X)
    blob_Y1_max(new Blob<Dtype>()),
    blob_Y2_max(new Blob<Dtype>()),
    blob_Y3_max(new Blob<Dtype>()),
    //Y=max(X)
    blob_Y1_gmax(new Blob<Dtype>()),
    blob_Y2_gmax(new Blob<Dtype>()),
    blob_Y3_gmax(new Blob<Dtype>()),

    blob_Yall_ave(new Blob<Dtype>()), //Y=AX
    blob_Yall_max(new Blob<Dtype>()), //Y=op(X)
    blob_Yall_bmax(new Blob<Dtype>()), //Y=max(X), batched
    blob_Yall_gmax(new Blob<Dtype>()), //Y=max(X)
    blob_Yall_bave(new Blob<Dtype>()), //Y=AX, batched
    blob_Yall_gave(new Blob<Dtype>()), //Y=AX, global

    blob_top_(new Blob<Dtype>())
  {
    // X1/X2/X3
    vector<int> shape(2);
    shape[0]=4; shape[1]=2;
    blob_X1->Reshape(shape);
    blob_X2->Reshape(shape);
    blob_X3->Reshape(shape);

    Dtype* X1 = blob_X1->mutable_cpu_data();
    Dtype* X2 = blob_X2->mutable_cpu_data();
    Dtype* X3 = blob_X3->mutable_cpu_data();
    const Dtype x1[4][2] = {
        {1,2},
        {2,3},
        {0,1},
        {4,0}
    };
    const Dtype x2[4][2] = {
        {0,1},
        {1,3},
        {2,0},
        {3,2}
    };
    const Dtype x3[4][2] = {
        {2,1},
        {3,4},
        {2,3},
        {0,1}
    };
    for(int i=0, cnt=0; i<4; ++i) {
      for(int j=0; j<2; ++j, ++cnt) {
        X1[cnt] = x1[i][j];
        X2[cnt] = x2[i][j];
        X3[cnt] = x3[i][j];
      }
    }

    //G1
    vector<int> shape0(1);
    shape0[0]=5;
    blob_G1_indptr->Reshape(shape0);
    shape0[0]=6;
    blob_G1_indices->Reshape(shape0);
    blob_G1_data->Reshape(shape0);
    const Dtype g1indptr[5]={0,1,2,3,6};
    const Dtype g1indices[6]={3,3,3,0,1,2};
    const Dtype g1data[6]={1,1,1,1,1,1};
    Dtype* G1_indptr = blob_G1_indptr->mutable_cpu_data();
    Dtype* G1_indices = blob_G1_indices->mutable_cpu_data();
    Dtype* G1_data = blob_G1_data->mutable_cpu_data();
    for(int i=0; i<5; ++i) G1_indptr[i]=g1indptr[i];
    for(int i=0; i<6; ++i) {
      G1_indices[i]=g1indices[i];
      G1_data[i]=g1data[i];
    }

    //G2
    shape0[0]=5;
    blob_G2_indptr->Reshape(shape0);
    shape0[0]=10;
    blob_G2_indices->Reshape(shape0);
    blob_G2_data->Reshape(shape0);
    const Dtype g2indptr[5]={0,2,5,8,10};
    const Dtype g2indices[10]={1,2,0,2,3,0,1,3,1,2};
    const Dtype g2data[10]={1,1,1,1,1,1,1,1,1,1};
    Dtype* G2_indptr = blob_G2_indptr->mutable_cpu_data();
    Dtype* G2_indices = blob_G2_indices->mutable_cpu_data();
    Dtype* G2_data = blob_G2_data->mutable_cpu_data();
    for(int i=0; i<5; ++i) G2_indptr[i]=g2indptr[i];
    for(int i=0; i<10; ++i) {
      G2_indices[i]=g2indices[i];
      G2_data[i]=g2data[i];
    }

    //G3
    shape0[0]=5;
    blob_G3_indptr->Reshape(shape0);
    shape0[0]=6;
    blob_G3_indices->Reshape(shape0);
    blob_G3_data->Reshape(shape0);
    const Dtype g3indptr[5]={0,1,3,5,6};
    const Dtype g3indices[6]={1,0,2,1,3,2};
    const Dtype g3data[6]={1,1,1,1,1,1};
    Dtype* G3_indptr = blob_G3_indptr->mutable_cpu_data();
    Dtype* G3_indices = blob_G3_indices->mutable_cpu_data();
    Dtype* G3_data = blob_G3_data->mutable_cpu_data();
    for(int i=0; i<5; ++i) G3_indptr[i]=g3indptr[i];
    for(int i=0; i<6; ++i) {
      G3_indices[i]=g3indices[i];
      G3_data[i]=g3data[i];
    }

    //G4
    shape0[0]=7;
    blob_G4_indptr->Reshape(shape0);
    shape0[0]=12;
    blob_G4_indices->Reshape(shape0);
    blob_G4_data->Reshape(shape0);
    const Dtype g4indptr[7]={0,2,4,6,8,10,12};
    const Dtype g4indices[12]={0,1,0,2,0,3,1,2,2,3,3,1};
    const Dtype g4data[12]={1,-1,1,-1,1,-1,1,-1,1,-1,1,-1};
    Dtype* G4_indptr = blob_G4_indptr->mutable_cpu_data();
    Dtype* G4_indices = blob_G4_indices->mutable_cpu_data();
    Dtype* G4_data = blob_G4_data->mutable_cpu_data();
    for(int i=0; i<7; ++i) G4_indptr[i]=g4indptr[i];
    for(int i=0; i<12; ++i) {
      G4_indices[i]=g4indices[i];
      G4_data[i]=g4data[i];
    }

    //Xall
    shape[0]=12; shape[1]=2;
    blob_Xall->Reshape(shape);
    Dtype* Xall=blob_Xall->mutable_cpu_data();
    for(int i=0, cnt=0; i<4; ++i) {
      for(int j=0; j<2; ++j, ++cnt) {
        Xall[cnt] = x1[i][j];
        Xall[cnt+8] = x2[i][j];
        Xall[cnt+16] = x3[i][j];
      }
    }

    //Gall
    shape0[0]=13;
    blob_Gall_indptr->Reshape(shape0);
    shape0[0]=22;
    blob_Gall_indices->Reshape(shape0);
    blob_Gall_data->Reshape(shape0);
    Dtype* Gall_indptr = blob_Gall_indptr->mutable_cpu_data();
    int cnt=0;
    for(int i=0; i<5; ++i, ++cnt) Gall_indptr[cnt] = G1_indptr[i];
    for(int i=1; i<5; ++i, ++cnt) Gall_indptr[cnt] = G2_indptr[i]+6;
    for(int i=1; i<5; ++i, ++cnt) Gall_indptr[cnt] = G3_indptr[i]+6+10;
    Dtype* Gall_indices = blob_Gall_indices->mutable_cpu_data();
    cnt=0;
    for(int i=0; i<6; ++i, ++cnt) Gall_indices[cnt] = G1_indices[i];
    for(int i=0; i<10; ++i, ++cnt) Gall_indices[cnt] = G2_indices[i]+4;
    for(int i=0; i<6; ++i, ++cnt) Gall_indices[cnt] = G3_indices[i]+8;
    Dtype* Gall_data = blob_Gall_data->mutable_cpu_data();
    cnt=0;
    for(int i=0; i<6; ++i, ++cnt) Gall_data[cnt] = G1_data[i];
    for(int i=0; i<10; ++i, ++cnt) Gall_data[cnt] = G2_data[i];
    for(int i=0; i<6; ++i, ++cnt) Gall_data[cnt] = G3_data[i];

    //n_offset
    shape0[0]=3;
    blob_n_offset->Reshape(shape0);
    Dtype* n_offset = blob_n_offset->mutable_cpu_data();
    n_offset[0]=4;
    n_offset[1]=8;
    n_offset[2]=12;

    //Y
    shape[0]=4; shape[1]=2;
    blob_Y1_ave->Reshape(shape);
    blob_Y1_max->Reshape(shape);
    blob_Y2_ave->Reshape(shape);
    blob_Y2_max->Reshape(shape);
    blob_Y3_ave->Reshape(shape);
    blob_Y3_max->Reshape(shape);
    shape[0]=6; shape[1]=2;
    blob_Y4_ave->Reshape(shape);

    Dtype* Y1_ave = blob_Y1_ave->mutable_cpu_data();
    Dtype* Y2_ave = blob_Y2_ave->mutable_cpu_data();
    Dtype* Y3_ave = blob_Y3_ave->mutable_cpu_data();
    Dtype* Y4_ave = blob_Y4_ave->mutable_cpu_data();
    Dtype* Y1_max = blob_Y1_max->mutable_cpu_data();
    Dtype* Y2_max = blob_Y2_max->mutable_cpu_data();
    Dtype* Y3_max = blob_Y3_max->mutable_cpu_data();
    const Dtype y1ave[4][2] = {
        {4,0},
        {4,0},
        {4,0},
        {3,6}
    };
    const Dtype y2ave[4][2] = {
        {3,3},
        {5,3},
        {4,6},
        {3,3}
    };
    const Dtype y3ave[4][2] = {
        {3,4},
        {4,4},
        {3,5},
        {2,3}
    };
    const Dtype y4ave[6][2] = {
        {-1,-1},
        {1,1},
        {-3,2},
        {2,2},
        {-4,1},
        {2,-3}
    };

    const Dtype y1max[4][2] = {
        {4,2},
        {4,3},
        {4,1},
        {4,3}
    };
    const Dtype y2max[4][2] = {
        {2,3},
        {3,3},
        {3,3},
        {3,3}
    };
    const Dtype y3max[4][2] = {
        {3,4},
        {3,4},
        {3,4},
        {2,3}
    };

    for(int i=0, cnt=0; i<4; ++i) {
      for(int j=0; j<2; ++j, ++cnt) {
        Y1_ave[cnt] = y1ave[i][j];
        Y2_ave[cnt] = y2ave[i][j];
        Y3_ave[cnt] = y3ave[i][j];

        Y1_max[cnt] = y1max[i][j];
        Y2_max[cnt] = y2max[i][j];
        Y3_max[cnt] = y3max[i][j];
      }
    }
    for(int i=0, cnt=0; i<6; ++i) {
      for(int j=0; j<2; ++j, ++cnt) {
        Y4_ave[cnt] = y4ave[i][j];
      }
    }

    shape[0]=1; shape[1]=2;
    blob_Y1_gmax->Reshape(shape);
    blob_Y2_gmax->Reshape(shape);
    blob_Y3_gmax->Reshape(shape);
    Dtype* Y1_gmax = blob_Y1_gmax->mutable_cpu_data();
    Dtype* Y2_gmax = blob_Y2_gmax->mutable_cpu_data();
    Dtype* Y3_gmax = blob_Y3_gmax->mutable_cpu_data();
    const Dtype y1gmax[2] = {4,3};
    const Dtype y2gmax[2] = {3,3};
    const Dtype y3gmax[2] = {3,4};

    for(int i=0; i<2; ++i) {
      Y1_gmax[i] = y1gmax[i];
      Y2_gmax[i] = y2gmax[i];
      Y3_gmax[i] = y3gmax[i];
    }

    //Yall
    shape[0]=12; shape[1]=2;
    blob_Yall_ave->Reshape(shape);
    blob_Yall_max->Reshape(shape);
    Dtype* Yall_ave = blob_Yall_ave->mutable_cpu_data();
    Dtype* Yall_max = blob_Yall_max->mutable_cpu_data();
    const Dtype yallave[12][2]={
        {4,0},
        {4,0},
        {4,0},
        {3,6},
        {3,3},
        {5,3},
        {4,6},
        {3,3},
        {3,4},
        {4,4},
        {3,5},
        {2,3}
    };
    const Dtype yallmax[12][2]={
        {4,2},
        {4,3},
        {4,1},
        {4,3},
        {2,3},
        {3,3},
        {3,3},
        {3,3},
        {3,4},
        {3,4},
        {3,4},
        {2,3}
    };
    for(int i=0, cnt=0; i<12; ++i) {
      for(int j=0; j<2; ++j, ++cnt) {
        Yall_ave[cnt] = yallave[i][j];
        Yall_max[cnt] = yallmax[i][j];
      }
    }

    shape[0]=3; shape[1]=2;
    blob_Yall_bmax->Reshape(shape);
    blob_Yall_bave->Reshape(shape);
    Dtype* Yall_bmax = blob_Yall_bmax->mutable_cpu_data();
    Dtype* Yall_bave = blob_Yall_bave->mutable_cpu_data();
    const Dtype yallbmax[3][2]={
        {4,3},
        {3,3},
        {3,4}
    };
    const Dtype yallbave[3][2]={
        {1.75, 1.5},
        {1.5, 1.5},
        {1.75, 2.25}
    };
    for(int i=0, cnt=0; i<3; ++i) {
      for(int j=0; j<2; ++j, ++cnt) {
        Yall_bmax[cnt] = yallbmax[i][j];
        Yall_bave[cnt] = yallbave[i][j];
      }
    }

    shape[0]=1; shape[1]=2;
    blob_Yall_gmax->Reshape(shape);
    blob_Yall_gave->Reshape(shape);
    Dtype* Yall_gmax = blob_Yall_gmax->mutable_cpu_data();
    Dtype* Yall_gave = blob_Yall_gave->mutable_cpu_data();
    const Dtype yallgmax[1][2]={
        {4,4}
    };
    const Dtype yallgave[1][2]={
        {1.666666666667, 1.75}
    };
    for(int i=0, cnt=0; i<1; ++i) {
      for(int j=0; j<2; ++j, ++cnt) {
        Yall_gmax[cnt] = yallgmax[i][j];
        Yall_gave[cnt] = yallgave[i][j];
      }
    }

    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~GraphPoolingLayerTest() {
    delete blob_X1;
    delete blob_X2;
    delete blob_X3;
    delete blob_G1_indptr;
    delete blob_G1_indices;
    delete blob_G1_data;
    delete blob_G2_indptr;
    delete blob_G2_indices;
    delete blob_G2_data;
    delete blob_G3_indptr;
    delete blob_G3_indices;
    delete blob_G3_data;
    delete blob_G4_indptr;
    delete blob_G4_indices;
    delete blob_G4_data;

    delete blob_Xall;
    delete blob_Gall_indptr;
    delete blob_Gall_indices;
    delete blob_Gall_data;
    delete blob_n_offset;

    delete blob_Y1_ave;
    delete blob_Y2_ave;
    delete blob_Y3_ave;
    delete blob_Y4_ave;

    delete blob_Y1_max;
    delete blob_Y2_max;
    delete blob_Y3_max;

    delete blob_Y1_gmax;
    delete blob_Y2_gmax;
    delete blob_Y3_gmax;

    delete blob_Yall_ave;
    delete blob_Yall_max;
    delete blob_Yall_bmax;
    delete blob_Yall_gmax;
    delete blob_Yall_bave;
    delete blob_Yall_gave;

    delete blob_top_;
  }
  Blob<Dtype>* const blob_X1;
  Blob<Dtype>* const blob_X2;
  Blob<Dtype>* const blob_X3;
  Blob<Dtype>* const blob_G1_indptr;
  Blob<Dtype>* const blob_G1_indices;
  Blob<Dtype>* const blob_G1_data;
  Blob<Dtype>* const blob_G2_indptr;
  Blob<Dtype>* const blob_G2_indices;
  Blob<Dtype>* const blob_G2_data;
  Blob<Dtype>* const blob_G3_indptr;
  Blob<Dtype>* const blob_G3_indices;
  Blob<Dtype>* const blob_G3_data;
  Blob<Dtype>* const blob_G4_indptr;
  Blob<Dtype>* const blob_G4_indices;
  Blob<Dtype>* const blob_G4_data;

  Blob<Dtype>* const blob_Xall;
  Blob<Dtype>* const blob_Gall_indptr;
  Blob<Dtype>* const blob_Gall_indices;
  Blob<Dtype>* const blob_Gall_data;
  Blob<Dtype>* const blob_n_offset;

  Blob<Dtype>* const blob_Y1_ave;
  Blob<Dtype>* const blob_Y2_ave;
  Blob<Dtype>* const blob_Y3_ave;
  Blob<Dtype>* const blob_Y4_ave;

  Blob<Dtype>* const blob_Y1_max;
  Blob<Dtype>* const blob_Y2_max;
  Blob<Dtype>* const blob_Y3_max;

  Blob<Dtype>* const blob_Y1_gmax;
  Blob<Dtype>* const blob_Y2_gmax;
  Blob<Dtype>* const blob_Y3_gmax;

  Blob<Dtype>* const blob_Yall_ave;
  Blob<Dtype>* const blob_Yall_max;
  Blob<Dtype>* const blob_Yall_bmax;
  Blob<Dtype>* const blob_Yall_gmax;
  Blob<Dtype>* const blob_Yall_bave;
  Blob<Dtype>* const blob_Yall_gave;

  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GraphPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(GraphPoolingLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<GraphPoolingLayer<Dtype> > layer(
      new GraphPoolingLayer<Dtype>(layer_param));

  //Y=op_{G}(X)
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X1);
  this->blob_bottom_vec_.push_back(this->blob_G1_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G1_indices);
  this->blob_bottom_vec_.push_back(this->blob_G1_data);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_X1->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_X1->shape(1));

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indptr);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indices);
  this->blob_bottom_vec_.push_back(this->blob_Gall_data);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_Xall->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_Xall->shape(1));

  //Y is batched global max pooling over X
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_n_offset);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_n_offset->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_Xall->shape(1));

  //Y=max(X)
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X3);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_X3->shape(1));

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_Xall->shape(1));
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
    std::cout<<std::endl<<std::flush;
  }
}

template <typename Dtype>
static void check_all_eq(
  const int N,
  const Dtype* calc,
  const Dtype* expect,
  const Dtype threshold = Dtype(1e-3))
{
  for(int i=0; i<N; ++i)
  {
    EXPECT_NEAR(calc[i], expect[i], threshold);
  }
}

template
static void check_all_eq<float>(const int N, const float* calc, const float* expect, const float threshold);

template
static void check_all_eq<double>(const int N, const double* calc, const double* expect, const double threshold);

TYPED_TEST(GraphPoolingLayerTest, TestSparseDenseMatMul) {
  typedef typename TypeParam::Dtype Dtype;

  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::GPU && IS_VALID_CUDA) {

      Blob<Dtype>* A_indptr(new Blob<Dtype>());
      Blob<Dtype>* A_indices(new Blob<Dtype>());
      Blob<Dtype>* A_data(new Blob<Dtype>());
      Blob<Dtype>* B(new Blob<Dtype>());
      Blob<Dtype>* F(new Blob<Dtype>());
      //C=A*B
      Blob<Dtype>* Cgpu(new Blob<Dtype>());
      Blob<Dtype>* Ccpu(new Blob<Dtype>());
      //D=A'*F
      Blob<Dtype>* Dgpu(new Blob<Dtype>());
      Blob<Dtype>* Dcpu(new Blob<Dtype>());

      Timer timer;
      timer.Start();
#if defined(NDEBUG)
      const int M=80000;
      const int N=1024;
      const int K = 40000;
      const float density = 0.0005;
#else	// #if defined(_DEBUG)
      const int M = 512;
      const int N = 64;
      const int K = 256;
      const float density = 0.05;
#endif	// #if defined(_DEBUG)

      int* nnz_per_row = new int[M];
      int nnz=0;
      for(int i=0; i<M; ++i) {
        nnz_per_row[i] = rand() % ((int)(K*density)) + 1;
        nnz+=nnz_per_row[i];
      }

      //set A
      vector<int> shape0(1), shape(2);
      shape0[0]=M+1;
      A_indptr->Reshape(shape0);
      shape0[0]=nnz;
      A_indices->Reshape(shape0);
      A_data->Reshape(shape0);

      Dtype* aindptr=A_indptr->mutable_cpu_data();
      Dtype* aindices=A_indices->mutable_cpu_data();
      Dtype* adata=A_data->mutable_cpu_data();
      caffe_set(M+1, (Dtype)0., aindptr);
      caffe_set(nnz, (Dtype)0., aindices);
      caffe_set(nnz, (Dtype)0., adata);

      aindptr[0]=0;
      for(int i=0, cnt=0; i<M; ++i) {
        if(cnt>=nnz) break;
        for(int j=0; j<K; ++j) {
          if(cnt>=nnz) break;
          if(rand()%1000 < (int)(1000*density)) {
            adata[cnt] = (rand()%1000)/500.0;
            aindices[cnt]=j;
            ++cnt;
          }
        }
        aindptr[i+1]=cnt;
      }

      //set B
      shape[0]=K; shape[1]=N;
      B->Reshape(shape);
      Dtype* b=B->mutable_cpu_data();
      for(int i=0,cnt=0; i<K; ++i) {
        for(int j=0; j<N; ++j,++cnt) {
          b[cnt]=(rand()%1000)/100.0;
        }
      }

      //set F
      shape[0]=M; shape[1]=N;
      F->Reshape(shape);
      Dtype* f=F->mutable_cpu_data();
      for(int i=0,cnt=0; i<M; ++i) {
        for(int j=0; j<N; ++j,++cnt) {
          f[cnt]=(rand()%1000)/100.0;
        }
      }

      //set C
      shape[0]=M; shape[1]=N;
      Cgpu->Reshape(shape);
      Ccpu->Reshape(shape);
      Dtype* cgpu=Cgpu->mutable_gpu_data();
      Dtype* ccpu=Ccpu->mutable_cpu_data();

      //set D
      shape[0]=K; shape[1]=N;
      Dgpu->Reshape(shape);
      Dcpu->Reshape(shape);
      Dtype* dgpu=Dgpu->mutable_gpu_data();
      Dtype* dcpu=Dcpu->mutable_cpu_data();

      std::cout<<"Setting up:"<<timer.MilliSeconds()<<"ms"<<std::endl<<std::flush;
      timer.Start();

      const Dtype* Aindptr=A_indptr->cpu_data();
      const Dtype* Aindices=A_indices->cpu_data();
      const Dtype* Adata=A_data->cpu_data();
      const Dtype* pB=B->cpu_data();
      const Dtype* pF=F->cpu_data();

      const Dtype* gpB=B->gpu_data();
      const Dtype* gpF=F->gpu_data();

      std::cout<<"Copy data:"<<timer.MilliSeconds()<<"ms"<<std::endl<<std::flush;

      //calc C/D in cpu
      timer.Start();
      caffe_cpu_csr_gemm(CblasNoTrans, CblasNoTrans,
        M, N, K,
        (Dtype)1.0,
        Adata, Aindices, Aindptr,
        pB,
        (Dtype)0.0,
        ccpu,
        CblasRowMajor
      );
      std::cout<<"caffe_cpu_csr_gemm (A*B):"<<timer.MilliSeconds()<<"ms"<<std::endl<<std::flush;

      timer.Start();
      caffe_cpu_csr_gemm(CblasTrans, CblasNoTrans,
        K, N, M,
        (Dtype)1.0,
        Adata, Aindices, Aindptr,
        pF,
        (Dtype)0.0,
        dcpu,
        CblasRowMajor
      );
      std::cout<<"caffe_cpu_csr_gemm (A'*F):"<<timer.MilliSeconds()<<"ms"<<std::endl<<std::flush;

      //calc C/D in gpu
      timer.Start();
      caffe_gpu_csr_gemm(CblasNoTrans, CblasNoTrans,
        M, N, K,
        (Dtype)1.0,
		Adata, Aindices, Aindptr,
        gpB,
        (Dtype)0.0,
        cgpu,
        CblasRowMajor
      );
      std::cout<<"caffe_gpu_csr_gemm (A*B):"<<timer.MilliSeconds()<<"ms"<<std::endl<<std::flush;

      timer.Start();
      caffe_gpu_csr_gemm(CblasTrans, CblasNoTrans,
        K, N, M,
        (Dtype)1.0,
		Adata, Aindices, Aindptr,
        gpF,
        (Dtype)0.0,
        dgpu,
        CblasRowMajor
      );
      std::cout<<"caffe_gpu_csr_gemm (A'*F):"<<timer.MilliSeconds()<<"ms"<<std::endl<<std::flush;

      timer.Start();
      check_all_eq<Dtype>(M*N, Cgpu->cpu_data(), Ccpu->cpu_data());
      check_all_eq<Dtype>(K*N, Dgpu->cpu_data(), Dcpu->cpu_data());
      std::cout<<"check_all_eq:"<<timer.MilliSeconds()<<"ms"<<std::endl<<std::flush;

      std::cout << "SparseDenseMatMul passed!"<<std::endl<<std::flush;

      delete A_indptr;
      delete A_indices;
      delete A_data;
      delete B;
      delete F;
      delete Ccpu;
      delete Cgpu;
      delete Dcpu;
      delete Dgpu;
      delete[] nnz_per_row;

  } else {
    LOG(INFO) << "Only test with GPU.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY1AVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X1);
  this->blob_bottom_vec_.push_back(this->blob_G1_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G1_indices);
  this->blob_bottom_vec_.push_back(this->blob_G1_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y1_ave_calc = this->blob_top_->cpu_data();
    const Dtype* Y1_ave = this->blob_Y1_ave->cpu_data();
    check_all_eq(8, Y1_ave_calc, Y1_ave);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY1AVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X1);
  this->blob_bottom_vec_.push_back(this->blob_G1_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G1_indices);
  this->blob_bottom_vec_.push_back(this->blob_G1_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY2AVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X2);
  this->blob_bottom_vec_.push_back(this->blob_G2_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G2_indices);
  this->blob_bottom_vec_.push_back(this->blob_G2_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y2_ave_calc = this->blob_top_->cpu_data();
    const Dtype* Y2_ave = this->blob_Y2_ave->cpu_data();
    check_all_eq(8, Y2_ave_calc, Y2_ave);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY2AVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X2);
  this->blob_bottom_vec_.push_back(this->blob_G2_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G2_indices);
  this->blob_bottom_vec_.push_back(this->blob_G2_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY3AVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X3);
  this->blob_bottom_vec_.push_back(this->blob_G3_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G3_indices);
  this->blob_bottom_vec_.push_back(this->blob_G3_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y3_ave_calc = this->blob_top_->cpu_data();
    const Dtype* Y3_ave = this->blob_Y3_ave->cpu_data();
    check_all_eq(8, Y3_ave_calc, Y3_ave);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY3AVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X3);
  this->blob_bottom_vec_.push_back(this->blob_G3_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G3_indices);
  this->blob_bottom_vec_.push_back(this->blob_G3_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY4AVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X1);
  this->blob_bottom_vec_.push_back(this->blob_G4_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G4_indices);
  this->blob_bottom_vec_.push_back(this->blob_G4_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y4_ave_calc = this->blob_top_->cpu_data();
    const Dtype* Y4_ave = this->blob_Y4_ave->cpu_data();
    check_all_eq(12, Y4_ave_calc, Y4_ave);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY4AVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X1);
  this->blob_bottom_vec_.push_back(this->blob_G4_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G4_indices);
  this->blob_bottom_vec_.push_back(this->blob_G4_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardYallAVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indptr);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indices);
  this->blob_bottom_vec_.push_back(this->blob_Gall_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Yall_ave_calc = this->blob_top_->cpu_data();
    const Dtype* Yall_ave = this->blob_Yall_ave->cpu_data();
    check_all_eq(24, Yall_ave_calc, Yall_ave);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientYallAVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indptr);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indices);
  this->blob_bottom_vec_.push_back(this->blob_Gall_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardYallBAVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_n_offset);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Yall_bave_calc = this->blob_top_->cpu_data();
    const Dtype* Yall_bave = this->blob_Yall_bave->cpu_data();
    check_all_eq(6, Yall_bave_calc, Yall_bave);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientYallBAVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_n_offset);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardYallGAVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Yall_gave_calc = this->blob_top_->cpu_data();
    const Dtype* Yall_gave = this->blob_Yall_gave->cpu_data();
    check_all_eq(2, Yall_gave_calc, Yall_gave);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientYallGAVE) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_AVE);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY1MAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X1);
  this->blob_bottom_vec_.push_back(this->blob_G1_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G1_indices);
  this->blob_bottom_vec_.push_back(this->blob_G1_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y1_max_calc = this->blob_top_->cpu_data();
    const Dtype* Y1_max = this->blob_Y1_max->cpu_data();
    check_all_eq(8, Y1_max_calc, Y1_max);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY1MAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X1);
  this->blob_bottom_vec_.push_back(this->blob_G1_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G1_indices);
  this->blob_bottom_vec_.push_back(this->blob_G1_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY2MAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X2);
  this->blob_bottom_vec_.push_back(this->blob_G2_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G2_indices);
  this->blob_bottom_vec_.push_back(this->blob_G2_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y2_max_calc = this->blob_top_->cpu_data();
    const Dtype* Y2_max = this->blob_Y2_max->cpu_data();
    check_all_eq(8, Y2_max_calc, Y2_max);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY2MAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X2);
  this->blob_bottom_vec_.push_back(this->blob_G2_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G2_indices);
  this->blob_bottom_vec_.push_back(this->blob_G2_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY3MAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X3);
  this->blob_bottom_vec_.push_back(this->blob_G3_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G3_indices);
  this->blob_bottom_vec_.push_back(this->blob_G3_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y3_max_calc = this->blob_top_->cpu_data();
    const Dtype* Y3_max = this->blob_Y3_max->cpu_data();
    check_all_eq(8, Y3_max_calc, Y3_max);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY3MAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X3);
  this->blob_bottom_vec_.push_back(this->blob_G3_indptr);
  this->blob_bottom_vec_.push_back(this->blob_G3_indices);
  this->blob_bottom_vec_.push_back(this->blob_G3_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardYallMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indptr);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indices);
  this->blob_bottom_vec_.push_back(this->blob_Gall_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Yall_max_calc = this->blob_top_->cpu_data();
    const Dtype* Yall_max = this->blob_Yall_max->cpu_data();
    check_all_eq(8, Yall_max_calc, Yall_max);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientYallMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indptr);
  this->blob_bottom_vec_.push_back(this->blob_Gall_indices);
  this->blob_bottom_vec_.push_back(this->blob_Gall_data);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY1GMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X1);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y1_gmax_calc = this->blob_top_->cpu_data();
    const Dtype* Y1_gmax = this->blob_Y1_gmax->cpu_data();
    check_all_eq(2, Y1_gmax_calc, Y1_gmax);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY1GMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X1);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY2GMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X2);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y2_gmax_calc = this->blob_top_->cpu_data();
    const Dtype* Y2_gmax = this->blob_Y2_gmax->cpu_data();
    check_all_eq(2, Y2_gmax_calc, Y2_gmax);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY2GMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X2);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardY3GMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X3);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Y3_gmax_calc = this->blob_top_->cpu_data();
    const Dtype* Y3_gmax = this->blob_Y3_gmax->cpu_data();
    check_all_eq(2, Y3_gmax_calc, Y3_gmax);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientY3GMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_X3);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardYallGMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Yall_gmax_calc = this->blob_top_->cpu_data();
    const Dtype* Yall_gmax = this->blob_Yall_gmax->cpu_data();
    check_all_eq(2, Yall_gmax_calc, Yall_gmax);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientYallGMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestForwardYallBMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_n_offset);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    shared_ptr<GraphPoolingLayer<Dtype> > layer(
        new GraphPoolingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* Yall_bmax_calc = this->blob_top_->cpu_data();
    const Dtype* Yall_bmax = this->blob_Yall_bmax->cpu_data();
    check_all_eq(6, Yall_bmax_calc, Yall_bmax);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GraphPoolingLayerTest, TestGradientYallBMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_Xall);
  this->blob_bottom_vec_.push_back(this->blob_n_offset);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    layer_param.mutable_graph_pooling_param()->set_mode(GraphPoolingParameter_Mode_MAX);
    GraphPoolingLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0); //note the forth argument is set to 0, meaning we only check gradient w.r.t. to bottom[0]
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
