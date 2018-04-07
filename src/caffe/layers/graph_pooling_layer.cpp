#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/graph_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#define GRAPH_USE_OPENMP 0

template <typename Dtype>
void GraphPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  if(bottom.size()!=4 && bottom.size()!=2 && bottom.size()!=1) {
    LOG(ERROR) << "number of bottoms to GraphPoolingLayer should be either 4 or 2 or 1, instead of "
               << bottom.size() << "!";
  }

  this->mode_ = this->layer_param_.graph_pooling_param().mode();
}

template <typename Dtype>
void GraphPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  //check X
  CHECK_EQ(bottom[0]->num_axes(), 2)
    << "bottom[0] should always have shape (N,D)!";

  vector<int> shape(2);
  switch(bottom.size()) {
  case 4:
    //check indptr
    CHECK_EQ(bottom[1]->num_axes(), 1)
      << "bottom[1] should always be an vector!";
    if(this->mode_ == GraphPoolingParameter_Mode_MAX) {
      CHECK_EQ(bottom[1]->count(), bottom[0]->shape(0)+1)
        << "bottom[1]'s length incompatible with bottom[0]'s number of rows!";
    }
    CHECK_EQ(bottom[1]->cpu_data()[0], 0)
        << "Invalid bottom[1] (indptr), first element is not 0!";

    //check indices
    CHECK_EQ(bottom[2]->num_axes(), 1)
      << "bottom[2] (indices) should always be an vector!";

    //check data
    CHECK_EQ(bottom[3]->num_axes(), 1)
      << "bottom[3] should always be an vector!";
    CHECK_EQ(bottom[3]->count(), bottom[2]->count())
      << "bottom[3] (data) incompatible with bottom[2] (indices)!";

    //TODO: use reinterpret_cast<int>(indptr[i]) in the future to support indptr[i]>=16777217
    //TODO: similarly for indices[kk]; add a sanity check for now
    CHECK_LT(bottom[2]->count(), 16777216)
      << "Number of graph edges (i.e., non-zero-entries of the graph matrix)>2^24=16777216 is not supported for now!"
         "Ref: https://stackoverflow.com/a/15094846/2303236";
    CHECK_LT(bottom[0]->shape(0), 16777216)
      << "Number of graph nodes>2^24=16777216 is not supported for now!"
         "Ref: https://stackoverflow.com/a/15094846/2303236";

    shape[0] = bottom[1]->count()-1; //rows for Y, determined from the sparse graph matrix
    shape[1] = bottom[0]->shape(1); //dimension of signal
    top[0]->Reshape(shape);
    break;
  case 2:
    //check n_offset
    CHECK_EQ(bottom[1]->num_axes(), 1)
      << "bottom[1] should always be an vector!";

    CHECK_LT(bottom[0]->shape(0), 16777216)
      << "Number of graph nodes>16777216 is not supported for now!";

    shape[0] = bottom[1]->count();
    shape[1] = bottom[0]->shape(1);
    top[0]->Reshape(shape);
    break;
  case 1:
    shape[0] = 1;
    shape[1] = bottom[0]->shape(1);
    top[0]->Reshape(shape);
    break;
  }

  if(this->mode_ == GraphPoolingParameter_Mode_MAX) {
    shape = top[0]->shape();
    this->idx_.Reshape(shape);
  }
}

template <typename Dtype>
void GraphPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X = bottom[0]->cpu_data();
  Dtype* Y = top[0]->mutable_cpu_data();

  const int N = bottom[0]->shape(0); //number of nodes for X
  const int D = bottom[0]->shape(1); //dimension of signal
  const int n_bottom = bottom.size();
#define MYIDX(ii, jj) ((ii)*D+(jj))  //(ii,jj)-th entry in matrix X or matrix Y

  if(this->mode_ == GraphPoolingParameter_Mode_MAX) //max pooling
  {
    int* idx = this->idx_.mutable_cpu_data();

    if(n_bottom==4) //graph max pooling
    {
      const Dtype* indptr = bottom[1]->cpu_data();
      const Dtype* indices = bottom[2]->cpu_data();

#if GRAPH_USE_OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<N; ++i) //for each row of Y
      {
        const int ind_begin = (int)indptr[i]; //TODO: use reinterpret_cast<int>
        const int ind_end = (int)indptr[i+1]; //TODO: use reinterpret_cast<int>
        for(int j=0; j<D; ++j) //for each dimension of the signal on the i-th node
        {
          const int ij = MYIDX(i, j); //always start with the signal itself
          Y[ij] = X[ij];
          idx[ij] = i;
          for(int kk=ind_begin; kk<ind_end; ++kk) //for each neighbor node index
          {
            const int k = (int)indices[kk]; //TODO: use reinterpret_cast<int>
            const int kj = MYIDX(k, j);
            if(Y[ij]<X[kj]) {
              Y[ij]=X[kj];
              idx[ij] = k;
            }
          }//for kk
        }//for j<D
      }//for i<N
    }
    else if(n_bottom==2) //batched global max pooling
    {
      const Dtype* n_offset = bottom[1]->cpu_data();
      const int B = bottom[1]->count(); //number of data in the batch

      CHECK_EQ((int)n_offset[B-1], N)
        << "bottom[1] (n_offset) indicates a different number of nodes than bottom[0] does!";

      const int BD = B*D;
#if GRAPH_USE_OPENMP
#pragma omp parallel for
#endif
      for(int bj=0; bj<BD; ++bj) //for each entry of Y
      {
        const int b = bj/D;
        const int j = bj%D;
        const int ind_begin = b==0 ? 0 : (int)n_offset[b-1]; //TODO: use reinterpret_cast<int>
        const int ind_end = (int)n_offset[b]; //TODO: use reinterpret_cast<int>
        for(int k=ind_begin; k<ind_end; ++k) //for all node index in this data
        {
          const int kj = MYIDX(k, j);
          if(k==ind_begin || Y[bj]<X[kj]) {
            Y[bj]=X[kj];
            idx[bj] = k;
          }
        }//for k
      }//for bj
    }
    else //global max pooling
    {
#if GRAPH_USE_OPENMP
#pragma omp parallel for
#endif
      for(int j=0; j<D; ++j) //for each dimension of the signal
      {
        for(int i=0; i<N; ++i) //for each node
        {
          const int ij = MYIDX(i, j);
          if (i==0 || Y[j]<X[ij]) {
            Y[j] = X[ij];
            idx[j] = i;
          }
        }//for j<D
      }//for i<N
    }
  }//if mode_==MAX
  else //average pooling, Y=AX
  {
    if(n_bottom==4) //graph ave pooling
    {
      const Dtype* indptr = bottom[1]->cpu_data();
      const Dtype* indices = bottom[2]->cpu_data();
      const Dtype* data = bottom[3]->cpu_data();

      const int M = bottom[1]->count()-1; //rows for Y

      caffe_cpu_csr_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        M, D, N,
        (Dtype)1.0,
        data, indices, indptr,
        X,
        (Dtype)0.0,
        Y,
        CblasRowMajor
      );
    }
    else if(n_bottom==2) //batched global ave pooling
    {
      const Dtype* n_offset = bottom[1]->cpu_data();
      const int B = bottom[1]->count(); //number of data in the batch

      CHECK_EQ((int)n_offset[B-1], N)
        << "bottom[1] (n_offset) indicates a different number of nodes than bottom[0] does!";

      const int BD = B*D;
#if GRAPH_USE_OPENMP
#pragma omp parallel for
#endif
      for(int bj=0; bj<BD; ++bj) //for each entry of Y
      {
        const int b = bj/D;
        const int j = bj%D;
        const int ind_begin = b==0 ? 0 : (int)n_offset[b-1]; //TODO: use reinterpret_cast<int>
        const int ind_end = (int)n_offset[b]; //TODO: use reinterpret_cast<int>
        Y[bj]=(Dtype)0.0;
        for(int k=ind_begin; k<ind_end; ++k) //for all node index in this data
        {
          const int kj = MYIDX(k, j);
          Y[bj]+=X[kj];
        }//for k
        const int nk=ind_end-ind_begin;
        if(nk>0)
          Y[bj]/=(Dtype)nk;
      }//for bj
    }
    else //global ave pooling
    {
#if GRAPH_USE_OPENMP
#pragma omp parallel for
#endif
      for(int j=0; j<D; ++j) //for each dimension of the signal
      {
        Y[j]=(Dtype)0.0;
        for(int i=0; i<N; ++i) //for each node
        {
          const int ij = MYIDX(i, j);
          Y[j]+=X[ij];
        }//for j<D
        Y[j]/=(Dtype)N;
      }//for i<N
    }
  }//if mode_!=MAX

#undef MYIDX
}

template <typename Dtype>
void GraphPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) return;

  const Dtype* dY = top[0]->cpu_diff();
  Dtype* dX = bottom[0]->mutable_cpu_diff();

  const int N = bottom[0]->shape(0); //number of nodes for X/Y
  const int D = bottom[0]->shape(1); //dimension of signal
  const int n_bottom = bottom.size();
#define MYIDX(ii, jj) ((ii)*D+(jj))  //(ii,jj)-th entry in matrix X or matrix Y

  if(this->mode_ == GraphPoolingParameter_Mode_MAX) //max pooling
  {
    const int* idx = this->idx_.cpu_data();

    if(n_bottom==4) //graph max pooling
    {
      const Dtype* indptr = bottom[1]->cpu_data();
      const Dtype* indices = bottom[2]->cpu_data();

#if GRAPH_USE_OPENMP
#pragma omp parallel for
#endif
      for(int ij=0; ij<N*D; ++ij) //for each entry of X.diff
      {
        const int i = ij/D;
        const int j = ij%D;
        const int ind_begin = (int)indptr[i]; //TODO: use reinterpret_cast<int>
        const int ind_end = (int)indptr[i+1]; //TODO: use reinterpret_cast<int>
        Dtype gradient = idx[ij]==i ? dY[ij] : (Dtype)0.;
        for(int kk=ind_begin; kk<ind_end; ++kk) //for each neighbor node index
        {
          const int k = (int)indices[kk]; //TODO: use reinterpret_cast<int>
          const int kj = MYIDX(k, j);
          if(idx[kj]==i) { //TODO: this only works for undirected graph
            gradient += dY[kj];
          }
        }//for kk
        dX[ij]=gradient;
      }//for i<N
    }
    else if(n_bottom==2) //batched global max pooling
    {
      caffe_set(bottom[0]->count(), (Dtype)0., dX);

      const int B = bottom[1]->count(); //number of data in the batch

      for(int b=0; b<B; ++b) //for each data in the batch
      {
        for(int j=0; j<D; ++j) //for each dimension of the signal
        {
          const int src = MYIDX(b, j);
          const int dst = MYIDX(idx[src], j);
          dX[dst] += dY[src];
        }
      }
    }
    else //global max pooling
    {
      caffe_set(bottom[0]->count(), (Dtype)0., dX);
      for(int j=0; j<D; ++j) //for each dimension of the signal
      {
        const int src = j;
        const int dst = MYIDX(idx[src], j);
        dX[dst] += dY[src];
      }
    }
  }
  else //average pooling, Y=AX
  {
    if(n_bottom==4) //graph ave pooling
    {
      const Dtype* indptr = bottom[1]->cpu_data();
      const Dtype* indices = bottom[2]->cpu_data();
      const Dtype* data = bottom[3]->cpu_data();

      const int M = bottom[1]->count()-1; //rows for Y

      caffe_cpu_csr_gemm<Dtype>(CblasTrans, CblasNoTrans,
        N, D, M,
        (Dtype)1.0,
        data, indices, indptr,
        dY,
        (Dtype)0.0,
        dX,
        CblasRowMajor
      );
    }
    else if(n_bottom==2) //batched global ave pooling
    {
      const Dtype* n_offset = bottom[1]->cpu_data();

      caffe_set(bottom[0]->count(), (Dtype)0., dX);

      const int B = bottom[1]->count(); //number of data in the batch

#if GRAPH_USE_OPENMP
#pragma omp parallel for
#endif
      for(int b=0; b<B; ++b) //for each data in the batch
      {
        const int ind_begin = b==0 ? 0 : (int)n_offset[b-1]; //TODO: use reinterpret_cast<int>
        const int ind_end = (int)n_offset[b]; //TODO: use reinterpret_cast<int>
        const int nk=ind_end-ind_begin;
        for(int k=ind_begin; k<ind_end; ++k) //for all node index in this data
        {
          caffe_cpu_scale(D, (Dtype)(1.0/nk), dY+b*D, dX+k*D);
        }//for k
      }//for bj

    }
    else //global ave pooling
    {
#if GRAPH_USE_OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<N; ++i) //for each dimension of the signal
      {
        caffe_cpu_scale(D, (Dtype)(1.0/N), dY, dX+i*D);
      }
    }
  }
#undef MYIDX
}//Backward_cpu

#ifdef CPU_ONLY
STUB_GPU(GraphPoolingLayer);
#endif

INSTANTIATE_CLASS(GraphPoolingLayer);
REGISTER_LAYER_CLASS(GraphPooling);

}  // namespace caffe
