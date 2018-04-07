#include <vector>
#include <sstream>

#include "caffe/filler.hpp"
#include "caffe/layers/bspline_basis_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define BSPLINE_OPENMP 1

inline void split(
  const std::string &s, char delim,
  std::vector<std::string> &elems, bool skipEmptyLine=true)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
    	if(skipEmptyLine && item.empty()) continue;
        elems.push_back(item);
    }
}

template<typename Dtype>
inline Dtype str2num(const std::string& str)
{
	std::stringstream ss;
	ss << str;
	Dtype ret;
	ss >> ret;
	return ret;
}

template<typename Dtype>
static void string2blob(const std::string& str, Blob<Dtype>& blob)
{
  std::vector<std::string> elems;
  split(str, ',', elems);

  std::vector<int> shape(1);
  shape[0]=elems.size();
  blob.Reshape(shape);

  Dtype* ptr=blob.mutable_cpu_data();
  for(size_t i=0; i<elems.size(); ++i) {
    ptr[i] = str2num<Dtype>(elems[i]);

    std::cout<<ptr[i];
    if(i<elems.size()-1) {
      std::cout<<",";
    }

    if(i==0) continue;
    CHECK_GE(ptr[i], ptr[i-1])
      <<"knot vector is not in ascending order!";
  }
  std::cout<<std::endl<<std::flush;
}

template <typename Dtype>
void BSplineBasisLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->degree_ = this->layer_param_.bspline_basis_param().degree();

  const int naxes = bottom[0]->num_axes();
  if(naxes!=2 && naxes!=3)
    LOG(ERROR) << "Input blob must be a matrix of shape (NxD) or (BxNxD)!";
  CHECK_EQ(this->layer_param_.bspline_basis_param().knot_vector_size(), bottom[0]->shape(-1))
    << "Not enough knot vectors are specified in bspline_basis_param!";

  const int input_dim = bottom[0]->shape(-1);
  this->knot_vectors_.resize(input_dim);

  std::vector<int> shape(1);
  shape[0]=input_dim;
  this->strides_.Reshape(shape);
  int* strides = this->strides_.mutable_cpu_data();

  int num_control = 1;
  for(int i=input_dim-1; i>=0; --i) {
    this->knot_vectors_[i] = new Blob<Dtype>();
    const std::string knot_vector_str = this->layer_param_.bspline_basis_param().knot_vector(i);

    std::cout<<"knot_vectors["<<i<<"]=";
    string2blob(knot_vector_str, *this->knot_vectors_[i]);

    CHECK_GE(this->knot_vectors_[i]->count(), 2*this->degree_+2)
      << "Each knot vector should have at least 2*degree+2 knots";

    const int nctrl_i = this->knot_vectors_[i]->count() - this->degree_ - 1;
    num_control *= nctrl_i;

    strides[i]=num_control;
  }
  this->output_dim_ = num_control;
}

template <typename Dtype>
void BSplineBasisLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int naxes = bottom[0]->num_axes();
  if(naxes!=2 && naxes!=3)
    LOG(ERROR) << "Input blob must be a matrix of shape (NxD) or (BxNxD)!";

  CHECK_EQ(bottom[0]->shape(-1), this->knot_vectors_.size())
    << "Input blob's second dimension not compatible with the number of knot vectors!";

  vector<int> top_shape = bottom[0]->shape();
  top_shape[top_shape.size()-1] = this->output_dim_; //NxC or BxNxC, C=output_dim_
  top[0]->Reshape(top_shape);
}

//last dimension moves fastest
//inline void ind2sub(int ind, const int*const strides, const int num_axes, std::vector<int>& sub)
//{
//  //CHECK_LT(ind, strides[0]) << "ind out of bound!";
//  sub.resize(num_axes);
//  for(size_t i=0; i<num_axes-1; ++i)
//  {
//    sub[i] = ind / strides[i+1];
//    ind -= sub[i]*strides[i+1];
//  }
//  sub[num_axes-1] = ind;
//}


inline int get_num_knots(const int degree, const int nCtrl) { return degree + nCtrl + 1; }
inline int get_num_ctrls(const int degree, const int nKnots) { return nKnots - degree - 1; }

template<typename T>
static int find_span(
  const T& u,               // knotVec[i] <= u < knotVec[i+1], i is the return value
  const T*const knotVec,    // knot vector
  const int knotVec_size,
  const int degree)
{
  const int nCtrl = get_num_ctrls(degree, knotVec_size);
  if (u <= knotVec[degree]) return degree;
  if (u >= knotVec[nCtrl]) return nCtrl - 1;
  int low = degree;
  int high = nCtrl;
  int mid = (low + high) / 2;
  while (true) {
    if (u < knotVec[mid]) high = mid;
    else if (u >= knotVec[mid + 1]) low = mid;
    else break;
    mid = (low + high) / 2;
  }
  return mid;
}

// given knotVec[span]<=u<knotVec[span+1], find (degree+1) many non-zero basis functions
// N[0],N[1],...,N[degree] corresponding to points P[span-degree],P[span-degree+1],...,P[span] respectively
template<typename T>
static void basis_funs_core(
    const int span,
    const T& u,
    const int degree,
    const T*const knotVec,
    T* N)
{
    //N.resize(degree + 1, (T)(0.0));
    std::vector<T> left(degree+1, (T)(0.0));
    std::vector<T> right(degree+1, (T)(0.0));

    N[0] = 1;
    for (int j = 1; j <= degree; ++j) {
        left[j] = u - knotVec[span + 1 - j];
        right[j] = knotVec[span + j] - u;
        T saved = 0.0;
        for (int r = 0; r < j; ++r) {
            T temp = N[r] / (right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

template<typename T>
inline void basis_funs(
    const T& u,
    const int degree,
    const T*const knotVec,
    const int knotVec_size,
    int& span,
    T* N)
{
    span = find_span(u, knotVec, knotVec_size, degree);
    basis_funs_core(span, u, degree, knotVec, N);
}

template <typename Dtype>
void BSplineBasisLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* X = bottom[0]->cpu_data();
  const int* strides = this->strides_.cpu_data();
  Dtype* Y = top[0]->mutable_cpu_data();

  const bool has_batch = bottom[0]->num_axes()==3;
  const int B = has_batch ? bottom[0]->shape(0) : 1;
  const int N = bottom[0]->shape(0+int(has_batch));
  const int D = bottom[0]->shape(1+int(has_batch));
  const int C = this->output_dim_;

#if BSPLINE_OPENMP
#pragma omp parallel for
#endif
  for(int b=0; b<B; ++b)
  {
    const Dtype* Xb = X+b*N*D;
    Dtype* Yb = Y+b*N*C;
    for(int i=0; i<N; ++i)
    {
      std::vector<Dtype> N_all((this->degree_+1)*D, (Dtype)0.0);
      std::vector<int> span_all(D, 0);
      const Dtype* Xi = Xb+i*D;
      Dtype* Yi = Yb+i*C; //i-th row of output Y

      for(int j=0; j<D; ++j)
      {
        basis_funs(Xi[j],
                   this->degree_,
                   this->knot_vectors_[j]->cpu_data(),
                   this->knot_vectors_[j]->count(),
                   span_all[j],
                   &N_all[0]+(this->degree_+1)*j);
      }

      for(int ind=0; ind<C; ++ind) //k-th column of output Y
      {
        int the_ind = ind;

        Dtype Yik(1.);
        for(int j=0; j<D; ++j)
        {
          const int sub = j==D-1 ? the_ind : (the_ind / strides[j+1]); //j-th sub
          if(j<D-1) the_ind -= sub*strides[j+1];
          const int jj=sub-(span_all[j]-this->degree_);
          if(jj<0 || jj>this->degree_) {
            Yik=(Dtype)0.;
            break;
          }
          Yik*=N_all[jj+(this->degree_+1)*j];
        }
        Yi[ind]=Yik;
      }//ind<output_dim_
    }//i<N
  }//b<B
}

template <typename Dtype>
void BSplineBasisLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(ERROR)<<"Backward not implemented for BSplineBasisLayer!";
  }
}

#ifdef CPU_ONLY
STUB_GPU(BSplineBasisLayer);
#endif

INSTANTIATE_CLASS(BSplineBasisLayer);
REGISTER_LAYER_CLASS(BSplineBasis);

}  // namespace caffe
