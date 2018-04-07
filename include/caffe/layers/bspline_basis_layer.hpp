#ifndef CAFFE_BSPLINE_BASIS_LAYER_HPP_
#define CAFFE_BSPLINE_BASIS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief modeling the N(x) function, which is the B-spline basis fuction
 *  currently there is NO backward, and NO learnable parameters
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BSplineBasisLayer : public Layer<Dtype> {
 public:
  explicit BSplineBasisLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BSplineBasis"; }
  //    bottom[0] <NxD|BxNxD>: input signal, each row serves as input argument x to the BSpline basis function N(x)
  //    top[0] <NxC|BxNxC>: output signal, C=C1*...*Cd, where Ci=Ki-degree-1 indicates the number of control points along the i-th dimension
  //    where knot vectors are parsed from BSplineBasisParameter
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int degree_;
  int output_dim_; //number of control points/basis inferred from degree and knot vectors in LayerSetUp
  vector<Blob<Dtype>*> knot_vectors_;
  Blob<int> strides_;
};

}  // namespace caffe

#endif  // CAFFE_BSPLINE_BASIS_LAYER_HPP_
