#ifndef CAFFE_MATRIX_TRANSPOSE_LAYER_HPP_
#define CAFFE_MATRIX_TRANSPOSE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief transpose input matrix X and output $Z=X^T$
 *
 * Input:
 * X: <B[xB2...xBn]xMxN> or <MxN>
 * Output:
 * Z: <B[xB2...xBn]xNxM> or <NxM>
 */
template <typename Dtype>
class MatrixTransposeLayer : public Layer<Dtype> {
 public:
  explicit MatrixTransposeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MatrixTranspose"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_; //X <MxN>, Z<NxM>
  int N_;
};

}  // namespace caffe

#endif  // CAFFE_MATRIX_TRANSPOSE_LAYER_HPP_
