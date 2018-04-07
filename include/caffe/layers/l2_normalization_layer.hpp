/*
 * l2_normalization_layer.hpp
 *
 *  Created on: Oct 20, 2017
 *      Author: cfeng
 */

#ifndef INCLUDE_CAFFE_LAYERS_L2_NORMALIZATION_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_L2_NORMALIZATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief L2-normalize input, maybe independently along an specific axis
 *
 * Input:
 * X: <S1xS2...xSN>
 * Output:
 * Y: <S1xS2...xSN>
 *
 */
template <typename Dtype>
class L2NormalizationLayer : public Layer<Dtype> {
 public:
  explicit L2NormalizationLayer(const LayerParameter& param)
      : Layer<Dtype>(param), axis_(0), is_global_(true), eps_(1e-10), n_(0), m_(0), too_small_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "L2Normalization"; }
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

  int axis_;
  bool is_global_;
  Dtype eps_;
  Dtype n_, m_; //catch for is_global_==true
  bool too_small_;
};

}  // namespace caffe

#endif /* INCLUDE_CAFFE_LAYERS_L2_NORMALIZATION_LAYER_HPP_ */
