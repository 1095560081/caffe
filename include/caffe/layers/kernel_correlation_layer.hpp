/*
 * kernel_correlation_layer.hpp
 *
 *  Created on: Oct 16, 2017
 *      Author: cfeng
 */

#ifndef INCLUDE_CAFFE_LAYERS_KERNEL_CORRELATION_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_KERNEL_CORRELATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{

/**
 * @brief Kernel correlation layer
 *        c(q, k) = \sum_i \sum_{j \in N(q)} exp{ -||k_i+q-p_j||^2 / \sigma }
 *  where:
 *        q <dx1>:     the current reference point
 *        k <mxd>:     the current Kernel point set containing m points {k_i}
 *        p_j <dx1>:   a point in the neighborhood of q
 *        N(q):        index set of q's neighborhood points
 *        c(q, k) <1>: kernel correlation between 1) the kernel point set and 2) point q's neighborhood point set
 *        \sigma <1>:  bandwidth of the kernel, a hyper-parameter to be specified in the prototxt
 *
 *  Note: this layer usually has more than one kernel point set, the number of kernels (l) and the number of points per kernel (m)
 *  are hyper-parameters similar to convolution layer's number of output and convolution window size
 *
 *  Regarding the input/output in the form of blobs:
 *        bottom[0]: P={q}, all points,       shape=[n, d]
 *        bottom[1]: indptr, graph (CSR),     shape=[n+1,]
 *        bottom[2]: indices,graph (CSR),     shape=[nnz,]
 *        top[0]:    C={c}, all correlations, shape=[n, l]
 *        params[0]: K={k}, all kernels,      shape=[l,m,d]
 *  where bottom[1] and bottom[2] stores a sparse matrix of shape [n,n], each row of which stores N(p).
 */
template<typename Dtype>
class KernelCorrelationLayer : public Layer<Dtype>
{
 public:
  explicit KernelCorrelationLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        num_output_(0),
        num_points_per_kernel_(0),
        sigma_(Dtype(0.0)),
        d_(0)
  {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const
  {
    return "KernelCorrelation";
  }

  virtual inline int ExactNumBottomBlobs() const
  {
    return 3;
  }
  virtual inline int ExactNumTopBlobs() const
  {
    return 1;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  int num_output_;
  int num_points_per_kernel_;
  Dtype sigma_;
  int d_;

  Blob<Dtype> tmp_dk_i;
  Blob<Dtype> tmp_multiplier;
};

}  // namespace caffe

#endif /* INCLUDE_CAFFE_LAYERS_KERNEL_CORRELATION_LAYER_HPP_ */
