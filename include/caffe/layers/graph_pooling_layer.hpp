#ifndef CAFFE_GRAPH_POOLING_LAYER_HPP_
#define CAFFE_GRAPH_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/sparse_dense_matmul.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Graph pooling layer, Yi = op({Xj|j \in N(i)})
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class GraphPoolingLayer : public Layer<Dtype> {
 public:
  explicit GraphPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GraphPooling"; }

  //we expect three kinds of bottom input cases:
  //a). 4 bottoms:
  //        bottom[0]: X, signals on nodes, shape=[n, D_]
  //        bottom[1]: indptr, graph (CSR), shape=[m+1,]
  //        bottom[2]: indices,graph (CSR), shape=[nnz,]
  //        bottom[3]: data,   graph (CSR), shape=[nnz,]
  //        top[0]: Y, output signal after pooling, shape=[m, D_]
  //    in this case, this layer performs Y=op_{G}(X) where op is a pooling operation built from the graph,
  //    either G directly for AVE mode or max for MAX mode. Users need to ensure (indptr,indices,data) forms
  //    a valid CSR sparse matrix of shape [m,n]. (m==n and an undirected graph must be true for MAX mode)
  //    Note this case can handle batched input as long as the graph is built such that
  //    each graph in the batch is a diagonal block
  //b). 2 bottoms:
  //        bottom[0]: X, signals on nodes, shape=[n1+...+nB, D_]
  //        bottom[1]: n offset between batches, shape=[B,], a vector storing [n1, n1+n2, ..., n1+...+nB]
  //        top[0]: Y, output signal after pooling, shape=[B, D_]
  //    in this case, this layer performs global max/mean pooling for each graph in the batch
  //c). 1 bottoms:
  //        bottom[0]: X, signals on nodes, shape=[n, D_]
  //        top[0]: Y, output signal after pooling, shape=[1, D_]
  //    in this case, this layer performs global max/mean pooling
  virtual inline int MinBottomBlobs() const { return 1; }
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

  //parameters set from prototxt
  GraphPoolingParameter_Mode mode_; //note: the type is generated from caffe.proto in file caffe.pb.h

  Blob<int> idx_; //storing max pooling index, idx_[i,j] means Y[i,j] comes from X[idx_[i,j], j] in max pooling
};

}  // namespace caffe

#endif  // CAFFE_GRAPH_POOLING_LAYER_HPP_
