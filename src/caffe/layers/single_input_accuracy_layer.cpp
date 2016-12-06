#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/single_input_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SingleInputAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->shape(1)<=2)<<"bottom[0] must be Nx1xHxW or Nx2xHxW!";
}

template <typename Dtype>
void SingleInputAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->shape(1)==1) {
    CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "bottom[0] and bottom[1] must have equal count if of shape Nx1xHxW!";
  } else {
    CHECK_EQ(bottom[0]->count(), 2*bottom[1]->count())
      << "bottom[0]'s count must be twice as many as bottom[1]'s if of shape Nx2xHxW!";
  }

  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    vector<int> top1_shape(1, 5); //TP, TN, FP, FN, COUNT
    top[1]->Reshape(top1_shape);
  }
}

template <typename Dtype>
void SingleInputAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int data_channels = bottom[0]->shape(1);
  const int count = bottom[1]->count();
  const Dtype thresh = 0.5; //TODO: add a list of thresholds to test
  Dtype TP=0, FP=0, FN=0, TN=0;
  for (int i = 0; i < count; ++i) {
      const int label_value = static_cast<int>(bottom_label[i]);
      const Dtype data_value = (data_channels==1)?bottom_data[i]:bottom_data[i*2+1];
      DCHECK_GE(label_value, 0);
      //LOG(INFO) << "y="<<label_value<<", p="<<data_value;
      if (label_value>0) {
        if (data_value>thresh)
          ++TP;
        else
          ++FN;
      } else {
        if (data_value>thresh)
          ++FP;
        else
          ++TN;
      }
  }
  //LOG(INFO) << "count=" << count << ", TP="<<TP<<",TN="<<TN<<",FP="<<FP<<",FN="<<FN;

  const Dtype accuracy = (TP+TN)/count;

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy;
  if (top.size() > 1) {
    top[1]->mutable_cpu_data()[0] = TP/count;
    top[1]->mutable_cpu_data()[1] = TN/count;
    top[1]->mutable_cpu_data()[2] = FP/count;
    top[1]->mutable_cpu_data()[3] = FN/count;
    top[1]->mutable_cpu_data()[4] = count;
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(SingleInputAccuracyLayer);
REGISTER_LAYER_CLASS(SingleInputAccuracy);

}  // namespace caffe
