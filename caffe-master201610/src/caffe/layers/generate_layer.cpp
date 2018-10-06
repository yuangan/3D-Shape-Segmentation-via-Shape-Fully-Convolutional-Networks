#include <vector>

#include "caffe/layers/generate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GenerateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(top.size(), 1)<<"top needs 1";
	CHECK_EQ(bottom.size(), 2)<<"bottom needs 2";
	CHECK_EQ(bottom[0]->height(), bottom[1]->height())<<"the first and the second input need same";
	//get the next kernel width
	GenerateParameter generate_param = this->layer_param_.generate_param();
	if (generate_param.has_next_kernel_w()){
		next_kernel_w_ = generate_param.next_kernel_w();
	}
	else{
		//default: next convolution kernel_w is 4
		next_kernel_w_ = 4;
	}
	CHECK_EQ(bottom[1]->width(), next_kernel_w_ - 1) << "bottom[1]->width() and (next_kernel_w - 1) need to be equal";
}

template <typename Dtype>
void GenerateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//bottom[0] is the feature
	//bottom[1] is the node num
	//top[0] is the output
	channel_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	top[0]->Reshape(bottom[0]->num(), channel_, height_, next_kernel_w_);
//	gen_b1.Reshape(bottom[1]->num(), bottom[1]->channels(), height_, next_kernel_w_ - 1);
}

template <typename Dtype>
void GenerateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_GT(channel_, 0) << "channel must larger than zero";
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* index_data = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	long long int offset_top = 0;
	//c*h*w sequence
	for (int n = 0; n < bottom[0]->num(); ++n){
		for (int c = 0; c < channel_; ++c){
			//height
			long long int offset_b0 = (n*channel_ + c)*height_;
			int offset_b1 = n*bottom[1]->channels()*height_*(next_kernel_w_ - 1);
			for (int h = 0; h < height_; ++h){
				CHECK_EQ(bottom[0]->width(), 1) << "the first input need to be one column";
				//copy the bottom0 value
				top_data[offset_top] = bottom_data[offset_b0 + h];
				offset_top++;
				for (int w = 1; w < next_kernel_w_; ++w, ++offset_top){
					//todo: copy other value w.r.t index
					int tmp_offset = index_data[offset_b1];
					top_data[offset_top] = bottom_data[offset_b0 + tmp_offset];
					offset_b1++;
				}
			}
		}
	}
}

template <typename Dtype>
void GenerateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_set(bottom[1]->count(), Dtype(0),bottom[1]->mutable_cpu_diff());
		int tmp_index = 0;
		for (int n = 0; n < bottom[0]->num(); ++n){
			CHECK_EQ(tmp_index, n*channel_*height_) << "tmp_index need to be equal to n*channel_height_";
			for (int c = 0; c < channel_; ++c){
				for (int h = 0; h < height_; ++h){
					bottom_diff[tmp_index] = top_diff[tmp_index*next_kernel_w_];
					tmp_index++;
				}
			}
		}
	}
}
/*
#ifdef CPU_ONLY
STUB_GPU(GenerateLayer);
#endif
*/
INSTANTIATE_CLASS(GenerateLayer);
REGISTER_LAYER_CLASS(Generate);

}  // namespace caffe
