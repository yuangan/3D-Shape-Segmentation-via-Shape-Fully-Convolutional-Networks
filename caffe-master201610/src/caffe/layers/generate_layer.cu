#include <vector>

#include "caffe/layers/generate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void GenerateForward(const int n, const Dtype* in1,const Dtype* in2, Dtype* out,
const int num,const int channels,const int height,const int next_kernel_w){
	CUDA_KERNEL_LOOP(index,n){
		const int offset_width = index % next_kernel_w;
		const int offset_height = (index / next_kernel_w) % height;
		const int offset_channel = (index / next_kernel_w / height) % channels;
		const int offset_n = index / next_kernel_w / height /channels;
		int offset_b1 = (offset_n*channels + offset_channel) * height;
		if(offset_width > 0){
			//
			int offset_b2 = offset_n*height*(next_kernel_w - 1) + offset_height * (next_kernel_w - 1) + (offset_width - 1);
			int tmp_offset = in2[offset_b2];
			out[index] = in1[offset_b1 + tmp_offset];
		}else{
			out[index] = in1[offset_b1 + offset_height];
		}
	}
}
template <typename Dtype>
void GenerateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* index_data = bottom[1]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = top[0]->count();
	//c*h*w sequence
	GenerateForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	count, bottom_data, index_data, top_data, bottom[0]->num(), channel_, height_, next_kernel_w_);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void GenerateBackward(const int n, const Dtype* in, Dtype* out, const int next_kernel_w){
	CUDA_KERNEL_LOOP(index,n){
        out[index] = in[index * next_kernel_w];
}
}

template <typename Dtype>
void GenerateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
		GenerateBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
     	count, top_diff, bottom_diff, next_kernel_w_);
	    CUDA_POST_KERNEL_CHECK;
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(GenerateLayer);

}  // namespace caffe
