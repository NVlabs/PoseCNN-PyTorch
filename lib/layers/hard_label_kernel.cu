// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
// This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
// text can be found in LICENSE.md

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename Dtype>
__global__ void HardLabelForward(const int nthreads, const float threshold, const float sample_percentage,
    const Dtype* bottom_prob, const Dtype* bottom_label, const Dtype* bottom_rand, Dtype* top_data)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    if (bottom_label[index] > 0 && (bottom_prob[index] < threshold || bottom_rand[index] < sample_percentage))
      top_data[index] = 1.0; 
  }
}


std::vector<at::Tensor> hard_label_cuda_forward(
    float threshold,
    float sample_percentage,
    at::Tensor bottom_prob,
    at::Tensor bottom_label,
    at::Tensor bottom_rand) 
{
  // run kernels
  const int kThreadsPerBlock = 1024;
  int output_size;

  if (bottom_prob.dim() == 4)
    output_size = bottom_prob.size(0) * bottom_prob.size(1) * bottom_prob.size(2) * bottom_prob.size(3);
  else
    output_size = bottom_prob.size(0) * bottom_prob.size(1);

  auto top_data = at::zeros(bottom_prob.sizes(), bottom_prob.options());

  AT_DISPATCH_FLOATING_TYPES(bottom_prob.type(), "hard_label_forward_cuda", ([&] {

    // compute the losses and gradients
    HardLabelForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size,
        threshold,
        sample_percentage,
        bottom_prob.data<scalar_t>(),
        bottom_label.data<scalar_t>(),
        bottom_rand.data<scalar_t>(),
        top_data.data<scalar_t>());

  }));

  return {top_data};
}


std::vector<at::Tensor> hard_label_cuda_backward(
    at::Tensor top_diff)
{
  auto grad_prob = at::zeros(top_diff.sizes(), top_diff.options());
  auto grad_label = at::zeros(top_diff.sizes(), top_diff.options());

  return {grad_prob, grad_label};
}
