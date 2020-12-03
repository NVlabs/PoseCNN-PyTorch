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
__global__ void BackprojectForward(const int nthreads, const int width, const float fx, 
  const float fy, const float px, const float py, const Dtype* depth, Dtype* top_data)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int x = index % width;
    int y = index / width;
    Dtype d = depth[index];

    top_data[3 * index + 0] = d * (x - px) / fx;
    top_data[3 * index + 1] = d * (y - py) / fy;
    top_data[3 * index + 2] = d;
  }
}


std::vector<at::Tensor> backproject_cuda_forward(
    float fx, float fy, float px, float py,
    at::Tensor depth)
{
  // run kernels
  cudaError_t err;
  const int kThreadsPerBlock = 512;
  int output_size;

  int height = depth.size(0);
  int width = depth.size(1);
  auto top_data = at::zeros({height, width, 3}, depth.options());

  // compute the losses and gradients
  output_size = height * width;
  BackprojectForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, width, fx, fy, px, py, depth.data<float>(), top_data.data<float>());
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return {top_data};
}
