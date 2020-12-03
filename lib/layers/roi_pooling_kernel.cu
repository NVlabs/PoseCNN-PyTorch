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

__global__ void ROIPoolForward(const int nthreads, const float* bottom_data, const float spatial_scale, const int height, const int width,
                               const int channels, const int channel_rois, const int pool_height, const int pool_width, 
                               const float* bottom_rois, float* top_data, float* argmax_data) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, c, ph, pw) is an element in the pool output
    // int n = index;
    // int pw = n % pool_width;
    // n /= pool_width;
    // int ph = n % pool_height;
    // n /= pool_height;
    // int c = n % channels;
    // n /= channels;

    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pool_width;
    int ph = (index / pool_width) % pool_height;
    int c  = (index / pool_width / pool_height) % channels;
    int n  = index / pool_width / pool_height / channels;

    float roi_batch_ind = bottom_rois[n * channel_rois + 0];
    float roi_start_w = bottom_rois[n * channel_rois + 2] * spatial_scale;
    float roi_start_h = bottom_rois[n * channel_rois + 3] * spatial_scale;
    float roi_end_w = bottom_rois[n * channel_rois + 4] * spatial_scale;
    float roi_end_h = bottom_rois[n * channel_rois + 5] * spatial_scale;

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1.0);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1.0);
    float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pool_height);
    float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pool_width);

    int hstart = static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(int(hstart + roi_start_h), 0), height);
    hend = min(max(int(hend + roi_start_h), 0), height);
    wstart = min(max(int(wstart + roi_start_w), 0), width);
    wend = min(max(int(wend + roi_start_w), 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    float maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    int offset = roi_batch_ind * channels * height * width;
    const float* offset_bottom_data = bottom_data + offset;
    for (int h = hstart; h < hend; ++h) 
    {
      for (int w = wstart; w < wend; ++w) 
      {
        int bottom_index = c * height * width + h * width + w;
        if (offset_bottom_data[bottom_index] > maxval) {
          maxval = offset_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}


std::vector<at::Tensor> roi_pool_cuda_forward(
    int pool_height,
    int pool_width,
    float spatial_scale,
    at::Tensor bottom_features,
    at::Tensor bottom_rois) 
{
  // run kernels
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  const int batch_size = bottom_features.size(0);
  const int num_channels = bottom_features.size(1);
  const int height = bottom_features.size(2);
  const int width = bottom_features.size(3);
  const int num_rois = bottom_rois.size(0);
  const int channel_rois = bottom_rois.size(1);

  auto top_data = at::zeros({num_rois, num_channels, pool_height, pool_width}, bottom_features.options());
  auto top_argmax = at::zeros({num_rois, num_channels, pool_height, pool_width}, bottom_features.options());
  const int output_size = num_rois * num_channels * pool_height * pool_width;

  ROIPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
    output_size, bottom_features.data<float>(), spatial_scale, height, width, num_channels, channel_rois,
    pool_height, pool_width, bottom_rois.data<float>(), top_data.data<float>(), top_argmax.data<float>());

  err = cudaGetLastError();
  if(cudaSuccess != err) 
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return {top_data, top_argmax};
}


__global__ void ROIPoolBackward(const int nthreads, const float* top_diff, const float spatial_scale, const int height, const int width,
                                const int num_rois, const int channels, const int channel_rois, 
                                const int pool_height, const int pool_width, float* bottom_diff, 
                                const float* bottom_rois, const float* argmax_data) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c  = (index / width / height) % channels;
    int n  = index / width / height / channels;

    float gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; roi_n++)
    {
      const float* offset_bottom_rois = bottom_rois + roi_n * channel_rois;
      int roi_batch_ind = int(offset_bottom_rois[0]);
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) 
        continue;

      int roi_start_w = round(offset_bottom_rois[2] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[4] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[5] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi)
        continue;

      int offset = roi_n * channels * pool_height * pool_width;
      const float* offset_top_diff = top_diff + offset;
      const float* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pool_height);
      float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pool_width);

      int phstart = floor(static_cast<float>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<float>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<float>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<float>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pool_height);
      phend = min(max(phend, 0), pool_height);
      pwstart = min(max(pwstart, 0), pool_width);
      pwend = min(max(pwend, 0), pool_width);

      for (int ph = phstart; ph < phend; ++ph) 
      {
        for (int pw = pwstart; pw < pwend; ++pw) 
        {
          if (int(offset_argmax_data[c * pool_height * pool_width + ph * pool_width + pw]) == c * height * width + h * width + w) 
            gradient += offset_top_diff[c * pool_height * pool_width + ph * pool_width + pw];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}


std::vector<at::Tensor> roi_pool_cuda_backward(
    int batch_size,
    int height,
    int width,
    float spatial_scale,
    at::Tensor top_diff,
    at::Tensor bottom_rois,
    at::Tensor argmax_data)
{
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  const int num_rois = top_diff.size(0);
  const int num_channels = top_diff.size(1);
  const int pool_height = top_diff.size(2);
  const int pool_width = top_diff.size(3);
  const int channel_rois = bottom_rois.size(1);

  auto bottom_diff = at::zeros({batch_size, num_channels, height, width}, top_diff.options());
  const int output_size = batch_size * num_channels * height * width;

  ROIPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
    output_size, top_diff.data<float>(), spatial_scale, height, width, num_rois, num_channels, channel_rois,
    pool_height, pool_width, bottom_diff.data<float>(), bottom_rois.data<float>(), argmax_data.data<float>());

  err = cudaGetLastError();
  if(cudaSuccess != err) 
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit(-1);
  }

  return {bottom_diff};
}
