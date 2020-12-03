// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
// This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
// text can be found in LICENSE.md

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <cfloat>
#include <time.h>
#include <thrust/extrema.h>
#include <Eigen/Geometry> 
#include <cublas_v2.h>

#define VERTEX_CHANNELS 3
#define MAX_ROI 128

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


__device__ inline float angle_distance(int cx, int cy, int x, int y, float u, float v)
{
  float dx = cx - x;
  float dy = cy - y;
  float n1 = sqrt(u * u + v * v);
  float n2 = sqrt(dx * dx + dy * dy);
  float dot = u * dx + v * dy;
  float distance = dot / (n1 * n2);

  return distance;
}


__device__ inline float angle_distance_label(int cx, int cy, int x, int y, float u, float v,
  int cls, const int height, const int width, const int* labelmap)
{
  float dx = cx - x;
  float dy = cy - y;
  float n1 = sqrt(u * u + v * v);
  float n2 = sqrt(dx * dx + dy * dy);
  float dot = u * dx + v * dy;
  float distance = dot / (n1 * n2);

  int num = 10;
  int count = 0;
  for (int i = 1; i <= num; i++)
  {
    float step = float(i) / float(num);
    int px = int(x + step * dx);
    int py = int(y + step * dy);
    if (px >= 0 && px < width && py >= 0 && py < height)
    {
      if (labelmap[py * width + px] == cls)
        count++;
    }
  }
  if ((float)count / float(num) < 0.5)
    distance = 0;

  return distance;
}


__device__ inline void project_box(int cls, const float* extents, const float* meta_data, float distance, float factor, float* threshold)
{
  float xHalf = extents[cls * 3 + 0] * 0.5;
  float yHalf = extents[cls * 3 + 1] * 0.5;
  float zHalf = extents[cls * 3 + 2] * 0.5;
  float bb3D[24];

  bb3D[0] = xHalf; bb3D[1] = yHalf; bb3D[2] = zHalf + distance;
  bb3D[3] = -xHalf; bb3D[4] = yHalf; bb3D[5] = zHalf + distance;
  bb3D[6] = xHalf; bb3D[7] = -yHalf; bb3D[8] = zHalf + distance;
  bb3D[9] = -xHalf; bb3D[10] = -yHalf; bb3D[11] = zHalf + distance;
  bb3D[12] = xHalf; bb3D[13] = yHalf; bb3D[14] = -zHalf + distance;
  bb3D[15] = -xHalf; bb3D[16] = yHalf; bb3D[17] = -zHalf + distance;
  bb3D[18] = xHalf; bb3D[19] = -yHalf; bb3D[20] = -zHalf + distance;
  bb3D[21] = -xHalf; bb3D[22] = -yHalf; bb3D[23] = -zHalf + distance;

  float fx = meta_data[0];
  float fy = meta_data[4];
  float px = meta_data[2];
  float py = meta_data[5];
  float minX = 1e8;
  float maxX = -1e8;
  float minY = 1e8;
  float maxY = -1e8;
  for (int i = 0; i < 8; i++)
  {
    float x = fx * (bb3D[i * 3] / bb3D[i * 3 + 2])  + px;
    float y = fy * (bb3D[i * 3 + 1] / bb3D[i * 3 + 2])  + py;
    minX = fmin(minX, x);
    minY = fmin(minY, y);
    maxX = fmax(maxX, x);
    maxY = fmax(maxY, y);
  }
  float width = maxX - minX + 1;
  float height = maxY - minY + 1;
  *threshold = fmax(width, height) * factor;
}


__global__ void compute_arrays_kernel(const int nthreads, const int* labelmap,
    int* arrays, int* array_size, const int height, const int width) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int cls = labelmap[index];
    if (cls > 0)
    {
      int size = atomicAdd(array_size + cls, 1);
      int offset = cls * height * width + size;
      arrays[offset] = index;
    }
  }
}


__global__ void compute_hough_kernel(const int nthreads, float* hough_space, float* hough_data, 
    const int* labelmap, const float* vertmap, const float* extents, const float* meta_data, 
    int* arrays, int* array_size, int* class_indexes, const int height, const int width, 
    const int num_classes, const int count, const float inlierThreshold, const int skip_pixels) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (cls, cx, cy) is an element in the hough space
    int ind = index / (height * width);
    int cls = class_indexes[ind];
    int n = index % (height * width);
    int cx = n % width;
    int cy = n / width;
    int size = array_size[cls];
    float distance = 0;
    float threshold;

    for (int i = 0; i < size; i += skip_pixels)
    {
      int offset = cls * height * width + i;
      int location = arrays[offset];
      int x = location % width;
      int y = location / width;

      // read the direction
      offset = VERTEX_CHANNELS * cls * height * width + y * width + x;
      float u = vertmap[offset];
      offset = VERTEX_CHANNELS * cls * height * width + height * width + y * width + x;
      float v = vertmap[offset];
      offset = VERTEX_CHANNELS * cls * height * width + 2 * height * width + y * width + x;
      float d = exp(vertmap[offset]);

      // vote
      if (angle_distance_label(cx, cy, x, y, u, v, cls, height, width, labelmap) > inlierThreshold)
      {
        project_box(cls, extents, meta_data, d, 0.6, &threshold);
        float dx = fabsf(x - cx);
        float dy = fabsf(y - cy);
        if (dx < threshold && dy < threshold)
        {
          hough_space[index]++;
          distance += d;
        }
      }
    }

    if (hough_space[index] > 0)
    {
      distance /= hough_space[index];

      float bb_width = -1;
      float bb_height = -1;
      for (int i = 0; i < size; i += skip_pixels)
      {
        int offset = cls * height * width + i;
        int location = arrays[offset];
        int x = location % width;
        int y = location / width;

        // read the direction
        offset = VERTEX_CHANNELS * cls * height * width + y * width + x;
        float u = vertmap[offset];
        offset = VERTEX_CHANNELS * cls * height * width + height * width + y * width + x;
        float v = vertmap[offset];

        // vote
        if (angle_distance_label(cx, cy, x, y, u, v, cls, height, width, labelmap) > inlierThreshold)
        {
          project_box(cls, extents, meta_data, distance, 0.6, &threshold);
          float dx = fabsf(x - cx);
          float dy = fabsf(y - cy);
          if (dx > bb_width && dx < threshold && dy < threshold)
            bb_width = dx;
          if (dy > bb_height && dx < threshold && dy < threshold)
            bb_height = dy;
        }
      }

      int offset = ind * height * width * 3 + 3 * (cy * width + cx);
      hough_data[offset] = distance;
      hough_data[offset + 1] = 2 * bb_height;
      hough_data[offset + 2] = 2 * bb_width;
    }
  }
}

__global__ void compute_max_indexes_kernel(const int nthreads, int* max_indexes, int index_size, int* num_max, float* hough_space, 
  float* hough_data, int height, int width, float threshold, float perThreshold, const int is_train)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (ind, cx, cy) is an element in the hough space
    int ind = index / (height * width);
    int n = index % (height * width);
    int cx = n % width;
    int cy = n / width;
    int kernel_size = 3;

    int offset = ind * height * width * 3 + 3 * (cy * width + cx);
    float bb_height = hough_data[offset + 1];
    float bb_width = hough_data[offset + 2];

    if (hough_space[index] > threshold && bb_height > 0 && bb_width > 0)
    {
      // check if the location is local maximum
      int flag = 0;
      for (int x = cx - kernel_size; x <= cx + kernel_size; x++)
      {
        for (int y = cy - kernel_size; y <= cy + kernel_size; y++)
        {
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            if (hough_space[ind * height * width + y * width + x] > hough_space[index])
            {
              flag = 1;
              break;
            }
            if (is_train == 0 && hough_space[ind * height * width + y * width + x] == hough_space[index])
            {
              if (ind * height * width + y * width + x > index)
              {
                flag = 1;
                break;
              }
            }
          }
        }

        // check the percentage of voting
        if (hough_space[index] / (bb_height * bb_width) < perThreshold)
          flag = 1;
      }

      if (flag == 0)
      {
        // add the location to max_indexes
        int max_index = atomicAdd(num_max, 1);
        if (max_index < index_size)
          max_indexes[max_index] = index;
      }
    }
  }
}


__global__ void compute_rois_kernel(const int nthreads, float* top_box, float* top_pose, 
    const float* meta_data, float* hough_space, float* hough_data, int* max_indexes, int* class_indexes,
    int batch_index, const int height, const int width, const int num_classes, int* num_rois, const int is_train) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    float scale = 0.0;
    int max_index = max_indexes[index];
    int ind = max_index / (height * width);
    int cls = class_indexes[ind];
    int n = max_index % (height * width);
    int x = n % width;
    int y = n / width;

    float fx = meta_data[0];
    float fy = meta_data[4];
    float px = meta_data[2];
    float py = meta_data[5];
    float rx = (x - px) / fx;
    float ry = (y - py) / fy;

    int offset = ind * height * width * 3 + 3 * (y * width + x);
    float bb_distance = hough_data[offset];
    float bb_height = hough_data[offset + 1];
    float bb_width = hough_data[offset + 2];

    if (is_train)
    {
      int roi_index = atomicAdd(num_rois, 9);
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x - bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 3] = y - bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 4] = x + bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 5] = y + bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      for (int j = 0; j < 9; j++)
      {
        top_pose[(roi_index + j) * 7 + 0] = 1;
        top_pose[(roi_index + j) * 7 + 1] = 0;
        top_pose[(roi_index + j) * 7 + 2] = 0;
        top_pose[(roi_index + j) * 7 + 3] = 0;
        top_pose[(roi_index + j) * 7 + 4] = rx;
        top_pose[(roi_index + j) * 7 + 5] = ry;
        top_pose[(roi_index + j) * 7 + 6] = bb_distance;
      }

      // add jittering boxes
      float x1 = top_box[roi_index * 7 + 2];
      float y1 = top_box[roi_index * 7 + 3];
      float x2 = top_box[roi_index * 7 + 4];
      float y2 = top_box[roi_index * 7 + 5];
      float ww = x2 - x1;
      float hh = y2 - y1;

      // (-1, -1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 - 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 - 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (+1, -1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 + 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 - 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (-1, +1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 - 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 + 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (+1, +1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 + 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 + 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (0, -1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1;
      top_box[roi_index * 7 + 3] = y1 - 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (-1, 0)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 - 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (0, +1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1;
      top_box[roi_index * 7 + 3] = y1 + 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (+1, 0)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 + 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];
    }
    else
    {
      int roi_index = atomicAdd(num_rois, 1);
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x - bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 3] = y - bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 4] = x + bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 5] = y + bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      top_pose[roi_index * 7 + 0] = 1;
      top_pose[roi_index * 7 + 1] = 0;
      top_pose[roi_index * 7 + 2] = 0;
      top_pose[roi_index * 7 + 3] = 0;
      top_pose[roi_index * 7 + 4] = rx;
      top_pose[roi_index * 7 + 5] = ry;
      top_pose[roi_index * 7 + 6] = bb_distance;
    }
  }
}


std::vector<at::Tensor> hough_voting_cuda_forward(
    at::Tensor bottom_label,
    at::Tensor bottom_vertex,
    at::Tensor bottom_meta_data,
    at::Tensor extents,
    int is_train,
    int skip_pixels,
    int labelThreshold,
    float inlierThreshold,
    float votingThreshold,
    float perThreshold)
{
  const int kThreadsPerBlock = 1024;
  int output_size;
  cudaError_t err;

  const int batch_size = bottom_vertex.size(0);
  const int num_classes = bottom_vertex.size(1) / VERTEX_CHANNELS;
  const int height = bottom_vertex.size(2);
  const int width = bottom_vertex.size(3);
  const int num_meta_data = bottom_meta_data.size(1);
  const int index_size = MAX_ROI / batch_size;

  auto top_box = at::zeros({MAX_ROI * 9, 7}, bottom_vertex.options());
  auto top_pose = at::zeros({MAX_ROI * 9, 7}, bottom_vertex.options());
  auto num_rois = at::zeros({1}, bottom_label.options());

  for (int batch_index = 0; batch_index < batch_size; batch_index++)
  {
    const int* labelmap = bottom_label.data<int>() + batch_index * height * width;
    const float* vertmap = bottom_vertex.data<float>() + batch_index * height * width * VERTEX_CHANNELS * num_classes;
    const float* meta_data = bottom_meta_data.data<float>() + batch_index * num_meta_data;

    // step 1: compute a label index array for each class
    auto arrays = at::zeros({num_classes, height * width}, bottom_label.options());
    auto array_sizes = at::zeros({num_classes}, bottom_label.options());
    output_size = height * width;
    compute_arrays_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, labelmap, arrays.data<int>(), array_sizes.data<int>(), height, width);
    cudaThreadSynchronize();

    // compute class indexes
    int* array_sizes_host = (int*)malloc(num_classes * sizeof(int));
    int* class_indexes_host = (int*)malloc(num_classes * sizeof(int));
    cudaMemcpy(array_sizes_host, array_sizes.data<int>(), num_classes * sizeof(int), cudaMemcpyDeviceToHost);
    int count = 0;
    for (int c = 1; c < num_classes; c++)
    {
      if (array_sizes_host[c] > labelThreshold)
      {
        class_indexes_host[count] = c;
        count++;
      }
      // else
      //  printf("class %d with only pixels %d\n", c, array_sizes_host[c]);
    }

    if (count == 0)
    {
      free(array_sizes_host);
      free(class_indexes_host);
      continue;
    }

    auto class_indexes = at::zeros({count}, bottom_label.options());
    cudaMemcpy(class_indexes.data<int>(), class_indexes_host, count * sizeof(int), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed compute label index: %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }

    // step 2: compute the hough space
    auto hough_space = at::zeros({count, height, width}, bottom_vertex.options());
    auto hough_data = at::zeros({count, height, width, 3}, bottom_vertex.options());

    output_size = count * height * width;
    compute_hough_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, hough_space.data<float>(), hough_data.data<float>(), labelmap, vertmap, extents.data<float>(), meta_data,
      arrays.data<int>(), array_sizes.data<int>(), class_indexes.data<int>(), height, width, num_classes, count, inlierThreshold, skip_pixels);
    cudaThreadSynchronize();

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed compute hough space: %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }

    // step 3: find the maximum in hough space
    auto num_max = at::zeros({1}, bottom_label.options());
    auto max_indexes = at::zeros({index_size}, bottom_label.options());

    if (votingThreshold > 0)
    {
      output_size = count * height * width;
      compute_max_indexes_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size, max_indexes.data<int>(), index_size, num_max.data<int>(), 
        hough_space.data<float>(), hough_data.data<float>(), height, width, votingThreshold, perThreshold, is_train);
      cudaThreadSynchronize();
    }
    else
    {
      int* max_indexes_host = (int*)malloc(count * sizeof(int));
      memset(max_indexes_host, 0, count * sizeof(int));
      for (int i = 0; i < count; i++)
      {
        float *hmax = thrust::max_element(thrust::device, hough_space.data<float>() + i * height * width, 
                                          hough_space.data<float>() + (i+1) * height * width);
        max_indexes_host[i] = hmax - hough_space.data<float>();
      }
      cudaMemcpy(num_max.data<int>(), &count, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(max_indexes.data<int>(), max_indexes_host, count * sizeof(int), cudaMemcpyHostToDevice);
      free(max_indexes_host);
    }

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed compute maximum: %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }

    // step 4: compute outputs
    int num_max_host;
    cudaMemcpy(&num_max_host, num_max.data<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    if (num_max_host >= index_size)
    {
      printf("hough voting num_max: %d exceeds capacity %d\n", num_max_host, index_size);
      num_max_host = index_size;
    }
    if (num_max_host > 0)
    {
      output_size = num_max_host;
      compute_rois_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        output_size, top_box.data<float>(), top_pose.data<float>(), meta_data, hough_space.data<float>(),
        hough_data.data<float>(), max_indexes.data<int>(), class_indexes.data<int>(),
        batch_index, height, width, num_classes, num_rois.data<int>(), is_train);
      cudaThreadSynchronize();
    }
  
    // clean up
    free(array_sizes_host);
    free(class_indexes_host);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed compute outputs: %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }
  }

  // copy outputs
  int num_rois_host;
  cudaMemcpy(&num_rois_host, num_rois.data<int>(), sizeof(int), cudaMemcpyDeviceToHost);
  if (num_rois_host == 0)
    num_rois_host = 1;
  auto top_box_final = at::zeros({num_rois_host, 7}, bottom_vertex.options());
  auto top_pose_final = at::zeros({num_rois_host, 7}, bottom_vertex.options());
  cudaMemcpy(top_box_final.data<float>(), top_box.data<float>(), num_rois_host * 7 * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(top_pose_final.data<float>(), top_pose.data<float>(), num_rois_host * 7 * sizeof(float), cudaMemcpyDeviceToDevice);

  return {top_box_final, top_pose_final};
}
