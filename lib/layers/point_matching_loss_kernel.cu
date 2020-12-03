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

#define POSE_CHANNELS 4

template <typename Dtype>
__global__ void AveragedistanceForward(const int nthreads, const Dtype* prediction, const Dtype* target,
    const Dtype* weight, const Dtype* point, const Dtype* symmetry, const int batch_size, const int num_classes, 
    const int num_points, const float hard_angle, Dtype* rotations, Dtype* losses, Dtype* diffs, Dtype* angles_batch) 
{
  CUDA_1D_KERNEL_LOOP(index_thread, nthreads) 
  {
    // batch index
    int n = index_thread / num_points;
    int p = index_thread % num_points;

    // find the class label and pose of this object
    int index_cls = -1, ind;
    Dtype s, u, v, w;
    for (int i = 0; i < POSE_CHANNELS * num_classes; i += POSE_CHANNELS)
    {
      int index = n * POSE_CHANNELS * num_classes + i;
      if (weight[index] > 0)
      {
        index_cls = i / POSE_CHANNELS;

        // gt quaternion
        s = target[index + 0];
        u = target[index + 1];
        v = target[index + 2];
        w = target[index + 3];

        // gt rotation matrix
        ind = n * num_points * 6 * 9 + p * 6 * 9;
        rotations[ind + 0] = s * s + u * u - v * v - w * w;
        rotations[ind + 1] = 2 * (u * v - s * w);
        rotations[ind + 2] = 2 * (u * w + s * v);
        rotations[ind + 3] = 2 * (u * v + s * w);
        rotations[ind + 4] = s * s - u * u + v * v - w * w;
        rotations[ind + 5] = 2 * (v * w - s * u);
        rotations[ind + 6] = 2 * (u * w - s * v);
        rotations[ind + 7] = 2 * (v * w + s * u);
        rotations[ind + 8] = s * s - u * u - v * v + w * w;

        // predicted quaternion
        s = prediction[index + 0];
        u = prediction[index + 1];
        v = prediction[index + 2];
        w = prediction[index + 3];

        // predicted rotation matrix
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 9;
        rotations[ind + 0] = s * s + u * u - v * v - w * w;
        rotations[ind + 1] = 2 * (u * v - s * w);
        rotations[ind + 2] = 2 * (u * w + s * v);
        rotations[ind + 3] = 2 * (u * v + s * w);
        rotations[ind + 4] = s * s - u * u + v * v - w * w;
        rotations[ind + 5] = 2 * (v * w - s * u);
        rotations[ind + 6] = 2 * (u * w - s * v);
        rotations[ind + 7] = 2 * (v * w + s * u);
        rotations[ind + 8] = s * s - u * u - v * v + w * w;

        // compute the angular distance between quarternions
        if (p == 0)
        {
          Dtype d = target[index + 0] * prediction[index + 0] + target[index + 1] * prediction[index + 1] 
                  + target[index + 2] * prediction[index + 2] + target[index + 3] * prediction[index + 3];
          Dtype angle = acos(2 * d * d - 1) * 180.0 / 3.14159265;
          if (angle > hard_angle)
            angles_batch[n] = 1.0;
        }

        break;
      }
    }
    if (index_cls == -1)
      continue;

    // derivatives of Ru to quaternion
    ind = n * num_points * 6 * 9 + p * 6 * 9 + 18;
    rotations[ind + 0] = 2 * s;
    rotations[ind + 1] = -2 * w;
    rotations[ind + 2] = 2 * v;
    rotations[ind + 3] = 2 * w;
    rotations[ind + 4] = 2 * s;
    rotations[ind + 5] = -2 * u;
    rotations[ind + 6] = -2 * v;
    rotations[ind + 7] = 2 * u;
    rotations[ind + 8] = 2 * s;

    ind = n * num_points * 6 * 9 + p * 6 * 9 + 27;
    rotations[ind + 0] = 2 * u;
    rotations[ind + 1] = 2 * v;
    rotations[ind + 2] = 2 * w;
    rotations[ind + 3] = 2 * v;
    rotations[ind + 4] = -2 * u;
    rotations[ind + 5] = -2 * s;
    rotations[ind + 6] = 2 * w;
    rotations[ind + 7] = 2 * s;
    rotations[ind + 8] = -2 * u;

    ind = n * num_points * 6 * 9 + p * 6 * 9 + 36;
    rotations[ind + 0] = -2 * v;
    rotations[ind + 1] = 2 * u;
    rotations[ind + 2] = 2 * s;
    rotations[ind + 3] = 2 * u;
    rotations[ind + 4] = 2 * v;
    rotations[ind + 5] = 2 * w;
    rotations[ind + 6] = -2 * s;
    rotations[ind + 7] = 2 * w;
    rotations[ind + 8] = -2 * v;

    ind = n * num_points * 6 * 9 + p * 6 * 9 + 45;
    rotations[ind + 0] = -2 * w;
    rotations[ind + 1] = -2 * s;
    rotations[ind + 2] = 2 * u;
    rotations[ind + 3] = 2 * s;
    rotations[ind + 4] = -2 * w;
    rotations[ind + 5] = 2 * v;
    rotations[ind + 6] = 2 * u;
    rotations[ind + 7] = 2 * v;
    rotations[ind + 8] = 2 * w;

    // for the point
    int index = index_cls * num_points * 3 + p * 3;
    ind = n * num_points * 6 * 9 + p * 6 * 9;

    // rotate the first point
    Dtype x1 = rotations[ind + 9 + 0] * point[index + 0] + rotations[ind + 9 + 1] * point[index + 1] + rotations[ind + 9 + 2] * point[index + 2];
    Dtype y1 = rotations[ind + 9 + 3] * point[index + 0] + rotations[ind + 9 + 4] * point[index + 1] + rotations[ind + 9 + 5] * point[index + 2];
    Dtype z1 = rotations[ind + 9 + 6] * point[index + 0] + rotations[ind + 9 + 7] * point[index + 1] + rotations[ind + 9 + 8] * point[index + 2];

    int index_min;
    Dtype x2, y2, z2;
    if (symmetry[index_cls] > 0)
    {
      // find the closet point for symmetry object
      Dtype dmin = FLT_MAX;
      for (int i = 0; i < num_points; i++)
      {
        int index2 = index_cls * num_points * 3 + i * 3;
        x2 = rotations[ind + 0] * point[index2 + 0] + rotations[ind + 1] * point[index2 + 1] + rotations[ind + 2] * point[index2 + 2];
        y2 = rotations[ind + 3] * point[index2 + 0] + rotations[ind + 4] * point[index2 + 1] + rotations[ind + 5] * point[index2 + 2];
        z2 = rotations[ind + 6] * point[index2 + 0] + rotations[ind + 7] * point[index2 + 1] + rotations[ind + 8] * point[index2 + 2];
        Dtype distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
        if (distance < dmin)
        {
          dmin = distance;
          index_min = index2;
        }
      }
    }
    else
      index_min = index;

    x2 = rotations[ind + 0] * point[index_min + 0] + rotations[ind + 1] * point[index_min + 1] + rotations[ind + 2] * point[index_min + 2];
    y2 = rotations[ind + 3] * point[index_min + 0] + rotations[ind + 4] * point[index_min + 1] + rotations[ind + 5] * point[index_min + 2];
    z2 = rotations[ind + 6] * point[index_min + 0] + rotations[ind + 7] * point[index_min + 1] + rotations[ind + 8] * point[index_min + 2];    

    // smooth l1 loss
    Dtype distance = 0;
    int index_diff = n * num_points * POSE_CHANNELS * num_classes + p * POSE_CHANNELS * num_classes + POSE_CHANNELS * index_cls;
    for (int j = 0; j < 3; j++)
    {
      Dtype diff, df;
      if (j == 0)
        diff = x1 - x2;
      else if (j == 1)
        diff = y1 - y2;
      else
        diff = z1 - z2;

      if (fabs(diff) < 1)
      {
        distance += 0.5 * diff * diff;
        df = diff;
      }
      else
      {
        distance += fabs(diff) - 0.5;
        if (diff > 0)
          df = 1.0;
        else
          df = -1.0;
      }

      for (int k = 0; k < 3; k++)
      {
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 18;
        diffs[index_diff + 0] += df * point[index + k] * rotations[ind + j * 3 + k] / num_points;
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 27;
        diffs[index_diff + 1] += df * point[index + k] * rotations[ind + j * 3 + k] / num_points;
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 36;
        diffs[index_diff + 2] += df * point[index + k] * rotations[ind + j * 3 + k] / num_points;
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 45;
        diffs[index_diff + 3] += df * point[index + k] * rotations[ind + j * 3 + k] / num_points;
      }
    }
    losses[index_thread] = distance / num_points;
  }
}



template <typename Dtype>
__global__ void sum_losses_gradients(const int nthreads, const Dtype* losses, const Dtype* diffs, 
    const int num_classes, const int num_points, const float batch_hard, Dtype* angles, Dtype* loss_batch, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int n = index / (POSE_CHANNELS * num_classes);
    int c = index % (POSE_CHANNELS * num_classes);

    bottom_diff[index] = 0;
    if (angles[n] > 0)
    {
      for (int p = 0; p < num_points; p++)
      {
        int index_diff = n * num_points * POSE_CHANNELS * num_classes + p * POSE_CHANNELS * num_classes + c;
        bottom_diff[index] += diffs[index_diff] / batch_hard;
      }
    }

    if (c == 0)
    {
      loss_batch[n] = 0;
      if (angles[n] > 0)
      {
        for (int p = 0; p < num_points; p++)
          loss_batch[n] += losses[n * num_points + p] / batch_hard;
      }
    }
  }
}


std::vector<at::Tensor> pml_cuda_forward(
    at::Tensor bottom_prediction,
    at::Tensor bottom_target,
    at::Tensor bottom_weight,
    at::Tensor points,
    at::Tensor symmetry,
    float hard_angle) 
{
  // run kernels
  cudaError_t err;
  const int kThreadsPerBlock = 512;
  int output_size;

  // temp losses
  const int batch_size = bottom_prediction.size(0);
  const int num_classes = points.size(1);
  const int num_points = points.size(2); 

  auto losses = at::zeros({batch_size, num_points}, points.options());
  auto losses_batch = at::zeros({batch_size}, points.options());
  auto angles_batch = at::zeros({batch_size}, points.options());
  auto top_data = at::zeros({1}, points.options());

  // temp diffs
  auto diffs = at::zeros({batch_size, num_points, POSE_CHANNELS * num_classes}, points.options());
  auto bottom_diff = at::zeros({batch_size, POSE_CHANNELS * num_classes}, points.options());

  // temp rotations
  auto rotations = at::zeros({batch_size, num_points, 6 * 9}, points.options());

  // compute the losses and gradients
  output_size = batch_size * num_points;
  AveragedistanceForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, bottom_prediction.data<float>(), bottom_target.data<float>(), bottom_weight.data<float>(), 
      points.data<float>(), symmetry.data<float>(), 
      batch_size, num_classes, num_points, hard_angle, rotations.data<float>(), losses.data<float>(), diffs.data<float>(), angles_batch.data<float>());
  cudaDeviceSynchronize();

  // sum the angle flags
  thrust::device_ptr<float> angles_ptr(angles_batch.data<float>());
  float batch_hard = thrust::reduce(angles_ptr, angles_ptr + batch_size);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // sum the diffs
  output_size = batch_size * POSE_CHANNELS * num_classes;
  sum_losses_gradients<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, losses.data<float>(), diffs.data<float>(), num_classes, 
      num_points, batch_hard, angles_batch.data<float>(), losses_batch.data<float>(), bottom_diff.data<float>());
  cudaDeviceSynchronize();

  // sum the loss
  thrust::device_ptr<float> losses_ptr(losses_batch.data<float>());
  float loss = thrust::reduce(losses_ptr, losses_ptr + batch_size);
  cudaMemcpy(top_data.data<float>(), &loss, sizeof(float), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return {top_data, bottom_diff};
}


template <typename Dtype>
__global__ void AveragedistanceBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_diff, Dtype* output) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    output[index] = top_diff[0] * bottom_diff[index];
  }
}


std::vector<at::Tensor> pml_cuda_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff)
{
  cudaError_t err;
  const int kThreadsPerBlock = 512;
  int output_size;
  const int batch_size = bottom_diff.size(0);
  const int num_classes = bottom_diff.size(1) / POSE_CHANNELS;
  auto grad_rotation = at::zeros({batch_size, POSE_CHANNELS * num_classes}, bottom_diff.options());

  output_size = batch_size * POSE_CHANNELS * num_classes;
  AveragedistanceBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, grad_loss.data<float>(), bottom_diff.data<float>(), grad_rotation.data<float>());

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return {grad_rotation};
}
