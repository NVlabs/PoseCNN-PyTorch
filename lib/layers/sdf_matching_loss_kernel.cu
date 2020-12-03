// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
// This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
// text can be found in LICENSE.md

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

__device__ __host__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ __host__ float3 operator-(const float3 &a, const float3 &b)
{
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

template <typename Dtype>
inline __device__ __host__ const Dtype & getValue(const int3 & v, const int3 & dim, const Dtype* sdf_grids)
{
  return sdf_grids[v.x * dim.y * dim.z + v.y * dim.z + v.z];
}

template <typename Dtype>
inline __device__ __host__ Dtype getValueInterpolated(const float3 & pGrid, const int3 & dim, const Dtype* sdf_grids)
{
  const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
  const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
  const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;

  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const int z1 = z0 + 1;

  if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) )
    return 0.1;

  const float dx00 = lerp( getValue(make_int3(x0,y0,z0), dim, sdf_grids), getValue(make_int3(x1,y0,z0), dim, sdf_grids), fx);
  const float dx01 = lerp( getValue(make_int3(x0,y0,z1), dim, sdf_grids), getValue(make_int3(x1,y0,z1), dim, sdf_grids), fx);
  const float dx10 = lerp( getValue(make_int3(x0,y1,z0), dim, sdf_grids), getValue(make_int3(x1,y1,z0), dim, sdf_grids), fx);
  const float dx11 = lerp( getValue(make_int3(x0,y1,z1), dim, sdf_grids), getValue(make_int3(x1,y1,z1), dim, sdf_grids), fx);

  const float dxy0 = lerp( dx00, dx10, fy );
  const float dxy1 = lerp( dx01, dx11, fy );
  float dxyz = lerp( dxy0, dxy1, fz );

  // penalize inside objects
  // if (dxyz < 0)
  //  dxyz *= 10;

  return dxyz;
}

template <typename Dtype>
inline __device__ __host__ float3 getGradientInterpolated(const float3 & pGrid, const int3 & dim, const Dtype* sdf_grids)
{
  const float3 delta_x = make_float3(1,0,0);
  const float3 delta_y = make_float3(0,1,0);
  const float3 delta_z = make_float3(0,0,1);

  Dtype f_px = getValueInterpolated(pGrid + delta_x, dim, sdf_grids);
  Dtype f_py = getValueInterpolated(pGrid + delta_y, dim, sdf_grids);
  Dtype f_pz = getValueInterpolated(pGrid + delta_z, dim, sdf_grids);

  Dtype f_mx = getValueInterpolated(pGrid - delta_x, dim, sdf_grids);
  Dtype f_my = getValueInterpolated(pGrid - delta_y, dim, sdf_grids);
  Dtype f_mz = getValueInterpolated(pGrid - delta_z, dim, sdf_grids);

  float3 grad;
  grad.x = 0.5*(f_px - f_mx);
  grad.y = 0.5*(f_py - f_my);
  grad.z = 0.5*(f_pz - f_mz);
  return grad;
}


/*******************************************/
/* pose_delta: num_objects x 6             */
/* pose_init: num_objects x 4 x 4          */
/* sdf_grid: num_classes x c x h x w       */
/* sdf_limits: num_classes x 9             */
/* points: num_points x 5                  */
/*******************************************/
template <typename Dtype>
__global__ void SDFdistanceForward(const int nthreads, const Dtype* pose_delta, const Dtype* pose_init,
    const Dtype* sdf_grids, const Dtype* sdf_limits, const Dtype* points, 
    const int num_points, Dtype* losses, Dtype* top_values, 
    Dtype* diffs, Dtype* JTJ, Dtype* top_se3) 
{
  typedef Sophus::SE3<Dtype> SE3;
  typedef Eigen::Matrix<Dtype,3,1,Eigen::DontAlign> Vec3;

  // index is the index of point
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int cls_index = int(points[5 * index + 3]);
    int obj_index = int(points[5 * index + 4]);
    int start_index;

    // convert delta pose
    Eigen::Matrix<Dtype,6,1> deltaPose;
    start_index = 6 * obj_index;
    deltaPose << pose_delta[start_index + 0], pose_delta[start_index + 1], pose_delta[start_index + 2], 
                 pose_delta[start_index + 3], pose_delta[start_index + 4], pose_delta[start_index + 5];
    SE3 deltaPoseMatrix = SE3::exp(deltaPose);

    // convert initial pose
    Eigen::Matrix<Dtype,4,4> initialPose;
    start_index = 16 * obj_index;
    initialPose << pose_init[start_index + 0], pose_init[start_index + 1], pose_init[start_index + 2], pose_init[start_index + 3],
                   pose_init[start_index + 4], pose_init[start_index + 5], pose_init[start_index + 6], pose_init[start_index + 7],
                   pose_init[start_index + 8], pose_init[start_index + 9], pose_init[start_index + 10], pose_init[start_index + 11],
                   pose_init[start_index + 12], pose_init[start_index + 13], pose_init[start_index + 14], pose_init[start_index + 15];
    SE3 initialPoseMatrix = SE3(initialPose);

    // start point of a new object
    if (index == 0 || int(points[5 * (index-1) + 4]) != obj_index)
    {
      SE3 pose = deltaPoseMatrix * initialPoseMatrix;
      Eigen::Matrix<Dtype,3,4> matrix = pose.matrix3x4();
      int count = 0;
      start_index = 16 * obj_index;
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 4; j++)
          top_se3[start_index + count++] = matrix(i, j);
      }
      top_se3[start_index + 15] = 1.0;
    }

    // convert point
    Vec3 point;
    point << points[5 * index], points[5 * index + 1], points[5 * index + 2];

    // transform the point
    const Vec3 updatedPoint = deltaPoseMatrix * initialPoseMatrix * point;

    // obtain sdf value
    start_index = 9 * cls_index;
    int d0 = int(sdf_limits[start_index + 6]);
    int d1 = int(sdf_limits[start_index + 7]);
    int d2 = int(sdf_limits[start_index + 8]);
    float px = (updatedPoint(0) - sdf_limits[start_index + 0]) / (sdf_limits[start_index + 3] - sdf_limits[start_index + 0]) * d0;
    float py = (updatedPoint(1) - sdf_limits[start_index + 1]) / (sdf_limits[start_index + 4] - sdf_limits[start_index + 1]) * d1;
    float pz = (updatedPoint(2) - sdf_limits[start_index + 2]) / (sdf_limits[start_index + 5] - sdf_limits[start_index + 2]) * d2;

    float3 pGrid = make_float3(px, py, pz);
    int3 dim = make_int3(d0, d1, d2);
    Dtype value = getValueInterpolated(pGrid, dim, sdf_grids + cls_index * d0 * d1 * d2);

    // L2 loss
    int flag = 1;
    if (value < 0)
      flag = -1;
    value *= flag;

    losses[index] = 0.5 * value * value;
    top_values[index] = losses[index];

    // L2 penalty on translation

    // float lambda = 0.1;
    // losses[index] += 0.5 * lambda * (pose_delta[0] * pose_delta[0] + pose_delta[1] * pose_delta[1] + pose_delta[2] * pose_delta[2]);

    // compute gradient
    float3 grad = getGradientInterpolated(pGrid, dim, sdf_grids + cls_index * d0 * d1 * d2);
    Vec3 sdfUpdate;
    sdfUpdate << grad.x, grad.y, grad.z;

    Eigen::Matrix<Dtype,3,6> dUpdate;
    dUpdate << 1, 0, 0,                     0,  updatedPoint(2), -updatedPoint(1),
               0, 1, 0, -updatedPoint(2),                     0,  updatedPoint(0),
               0, 0, 1,  updatedPoint(1), -updatedPoint(0),                     0;

    Eigen::Matrix<Dtype,1,6> J = flag * sdfUpdate.transpose() * dUpdate;

    // assign gradient
    for (int i = 0; i < 6; i++)
      diffs[6 * index + i] = value * J(i);

    // L2 penalty on translation
    // diffs[6 * index + 0] += lambda * pose_delta[0];
    // diffs[6 * index + 1] += lambda * pose_delta[1];
    // diffs[6 * index + 2] += lambda * pose_delta[2];

    // compute JTJ
    Eigen::Matrix<Dtype,6,6> result = J.transpose() * J;
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
        JTJ[36 * index + i * 6 + j] = result(i, j);
    }
  }
}

/* diffs: num_points x num_channels */
/* bottom_diff: num_objects x num_channels */
template <typename Dtype>
__global__ void sum_gradients(const int nthreads, const Dtype* diffs, const int num_channels, const Dtype* points, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int p = index / num_channels;
    int c = index % num_channels;
    int obj_index = int(points[5 * p + 4]);
    atomicAdd(bottom_diff + obj_index * num_channels + c, diffs[index]);
  }
}


/*******************************************/
/* pose_delta: num_objects x 6             */
/* pose_init: num_objects x 4 x 4          */
/* sdf_grid: num_classes x c x h x w       */
/* sdf_limits: num_classes x 9             */
/* points: num_points x 5                  */
/*******************************************/
std::vector<at::Tensor> sdf_loss_cuda_forward(
    at::Tensor pose_delta,
    at::Tensor pose_init,
    at::Tensor sdf_grids,
    at::Tensor sdf_limits,
    at::Tensor points, 
    at::Tensor regularization) 
{
  // run kernels
  cudaError_t err;
  const int kThreadsPerBlock = 512;
  const int num_channels = 6;
  int output_size;

  // sizes
  const int num_objects = pose_delta.size(0);
  const int num_classes = sdf_grids.size(0);
  const int num_points = points.size(0);

  // temp losses
  auto losses = at::zeros({num_points}, points.options());
  auto top_values = at::zeros({num_points}, points.options());
  auto top_data = at::zeros({1}, points.options());
  auto top_se3 = at::zeros({num_objects, 4, 4}, points.options());

  // temp diffs
  auto diffs = at::zeros({num_points, num_channels}, points.options());
  auto JTJ = at::zeros({num_points, num_channels, num_channels}, points.options());
  auto bottom_diff = at::zeros({num_objects, num_channels}, points.options());
  auto bottom_JTJ = at::zeros({num_objects, num_channels, num_channels}, points.options());

  // compute the losses and gradients
  output_size = num_points;
  SDFdistanceForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, pose_delta.data<float>(), pose_init.data<float>(), sdf_grids.data<float>(), sdf_limits.data<float>(),
      points.data<float>(), num_points, losses.data<float>(), top_values.data<float>(), 
      diffs.data<float>(), JTJ.data<float>(), top_se3.data<float>());
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // sum the diffs
  output_size = num_points * num_channels;
  sum_gradients<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, diffs.data<float>(), num_channels, points.data<float>(), bottom_diff.data<float>());

  output_size = num_points * num_channels * num_channels;
  sum_gradients<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, JTJ.data<float>(), num_channels * num_channels, points.data<float>(), bottom_JTJ.data<float>());
  cudaDeviceSynchronize();

  // sum the loss
  thrust::device_ptr<float> losses_ptr(losses.data<float>());
  float loss = thrust::reduce(losses_ptr, losses_ptr + num_points) / num_points;
  cudaMemcpy(top_data.data<float>(), &loss, sizeof(float), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // compute Gauss Newton update
  float* bottom_diff_host = (float*)malloc(num_objects * num_channels * sizeof(float));
  float* regularization_host = (float*)malloc(num_channels * sizeof(float));
  float* bottom_JTJ_host = (float*)malloc(num_objects * num_channels * num_channels * sizeof(float));
  cudaMemcpy(bottom_diff_host, bottom_diff.data<float>(), num_objects * num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(regularization_host, regularization.data<float>(), num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(bottom_JTJ_host, bottom_JTJ.data<float>(), num_objects * num_channels * num_channels * sizeof(float), cudaMemcpyDeviceToHost);

  Eigen::Matrix<float,6,1> J_eigen;
  Eigen::Matrix<float,6,6> JTJ_eigen;
  float* dalpha_all = (float*)malloc(num_objects * num_channels * sizeof(float));
  for (int k = 0; k < num_objects; k++)
  {
    for (int i = 0; i < num_channels; i++)
    {
      J_eigen(i) = bottom_diff_host[k * num_channels + i];
      for (int j = 0; j < num_channels; j++)
        JTJ_eigen(i, j) = bottom_JTJ_host[k * num_channels * num_channels + i * num_channels + j];
      JTJ_eigen(i, i) += regularization_host[i];
    }
    Eigen::Matrix<float,6,1> dalpha = JTJ_eigen.ldlt().solve(J_eigen);
    for (int i = 0; i < num_channels; i++)
      dalpha_all[k * num_channels + i] = dalpha(i);
  }

  auto bottom_delta = at::zeros({num_objects, num_channels}, points.options());
  cudaMemcpy(bottom_delta.data<float>(), dalpha_all, num_objects * num_channels * sizeof(float), cudaMemcpyHostToDevice);
  free(bottom_diff_host);
  free(regularization_host);
  free(bottom_JTJ_host);
  free(dalpha_all);

  return {top_data, top_values, top_se3, bottom_delta, bottom_diff};
}


template <typename Dtype>
__global__ void SDFdistanceBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_diff, Dtype* output) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    output[index] = top_diff[0] * bottom_diff[index];
  }
}


std::vector<at::Tensor> sdf_loss_cuda_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff)
{
  cudaError_t err;
  const int kThreadsPerBlock = 512;
  int output_size;
  const int batch_size = bottom_diff.size(0);
  const int num_channels = bottom_diff.size(1);
  auto grad_pose = at::zeros({batch_size, num_channels}, bottom_diff.options());

  output_size = batch_size * num_channels;
  SDFdistanceBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, grad_loss.data<float>(), bottom_diff.data<float>(), grad_pose.data<float>());

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return {grad_pose};
}
