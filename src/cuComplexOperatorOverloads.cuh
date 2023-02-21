/*
 * Copyright (C) 2023 Rick Kern <kernrj@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
 * Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Affero General Public License along with this program.  If not, see
 * <https://www.gnu.org/licenses/>.
 */

#ifndef SDRTEST_SRC_CUCOMPLEXOPERATOROVERLOADS_H_
#define SDRTEST_SRC_CUCOMPLEXOPERATOROVERLOADS_H_

#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include "gsdr/cuda_util.h"

__forceinline__ __host__ __device__ cuComplex operator*(const cuComplex c1, const cuComplex c2) {
  return cuCmulf(c1, c2);
}

__forceinline__ __host__ __device__ cuComplex operator*(const cuComplex c, const float r) {
  return make_cuComplex(cuCrealf(c) * r, cuCimagf(c) * r);
}

__forceinline__ __host__ __device__ cuComplex operator*(const float r, const cuComplex c) { return c * r; }

__forceinline__ __host__ __device__ cuComplex operator/(const cuComplex c1, const cuComplex c2) {
  return cuCdivf(c1, c2);
}

__forceinline__ __host__ __device__ cuComplex operator/(const cuComplex c, const float r) {
  return make_cuComplex(cuCrealf(c) / r, cuCimagf(c) / r);
}

__forceinline__ __host__ __device__ cuComplex operator/(const float r, const cuComplex c) {
  return make_cuComplex(r / cuCrealf(c), r / cuCimagf(c));
}

__forceinline__ __host__ __device__ cuComplex operator+(const cuComplex c1, const cuComplex c2) {
  return cuCaddf(c1, c2);
}

__forceinline__ __host__ __device__ cuComplex operator+(const cuComplex c, const float r) {
  return make_cuComplex(cuCrealf(c) + r, cuCimagf(c));
}

__forceinline__ __host__ __device__ cuComplex operator+(const float r, const cuComplex c) { return c + r; }

__forceinline__ __host__ __device__ cuComplex& operator+=(cuComplex& lhs, const cuComplex rhs) {
  lhs.x += rhs.x;
  lhs.y += rhs.y;

  return lhs;
}

template <class T>
__forceinline__ __host__ __device__ T zero() {
  return 0;
}

template <>
__forceinline__ __host__ __device__ cuComplex zero<cuComplex>() {
  return make_cuComplex(0, 0);
}

#define SIMPLE_CUDA_FNC_START(__fncName)                                         \
  const int32_t previousCudaDevice = getCurrentCudaDevice();                     \
  if (previousCudaDevice < 0) {                                                  \
    /* On error, previousCudaDevice is the negative of a cudaError_t */          \
    return (cudaError_t)-previousCudaDevice;                                     \
  }                                                                              \
                                                                                 \
  SAFE_CUDA_RET(cudaSetDevice(cudaDevice));                                      \
                                                                                 \
  const size_t threadsPerWarp = 32;                                              \
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp; \
                                                                                 \
  const dim3 blocks(blockCount);                                                 \
  const dim3 threads(threadsPerWarp);                                            \
  CHECK_CUDA_RET("Before " __fncName);

#define SIMPLE_CUDA_FNC_END(__fncName)              \
  CHECK_CUDA_RET("After " #__fncName);              \
  SAFE_CUDA_RET(cudaSetDevice(previousCudaDevice)); \
  return cudaSuccess;

#endif  // SDRTEST_SRC_CUCOMPLEXOPERATOROVERLOADS_H_
