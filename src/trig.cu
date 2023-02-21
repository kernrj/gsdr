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
#include "cuComplexOperatorOverloads.cuh"
#include "gsdr/trig.h"

__global__ void k_ComplexCosine(float indexToRadiansMultiplier, float phi, cuComplex* values, size_t numElements) {
  const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x > numElements) {
    return;
  }

  const float theta = phi + __uint2float_rn(x) * indexToRadiansMultiplier;

  cuComplex result;

  // complex cosine(theta) = cos(theta) + i * sin(theta)
  sincosf(theta, &result.y, &result.x);

  values[x] = result;
}

__global__ void k_RealCosine(float indexToRadiansMultiplier, float phi, float* values, size_t numElements) {
  const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x > numElements) {
    return;
  }

  const float theta = phi + __uint2float_rn(x) * indexToRadiansMultiplier;

  values[x] = cosf(theta);
}

GSDR_C_LINKAGE cudaError_t gsdrCosineC(
    float phiBegin,
    float phiEnd,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_ComplexCosine()")

  const auto indexToRadiansMultiplier = static_cast<float>((phiEnd - phiBegin) / static_cast<double>(numElements));
  k_ComplexCosine<<<blocks, threads, 0, cudaStream>>>(indexToRadiansMultiplier, phiBegin, output, numElements);

  SIMPLE_CUDA_FNC_END("k_ComplexCosine()")
}

GSDR_C_LINKAGE cudaError_t gsdrCosineF(
    float phiBegin,
    float phiEnd,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_RealCosine()")

  const auto indexToRadiansMultiplier = static_cast<float>((phiEnd - phiBegin) / static_cast<double>(numElements));
  k_RealCosine<<<blocks, threads, 0, cudaStream>>>(indexToRadiansMultiplier, phiBegin, output, numElements);

  SIMPLE_CUDA_FNC_END("k_RealCosine()")
}
