/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
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
