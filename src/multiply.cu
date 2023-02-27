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
#include "gsdr/arithmetic.h"

template <class IN1_T, class IN2_T, class OUT_T>
__global__ void k_Multiply(const IN1_T* in1, const IN2_T* in2, OUT_T* out, size_t numElements) {
  const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x > numElements) {
    return;
  }

  out[x] = in1[x] * in2[x];
}

template <class IN1_T, class IN2_T, class OUT_T>
static cudaError_t multiplyGeneric(
    const IN1_T* in1,
    const IN2_T* in2,
    OUT_T* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  SIMPLE_CUDA_FNC_START("k_Multiply()");
  k_Multiply<IN1_T, IN2_T, OUT_T><<<blocks, threads, 0, cudaStream>>>(in1, in2, out, numElements);
  SIMPLE_CUDA_FNC_END("k_Multiply()");
}

GSDR_C_LINKAGE cudaError_t gsdrMultiplyCC(
    const cuComplex* in1,
    const cuComplex* in2,
    cuComplex* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  return multiplyGeneric(in1, in2, out, numElements, cudaDevice, cudaStream);
}

GSDR_C_LINKAGE cudaError_t gsdrMultiplyFF(
    const float* in1,
    const float* in2,
    float* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  return multiplyGeneric(in1, in2, out, numElements, cudaDevice, cudaStream);
}

GSDR_C_LINKAGE cudaError_t gsdrMultiplyCF(
    const cuComplex* in1,
    const float* in2,
    cuComplex* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  return multiplyGeneric(in1, in2, out, numElements, cudaDevice, cudaStream);
}
