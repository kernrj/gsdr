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

template <class IN1_T, class CONST_T, class OUT_T>
__global__ void k_AddConst(const IN1_T* input, CONST_T addConst, OUT_T* output, size_t numElements) {
  const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x > numElements) {
    return;
  }

  output[x] = addConst + input[x];
}

__global__ void k_AddToMagnitude(const cuComplex* in, float addToMagnitude, cuComplex* out, size_t numElements) {
  const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x > numElements) {
    return;
  }

  const cuComplex inputValue = in[x];
  const float oldLength = hypotf(inputValue.x, inputValue.y);
  const cuComplex normVec = inputValue / oldLength;
  const float newLength = addToMagnitude + oldLength;

  out[x] = newLength * normVec;
}

template <class IN1_T, class CONST_T, class OUT_T>
static cudaError_t addConstGeneric(
    const IN1_T* input,
    CONST_T addConst,
    OUT_T* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_AddConst()");
  k_AddConst<IN1_T, CONST_T, OUT_T><<<blocks, threads, 0, cudaStream>>>(input, addConst, output, numElements);
  SIMPLE_CUDA_FNC_END("k_AddConst()");
}

GSDR_C_LINKAGE cudaError_t gsdrAddConstFF(
    const float* input,
    float addConst,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  return addConstGeneric(input, addConst, output, numElements, cudaDevice, cudaStream);
}

GSDR_C_LINKAGE cudaError_t gsdrAddConstCC(
    const cuComplex* input,
    cuComplex addConst,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  return addConstGeneric(input, addConst, output, numElements, cudaDevice, cudaStream);
}

GSDR_C_LINKAGE cudaError_t gsdrAddConstCF(
    const cuComplex* input,
    float addConst,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  return addConstGeneric(input, addConst, output, numElements, cudaDevice, cudaStream);
}

GSDR_C_LINKAGE cudaError_t gsdrAddConstFC(
    const float* input,
    cuComplex addConst,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  return addConstGeneric(input, addConst, output, numElements, cudaDevice, cudaStream);
}

GSDR_C_LINKAGE cudaError_t gsdrAddToMagnitude(
    const cuComplex* input,
    float addToMagnitude,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_AddToMagnitude()");
  k_AddToMagnitude<<<blocks, threads, 0, cudaStream>>>(input, addToMagnitude, output, numElements);
  SIMPLE_CUDA_FNC_END("k_AddToMagnitude()");
}
