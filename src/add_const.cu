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
    cudaStream_t cudaStream) {
  SIMPLE_CUDA_FNC_START("k_AddConst()");
  k_AddConst<IN1_T, CONST_T, OUT_T><<<blocks, threads, 0, cudaStream>>>(input, addConst, output, numElements);
  SIMPLE_CUDA_FNC_END("k_AddConst()");
}

C_LINKAGE cudaError_t gsdrAddConstFF(
    const float* input,
    float addConst,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  return addConstGeneric(input, addConst, output, numElements, cudaDevice, cudaStream);
}

C_LINKAGE cudaError_t gsdrAddConstCC(
    const cuComplex* input,
    cuComplex addConst,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  return addConstGeneric(input, addConst, output, numElements, cudaDevice, cudaStream);
}

C_LINKAGE cudaError_t gsdrAddConstCF(
    const cuComplex* input,
    float addConst,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  return addConstGeneric(input, addConst, output, numElements, cudaDevice, cudaStream);
}

C_LINKAGE cudaError_t gsdrAddConstFC(
    const float* input,
    cuComplex addConst,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  return addConstGeneric(input, addConst, output, numElements, cudaDevice, cudaStream);
}

C_LINKAGE cudaError_t gsdrAddToMagnitude(
    const cuComplex* input,
    float addToMagnitude,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  SIMPLE_CUDA_FNC_START("k_AddToMagnitude()");
  k_AddToMagnitude<<<blocks, threads, 0, cudaStream>>>(input, addToMagnitude, output, numElements);
  SIMPLE_CUDA_FNC_END("k_AddToMagnitude()");
}
