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

C_LINKAGE cudaError_t gsdrMultiplyCC(
    const cuComplex* in1,
    const cuComplex* in2,
    cuComplex* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  return multiplyGeneric(in1, in2, out, numElements, cudaDevice, cudaStream);
}

C_LINKAGE cudaError_t gsdrMultiplyFF(
    const float* in1,
    const float* in2,
    float* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  return multiplyGeneric(in1, in2, out, numElements, cudaDevice, cudaStream);
}

C_LINKAGE cudaError_t gsdrMultiplyCF(
    const cuComplex* in1,
    const float* in2,
    cuComplex* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  return multiplyGeneric(in1, in2, out, numElements, cudaDevice, cudaStream);
}
