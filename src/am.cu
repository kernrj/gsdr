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

#include "adjustFrequency.cuh"
#include "cuComplexOperatorOverloads.cuh"
#include "gsdr/am.h"

__global__ void k_Am(
    const cuComplex* __restrict__ rfInput,
    const float* __restrict__ lowPassTaps,
    uint32_t numTaps,
    uint32_t decimation,
    uint32_t firstSampleIndex,
    float sampleFrequency,
    float frequencyShift,
    float* __restrict__ output,
    uint32_t numOutputs) {
  uint32_t outputIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (outputIndex >= numOutputs) {
    return;
  }

  float& outputSample = output[outputIndex];

  const uint32_t initialInputIndex = decimation * outputIndex;
  const uint32_t firstSampleIndexForThisThread = firstSampleIndex + initialInputIndex;
  cuComplex sample = k_AdjustFrequency(
      frequencyShift,
      firstSampleIndexForThisThread,
      sampleFrequency,
      rfInput + initialInputIndex,
      lowPassTaps,
      numTaps);

  outputSample = __saturatef(hypotf(sample.x, sample.y)) * 2.0f - 1.0f;
}

cudaError_t gsdrAmDemod(
    float rfSampleRate,
    float centerFrequency,
    float channelFrequency,
    uint32_t decimation,
    size_t firstSampleIndex,
    const float* lowPassTaps,
    size_t numLowPassTaps,
    const cuComplex* input,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_Am()")

  const auto firstSampleIndexUInt32 = (uint32_t)fmodf((float)firstSampleIndex, rfSampleRate);
  const float cosineFrequency = centerFrequency - channelFrequency;
  k_Am<<<blocks, threads, 0, cudaStream>>>(
      input,
      lowPassTaps,
      numLowPassTaps,
      decimation,
      firstSampleIndexUInt32,
      rfSampleRate,
      cosineFrequency,
      output,
      numElements);

  SIMPLE_CUDA_FNC_END("k_Am()")
}
