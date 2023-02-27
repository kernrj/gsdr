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

#include "adjustFrequency.cuh"
#include "cuComplexOperatorOverloads.cuh"

/**
 * Get the modulus, then normalize to [0, 1): (n % maxVal) / maxVal
 */
__device__ float modNorm(uint32_t n, float maxVal) { return fmodf(__uint2float_rn(n), maxVal) / maxVal; }

__device__ cuComplex k_AdjustFrequency(
    float frequencyShift,
    uint32_t firstSampleIndex,
    float sampleRate,
    const cuComplex* __restrict__ input,
    const float* __restrict__ lowPassTaps,
    uint32_t numLowPassTaps) {
  uint32_t sampleIndex = firstSampleIndex;
  cuComplex sample = zero<cuComplex>();

  const float period = __frcp_rn(frequencyShift);
  for (uint32_t i = 0; i < numLowPassTaps; i++, input++, sampleIndex++) {
    const float timeSeconds = modNorm(sampleIndex, sampleRate);

    // Fraction of one period, in the range [0, 1), is theta/(2π)
    const float thetaDiv2Pi = fmodf(timeSeconds, period);

    // Multiply by 2 to use sincospif() (which only scales by π instead of 2π).
    const float thetaDivPi = scalbnf(thetaDiv2Pi, 1);

    const float tapVal = lowPassTaps[i];
    const cuComplex inVal = *input;

    cuComplex cosVal;

    sincospif(thetaDivPi, &cosVal.y, &cosVal.x);
    const cuComplex multipliedValue = inVal * cosVal;
    const cuComplex filteredValue = multipliedValue * tapVal;

    sample += filteredValue;
  }
}
