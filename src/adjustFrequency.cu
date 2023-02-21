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
