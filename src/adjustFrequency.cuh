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

#ifndef GSDR_SRC_ADJUSTFREQUENCY_CUH_
#define GSDR_SRC_ADJUSTFREQUENCY_CUH_

#include <cuComplex.h>

#include <cstdint>

/**
 * @param firstSampleIndex May be % sampleRate. sampleIndex=0 and sampleIndex=sampleRate give the same output.
 */
__device__ cuComplex k_AdjustFrequency(
    float frequencyShift,
    uint32_t firstSampleIndex,
    float sampleRate,
    const cuComplex* __restrict__ input,
    const float* __restrict__ lowPassTaps,
    uint32_t numLowPassTaps);

#endif  // GSDR_SRC_ADJUSTFREQUENCY_CUH_
