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

#ifndef GPUSDR_IIR_H
#define GPUSDR_IIR_H

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gsdr/gsdr_export.h>
#include <gsdr/util.h>
#include <stdint.h>

/**
 * Runs the IIR filter with the given feedforward (b) and feedback (a) coefficients.
 * input and output must be in GPU memory.
 * The filter uses a direct form II structure.
 * @param bCoeffs - feedforward coefficients (b0, b1, b2, ...)
 * @param aCoeffs - feedback coefficients (1.0, -a1, -a2, ...) - note a0 is always 1.0
 * @param coeffCount - number of coefficients (should be the same for b and a)
 * @param inputHistory - input history buffer (must be pre-allocated with size coeffCount-1)
 * @param outputHistory - output history buffer (must be pre-allocated with size coeffCount-1)
 * @return
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrIirFF(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    float* inputHistory,
    float* outputHistory,
    const float* input,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrIirCC(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    cuComplex* inputHistory,
    cuComplex* outputHistory,
    const cuComplex* input,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

#endif  // GPUSDR_IIR_H