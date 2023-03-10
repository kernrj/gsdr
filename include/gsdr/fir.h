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

#ifndef SDRTEST_SRC_FIR_H_
#define SDRTEST_SRC_FIR_H_

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gsdr/gsdr_export.h>
#include <gsdr/util.h>
#include <stdint.h>

/**
 * Runs the FIR filter for the given taps, and returns the number of elements written to output.
 * input, output, and taps must all be in GPU memory.
 * @return
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrFirFC(
    size_t decimation,
    const float* taps,
    size_t tapCount,
    const cuComplex* input,
    cuComplex* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrFirFF(
    size_t decimation,
    const float* taps,
    size_t tapCount,
    const float* input,
    float* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrFirCC(
    size_t decimation,
    const cuComplex* taps,
    size_t tapCount,
    const cuComplex* input,
    cuComplex* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrFirCF(
    size_t decimation,
    const cuComplex* taps,
    size_t tapCount,
    const float* input,
    cuComplex* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

#endif  // SDRTEST_SRC_FIR_H_
