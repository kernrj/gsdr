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

#ifndef GSDR_INCLUDE_GSDR_QPSK256_H_
#define GSDR_INCLUDE_GSDR_QPSK256_H_

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gsdr/gsdr_export.h>
#include <gsdr/util.h>
#include <stdint.h>

/**
 * QPSK256 Modulator - Maps 8-bit input bytes to QPSK256 constellation points.
 * Each thread processes one symbol (8 bits).
 *
 * @param inputBytes Array of input bytes (0-255), each representing one symbol
 * @param output Complex output samples
 * @param numSymbols Number of symbols to generate
 * @param amplitude Amplitude scaling factor for the constellation
 * @param constellationType Type of constellation (0=rectangular, 1=circular)
 */
__global__ void k_Qpsk256Modulate(
    const uint8_t* __restrict__ inputBytes,
    cuComplex* __restrict__ output,
    uint32_t numSymbols,
    float amplitude,
    uint32_t constellationType);

/**
 * QPSK256 Modulator (4 streams) - Maps 8-bit input bytes to QPSK256 constellation points for 4 streams simultaneously.
 * Each thread processes one symbol across 4 streams.
 *
 * @param inputBytes0 Array of input bytes for stream 0
 * @param inputBytes1 Array of input bytes for stream 1
 * @param inputBytes2 Array of input bytes for stream 2
 * @param inputBytes3 Array of input bytes for stream 3
 * @param output0 Complex output samples for stream 0
 * @param output1 Complex output samples for stream 1
 * @param output2 Complex output samples for stream 2
 * @param output3 Complex output samples for stream 3
 * @param numSymbols Number of symbols to generate per stream
 * @param amplitude Amplitude scaling factor for the constellation
 * @param constellationType Type of constellation (0=rectangular, 1=circular)
 */
__global__ void k_Qpsk256Modulate4x(
    const uint8_t* __restrict__ inputBytes0,
    const uint8_t* __restrict__ inputBytes1,
    const uint8_t* __restrict__ inputBytes2,
    const uint8_t* __restrict__ inputBytes3,
    cuComplex* __restrict__ output0,
    cuComplex* __restrict__ output1,
    cuComplex* __restrict__ output2,
    cuComplex* __restrict__ output3,
    uint32_t numSymbols,
    float amplitude,
    uint32_t constellationType);

/**
 * QPSK256 Demodulator - Demodulates QPSK256 symbols to output bytes.
 * Each thread processes one symbol.
 *
 * @param input Complex input samples
 * @param outputBytes Array to store output bytes (0-255)
 * @param numSymbols Number of symbols to demodulate
 * @param constellationType Type of constellation (0=rectangular, 1=circular)
 */
__global__ void k_Qpsk256Demodulate(
    const cuComplex* __restrict__ input,
    uint8_t* __restrict__ outputBytes,
    uint32_t numSymbols,
    uint32_t constellationType);

/**
 * QPSK256 Demodulator (4 streams) - Demodulates QPSK256 symbols to output bytes for 4 streams simultaneously.
 * Each thread processes one symbol across 4 streams.
 *
 * @param input0 Complex input samples for stream 0
 * @param input1 Complex input samples for stream 1
 * @param input2 Complex input samples for stream 2
 * @param input3 Complex input samples for stream 3
 * @param outputBytes0 Array to store output bytes for stream 0
 * @param outputBytes1 Array to store output bytes for stream 1
 * @param outputBytes2 Array to store output bytes for stream 2
 * @param outputBytes3 Array to store output bytes for stream 3
 * @param numSymbols Number of symbols to demodulate per stream
 * @param constellationType Type of constellation (0=rectangular, 1=circular)
 */
__global__ void k_Qpsk256Demodulate4x(
    const cuComplex* __restrict__ input0,
    const cuComplex* __restrict__ input1,
    const cuComplex* __restrict__ input2,
    const cuComplex* __restrict__ input3,
    uint8_t* __restrict__ outputBytes0,
    uint8_t* __restrict__ outputBytes1,
    uint8_t* __restrict__ outputBytes2,
    uint8_t* __restrict__ outputBytes3,
    uint32_t numSymbols,
    uint32_t constellationType);

/**
 * QPSK256 Modulator - High-level interface for QPSK256 modulation.
 * Maps 8-bit input bytes to QPSK256 constellation points.
 *
 * @param inputBytes Array of input bytes (0-255), each representing one symbol
 * @param output Complex output samples
 * @param numSymbols Number of symbols to generate
 * @param amplitude Amplitude scaling factor for the constellation
 * @param constellationType Type of constellation (0=rectangular, 1=circular)
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpsk256Modulate(
    const uint8_t* inputBytes,
    cuComplex* output,
    uint32_t numSymbols,
    float amplitude,
    uint32_t constellationType,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * QPSK256 Demodulator - High-level interface for QPSK256 demodulation.
 * Demodulates QPSK256 symbols to output bytes.
 *
 * @param input Complex input samples
 * @param outputBytes Array to store output bytes (0-255)
 * @param numSymbols Number of symbols to demodulate
 * @param constellationType Type of constellation (0=rectangular, 1=circular)
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpsk256Demodulate(
    const cuComplex* input,
    uint8_t* outputBytes,
    uint32_t numSymbols,
    uint32_t constellationType,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * QPSK256 Modulator (4 streams) - High-level interface for QPSK256 modulation of 4 streams simultaneously.
 * Maps 8-bit input bytes to QPSK256 constellation points for 4 streams simultaneously.
 *
 * @param inputBytes0 Array of input bytes for stream 0
 * @param inputBytes1 Array of input bytes for stream 1
 * @param inputBytes2 Array of input bytes for stream 2
 * @param inputBytes3 Array of input bytes for stream 3
 * @param output0 Complex output samples for stream 0
 * @param output1 Complex output samples for stream 1
 * @param output2 Complex output samples for stream 2
 * @param output3 Complex output samples for stream 3
 * @param numSymbols Number of symbols to generate per stream
 * @param amplitude Amplitude scaling factor for the constellation
 * @param constellationType Type of constellation (0=rectangular, 1=circular)
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpsk256Modulate4x(
    const uint8_t* inputBytes0,
    const uint8_t* inputBytes1,
    const uint8_t* inputBytes2,
    const uint8_t* inputBytes3,
    cuComplex* output0,
    cuComplex* output1,
    cuComplex* output2,
    cuComplex* output3,
    uint32_t numSymbols,
    float amplitude,
    uint32_t constellationType,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * QPSK256 Demodulator (4 streams) - High-level interface for QPSK256 demodulation of 4 streams simultaneously.
 * Demodulates QPSK256 symbols to output bytes for 4 streams simultaneously.
 *
 * @param input0 Complex input samples for stream 0
 * @param input1 Complex input samples for stream 1
 * @param input2 Complex input samples for stream 2
 * @param input3 Complex input samples for stream 3
 * @param outputBytes0 Array to store output bytes for stream 0
 * @param outputBytes1 Array to store output bytes for stream 1
 * @param outputBytes2 Array to store output bytes for stream 2
 * @param outputBytes3 Array to store output bytes for stream 3
 * @param numSymbols Number of symbols to demodulate per stream
 * @param constellationType Type of constellation (0=rectangular, 1=circular)
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpsk256Demodulate4x(
    const cuComplex* input0,
    const cuComplex* input1,
    const cuComplex* input2,
    const cuComplex* input3,
    uint8_t* outputBytes0,
    uint8_t* outputBytes1,
    uint8_t* outputBytes2,
    uint8_t* outputBytes3,
    uint32_t numSymbols,
    uint32_t constellationType,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * QPSK256 Constellation Initialization - Initializes the constellation points in device constant memory.
 * Must be called before using QPSK256 modulation/demodulation functions.
 *
 * @param constellationType Type of constellation (0=rectangular, 1=circular)
 * @param amplitude Amplitude scaling factor for the constellation
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpsk256InitConstellation(
    uint32_t constellationType,
    float amplitude,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

#endif  // GSDR_INCLUDE_GSDR_QPSK256_H_