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

#ifndef GSDR_INCLUDE_GSDR_QPSK_H_
#define GSDR_INCLUDE_GSDR_QPSK_H_

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gsdr/gsdr_export.h>
#include <gsdr/util.h>
#include <stdint.h>

/**
 * QPSK Modulator - Maps input bits to QPSK constellation points.
 * Each thread processes one symbol (2 bits).
 *
 * @param inputBits Array of input bits (0 or 1), packed as uint8_t
 * @param output Complex output samples
 * @param numSymbols Number of symbols to generate
 * @param amplitude Amplitude scaling factor for the constellation
 */
__global__ void k_QpskModulate(
    const uint8_t* __restrict__ inputBits,
    cuComplex* __restrict__ output,
    uint32_t numSymbols,
    float amplitude);

/**
 * QPSK Modulator (4 streams) - Maps input bits to QPSK constellation points for 4 streams simultaneously.
 * Each thread processes one symbol (2 bits) across 4 streams.
 *
 * @param inputBits0 Array of input bits (0 or 1) for stream 0, packed as uint8_t
 * @param inputBits1 Array of input bits (0 or 1) for stream 1, packed as uint8_t
 * @param inputBits2 Array of input bits (0 or 1) for stream 2, packed as uint8_t
 * @param inputBits3 Array of input bits (0 or 1) for stream 3, packed as uint8_t
 * @param output0 Complex output samples for stream 0
 * @param output1 Complex output samples for stream 1
 * @param output2 Complex output samples for stream 2
 * @param output3 Complex output samples for stream 3
 * @param numSymbols Number of symbols to generate per stream
 * @param amplitude Amplitude scaling factor for the constellation
 */
__global__ void k_QpskModulate4x(
    const uint8_t* __restrict__ inputBits0,
    const uint8_t* __restrict__ inputBits1,
    const uint8_t* __restrict__ inputBits2,
    const uint8_t* __restrict__ inputBits3,
    cuComplex* __restrict__ output0,
    cuComplex* __restrict__ output1,
    cuComplex* __restrict__ output2,
    cuComplex* __restrict__ output3,
    uint32_t numSymbols,
    float amplitude);

/**
 * QPSK Demodulator - Demodulates QPSK symbols to output bits.
 * Each thread processes one symbol.
 *
 * @param input Complex input samples
 * @param outputBits Array to store output bits (0 or 1), packed as uint8_t
 * @param numSymbols Number of symbols to demodulate
 */
__global__ void k_QpskDemodulate(
    const cuComplex* __restrict__ input,
    uint8_t* __restrict__ outputBits,
    uint32_t numSymbols);

/**
 * QPSK Demodulator (4 streams) - Demodulates QPSK symbols to output bits for 4 streams simultaneously.
 * Each thread processes one symbol across 4 streams.
 *
 * @param input0 Complex input samples for stream 0
 * @param input1 Complex input samples for stream 1
 * @param input2 Complex input samples for stream 2
 * @param input3 Complex input samples for stream 3
 * @param outputBits0 Array to store output bits for stream 0
 * @param outputBits1 Array to store output bits for stream 1
 * @param outputBits2 Array to store output bits for stream 2
 * @param outputBits3 Array to store output bits for stream 3
 * @param numSymbols Number of symbols to demodulate per stream
 */
__global__ void k_QpskDemodulate4x(
    const cuComplex* __restrict__ input0,
    const cuComplex* __restrict__ input1,
    const cuComplex* __restrict__ input2,
    const cuComplex* __restrict__ input3,
    uint8_t* __restrict__ outputBits0,
    uint8_t* __restrict__ outputBits1,
    uint8_t* __restrict__ outputBits2,
    uint8_t* __restrict__ outputBits3,
    uint32_t numSymbols);

/**
 * QPSK Modulator - High-level interface for QPSK modulation.
 * Maps input bits to QPSK constellation points.
 *
 * @param inputBits Array of input bits (0 or 1), packed as uint8_t
 * @param output Complex output samples
 * @param numSymbols Number of symbols to generate
 * @param amplitude Amplitude scaling factor for the constellation
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpskModulate(
    const uint8_t* inputBits,
    cuComplex* output,
    uint32_t numSymbols,
    float amplitude,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * QPSK Modulator (4 streams) - High-level interface for QPSK modulation of 4 streams simultaneously.
 * Maps input bits to QPSK constellation points for 4 streams simultaneously.
 *
 * @param inputBits0 Array of input bits (0 or 1) for stream 0, packed as uint8_t
 * @param inputBits1 Array of input bits (0 or 1) for stream 1, packed as uint8_t
 * @param inputBits2 Array of input bits (0 or 1) for stream 2, packed as uint8_t
 * @param inputBits3 Array of input bits (0 or 1) for stream 3, packed as uint8_t
 * @param output0 Complex output samples for stream 0
 * @param output1 Complex output samples for stream 1
 * @param output2 Complex output samples for stream 2
 * @param output3 Complex output samples for stream 3
 * @param numSymbols Number of symbols to generate per stream
 * @param amplitude Amplitude scaling factor for the constellation
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpskModulate4x(
    const uint8_t* inputBits0,
    const uint8_t* inputBits1,
    const uint8_t* inputBits2,
    const uint8_t* inputBits3,
    cuComplex* output0,
    cuComplex* output1,
    cuComplex* output2,
    cuComplex* output3,
    uint32_t numSymbols,
    float amplitude,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * QPSK Demodulator (4 streams) - High-level interface for QPSK demodulation of 4 streams simultaneously.
 * Demodulates QPSK symbols to output bits for 4 streams simultaneously.
 *
 * @param input0 Complex input samples for stream 0
 * @param input1 Complex input samples for stream 1
 * @param input2 Complex input samples for stream 2
 * @param input3 Complex input samples for stream 3
 * @param outputBits0 Array to store output bits for stream 0
 * @param outputBits1 Array to store output bits for stream 1
 * @param outputBits2 Array to store output bits for stream 2
 * @param outputBits3 Array to store output bits for stream 3
 * @param numSymbols Number of symbols to demodulate per stream
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpskDemodulate4x(
    const cuComplex* input0,
    const cuComplex* input1,
    const cuComplex* input2,
    const cuComplex* input3,
    uint8_t* outputBits0,
    uint8_t* outputBits1,
    uint8_t* outputBits2,
    uint8_t* outputBits3,
    uint32_t numSymbols,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * QPSK Demodulator - High-level interface for QPSK demodulation.
 * Demodulates QPSK symbols to output bits.
 *
 * @param input Complex input samples
 * @param outputBits Array to store output bits (0 or 1), packed as uint8_t
 * @param numSymbols Number of symbols to demodulate
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpskDemodulate(
    const cuComplex* input,
    uint8_t* outputBits,
    uint32_t numSymbols,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * Template-based QPSK Modulator - Optimized for different stream counts using templates.
 * Supports 1, 2, 4, and 8 streams with optimized kernels.
 *
 * @param inputBits Array of input bits (0 or 1), packed as uint8_t
 * @param output Complex output samples
 * @param numSymbols Number of symbols to generate
 * @param amplitude Amplitude scaling factor for the constellation
 * @param numStreams Number of streams to process (1, 2, 4, or 8)
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpskModulateTemplated(
    const uint8_t* inputBits,
    cuComplex* output,
    uint32_t numSymbols,
    float amplitude,
    int numStreams,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * Template-based QPSK Demodulator - Optimized for different stream counts using templates.
 * Supports 1, 2, 4, and 8 streams with optimized kernels.
 *
 * @param input Complex input samples
 * @param outputBits Array to store output bits (0 or 1), packed as uint8_t
 * @param numSymbols Number of symbols to demodulate
 * @param numStreams Number of streams to process (1, 2, 4, or 8)
 * @param cudaDevice The CUDA device index to run on
 * @param cudaStream The CUDA stream to use for processing
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQpskDemodulateTemplated(
    const cuComplex* input,
    uint8_t* outputBits,
    uint32_t numSymbols,
    int numStreams,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

#endif  // GSDR_INCLUDE_GSDR_QPSK_H_