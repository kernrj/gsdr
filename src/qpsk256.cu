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

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "cuComplexOperatorOverloads.cuh"
#include "gsdr/qpsk256.h"
#include "gsdr/util.h"

// Device constant memory for QPSK256 constellation points
__constant__ cuComplex c_qpsk256_rectangular[256];
__constant__ cuComplex c_qpsk256_circular[256];

// Initialize rectangular constellation (16x16 grid)
__global__ void k_InitQpsk256Rectangular(cuComplex* constellation, float amplitude) {
  for (int i = 0; i < 16; ++i) {
    for (int q = 0; q < 16; ++q) {
      int index = i * 16 + q;
      float I = (i - 7.5f) / 7.5f * amplitude;  // Range: -1 to +1
      float Q = (q - 7.5f) / 7.5f * amplitude;  // Range: -1 to +1
      constellation[index] = make_cuComplex(I, Q);
    }
  }
}

// Initialize circular constellation with optimal spacing
__global__ void k_InitQpsk256Circular(cuComplex* constellation, float amplitude) {
  // Use concentric circles with optimal point distribution
  // Based on circle packing theory and optimal constellation design
  const int points_per_circle[8] = {1, 8, 16, 24, 32, 40, 48, 56};  // Total: 225, close to 256
  const float radii[8] = {0.0f, 0.3f, 0.6f, 0.85f, 1.1f, 1.35f, 1.6f, 1.85f};

  int point_index = 0;

  for (int circle = 0; circle < 8 && point_index < 256; ++circle) {
    int points = min(points_per_circle[circle], 256 - point_index);
    float radius = radii[circle] * amplitude;

    for (int p = 0; p < points && point_index < 256; ++p) {
      float angle = 2.0f * M_PIf * p / points + (circle * 0.5f);  // Offset each circle
      float I = radius * cosf(angle);
      float Q = radius * sinf(angle);
      constellation[point_index] = make_cuComplex(I, Q);
      point_index++;
    }
  }

  // Fill remaining points if needed
  while (point_index < 256) {
    float angle = 2.0f * M_PIf * point_index / 256.0f;
    float radius = amplitude * 0.95f;
    float I = radius * cosf(angle);
    float Q = radius * sinf(angle);
    constellation[point_index] = make_cuComplex(I, Q);
    point_index++;
  }
}

// QPSK256 modulation kernel
__global__ void k_Qpsk256Modulate(
    const uint8_t* __restrict__ inputBytes,
    cuComplex* __restrict__ output,
    uint32_t numSymbols,
    float amplitude,
    uint32_t constellationType) {

  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Get the 8-bit symbol value
  uint8_t symbol = inputBytes[symbolIndex];

  // Map to constellation point
  cuComplex point;
  if (constellationType == 0) {
    // Rectangular constellation (16x16 grid)
    point = c_qpsk256_rectangular[symbol];
  } else {
    // Circular constellation
    point = c_qpsk256_circular[symbol];
  }

  output[symbolIndex] = point;
}

// QPSK256 modulation kernel (4 streams)
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
    uint32_t constellationType) {

  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Process 4 streams simultaneously
  uint8_t symbols[4] = {
    inputBytes0[symbolIndex],
    inputBytes1[symbolIndex],
    inputBytes2[symbolIndex],
    inputBytes3[symbolIndex]
  };

  // Map each symbol to constellation point
  cuComplex points[4];
  if (constellationType == 0) {
    // Rectangular constellation
    points[0] = c_qpsk256_rectangular[symbols[0]];
    points[1] = c_qpsk256_rectangular[symbols[1]];
    points[2] = c_qpsk256_rectangular[symbols[2]];
    points[3] = c_qpsk256_rectangular[symbols[3]];
  } else {
    // Circular constellation
    points[0] = c_qpsk256_circular[symbols[0]];
    points[1] = c_qpsk256_circular[symbols[1]];
    points[2] = c_qpsk256_circular[symbols[2]];
    points[3] = c_qpsk256_circular[symbols[3]];
  }

  output0[symbolIndex] = points[0];
  output1[symbolIndex] = points[1];
  output2[symbolIndex] = points[2];
  output3[symbolIndex] = points[3];
}

// QPSK256 demodulation kernel (single stream)
__global__ void k_Qpsk256Demodulate(
    const cuComplex* __restrict__ input,
    uint8_t* __restrict__ outputBytes,
    uint32_t numSymbols,
    uint32_t constellationType) {

  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  cuComplex received = input[symbolIndex];

  // Find the closest constellation point
  float min_distance = INFINITY;
  uint8_t best_symbol = 0;

  if (constellationType == 0) {
    // Search rectangular constellation
    for (int i = 0; i < 256; ++i) {
      cuComplex point = c_qpsk256_rectangular[i];
      float distance = cuCabsf(received - point);
      if (distance < min_distance) {
        min_distance = distance;
        best_symbol = i;
      }
    }
  } else {
    // Search circular constellation
    for (int i = 0; i < 256; ++i) {
      cuComplex point = c_qpsk256_circular[i];
      float distance = cuCabsf(received - point);
      if (distance < min_distance) {
        min_distance = distance;
        best_symbol = i;
      }
    }
  }

  outputBytes[symbolIndex] = best_symbol;
}

// QPSK256 demodulation kernel (4 streams)
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
    uint32_t constellationType) {

  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Process 4 streams simultaneously
  cuComplex received[4] = {
    input0[symbolIndex],
    input1[symbolIndex],
    input2[symbolIndex],
    input3[symbolIndex]
  };

  uint8_t best_symbols[4] = {0, 0, 0, 0};

  for (int stream = 0; stream < 4; ++stream) {
    float min_distance = INFINITY;
    uint8_t best_symbol = 0;

    if (constellationType == 0) {
      // Search rectangular constellation
      for (int i = 0; i < 256; ++i) {
        cuComplex point = c_qpsk256_rectangular[i];
        float distance = cuCabsf(received[stream] - point);
        if (distance < min_distance) {
          min_distance = distance;
          best_symbol = i;
        }
      }
    } else {
      // Search circular constellation
      for (int i = 0; i < 256; ++i) {
        cuComplex point = c_qpsk256_circular[i];
        float distance = cuCabsf(received[stream] - point);
        if (distance < min_distance) {
          min_distance = distance;
          best_symbol = i;
        }
      }
    }

    best_symbols[stream] = best_symbol;
  }

  outputBytes0[symbolIndex] = best_symbols[0];
  outputBytes1[symbolIndex] = best_symbols[1];
  outputBytes2[symbolIndex] = best_symbols[2];
  outputBytes3[symbolIndex] = best_symbols[3];
}

// Initialize constellation points
GSDR_C_LINKAGE cudaError_t gsdrQpsk256InitConstellation(
    uint32_t constellationType,
    float amplitude,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {

  const size_t numElements = 256;
  SIMPLE_CUDA_FNC_START("gsdrQpsk256InitConstellation()")

  // Allocate temporary memory for constellation initialization
  cuComplex* temp_constellation;
  SAFE_CUDA_RET(cudaMalloc(&temp_constellation, 256 * sizeof(cuComplex)));

  // Initialize constellation on GPU
  if (constellationType == 0) {
    // Rectangular constellation
    k_InitQpsk256Rectangular<<<1, 1, 0, cudaStream>>>(temp_constellation, amplitude);
  } else {
    // Circular constellation
    k_InitQpsk256Circular<<<1, 1, 0, cudaStream>>>(temp_constellation, amplitude);
  }

  // Copy to constant memory
  if (constellationType == 0) {
    SAFE_CUDA_RET(cudaMemcpyToSymbol(c_qpsk256_rectangular, temp_constellation, 256 * sizeof(cuComplex)));
  } else {
    SAFE_CUDA_RET(cudaMemcpyToSymbol(c_qpsk256_circular, temp_constellation, 256 * sizeof(cuComplex)));
  }

  // Clean up
  SAFE_CUDA_RET(cudaFree(temp_constellation));

  SIMPLE_CUDA_FNC_END("gsdrQpsk256InitConstellation()")
}

// QPSK256 modulation function
GSDR_C_LINKAGE cudaError_t gsdrQpsk256Modulate(
    const uint8_t* inputBytes,
    cuComplex* output,
    uint32_t numSymbols,
    float amplitude,
    uint32_t constellationType,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {

  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_Qpsk256Modulate()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  k_Qpsk256Modulate<<<blocks, threads, 0, cudaStream>>>(
      inputBytes,
      output,
      numSymbols,
      amplitude,
      constellationType);

  SIMPLE_CUDA_FNC_END("k_Qpsk256Modulate()")
}

// QPSK256 demodulation function
GSDR_C_LINKAGE cudaError_t gsdrQpsk256Demodulate(
    const cuComplex* input,
    uint8_t* outputBytes,
    uint32_t numSymbols,
    uint32_t constellationType,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {

  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_Qpsk256Demodulate()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  k_Qpsk256Demodulate<<<blocks, threads, 0, cudaStream>>>(
      input,
      outputBytes,
      numSymbols,
      constellationType);

  SIMPLE_CUDA_FNC_END("k_Qpsk256Demodulate()")
}

// QPSK256 modulation 4x function
GSDR_C_LINKAGE cudaError_t gsdrQpsk256Modulate4x(
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
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {

  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_Qpsk256Modulate4x()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  k_Qpsk256Modulate4x<<<blocks, threads, 0, cudaStream>>>(
      inputBytes0,
      inputBytes1,
      inputBytes2,
      inputBytes3,
      output0,
      output1,
      output2,
      output3,
      numSymbols,
      amplitude,
      constellationType);

  SIMPLE_CUDA_FNC_END("k_Qpsk256Modulate4x()")
}

// QPSK256 demodulation 4x function
GSDR_C_LINKAGE cudaError_t gsdrQpsk256Demodulate4x(
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
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {

  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_Qpsk256Demodulate4x()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  k_Qpsk256Demodulate4x<<<blocks, threads, 0, cudaStream>>>(
      input0,
      input1,
      input2,
      input3,
      outputBytes0,
      outputBytes1,
      outputBytes2,
      outputBytes3,
      numSymbols,
      constellationType);

  SIMPLE_CUDA_FNC_END("k_Qpsk256Demodulate4x()")
}