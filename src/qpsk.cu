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
#include "gsdr/qpsk.h"
#include "gsdr/util.h"

// Template for QPSK modulation with variable number of streams
template <int NUM_STREAMS>
__global__ void k_QpskModulateTemplated(const uint8_t* __restrict__ inputBits, cuComplex* __restrict__ output, uint32_t numSymbols, float amplitude) {
  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Each symbol is 2 bits, so we need to read from inputBits[symbolIndex / 4]
  const uint32_t byteIndex = symbolIndex / 4;
  const uint32_t bitOffset = (symbolIndex % 4) * 2;

  // Process multiple streams
  for (int streamIdx = 0; streamIdx < NUM_STREAMS; ++streamIdx) {
    const uint8_t* currentInputBits = inputBits + streamIdx * (numSymbols / 4 + 1);  // Each stream has its own input array
    const uint8_t bits = (currentInputBits[byteIndex] >> bitOffset) & 0x3;

    // Map bits to QPSK constellation
    cuComplex symbol;
    switch (bits) {
      case 0: symbol = make_cuComplex(amplitude, amplitude); break;
      case 1: symbol = make_cuComplex(-amplitude, amplitude); break;
      case 3: symbol = make_cuComplex(-amplitude, -amplitude); break;
      case 2: symbol = make_cuComplex(amplitude, -amplitude); break;
      default: symbol = make_cuComplex(0.0f, 0.0f); break;
    }

    output[streamIdx * numSymbols + symbolIndex] = symbol;
  }
}

// Template for QPSK demodulation with variable number of streams
template <int NUM_STREAMS>
__global__ void k_QpskDemodulateTemplated(const cuComplex* __restrict__ input, uint8_t* __restrict__ outputBits, uint32_t numSymbols) {
  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Process multiple streams
  for (int streamIdx = 0; streamIdx < NUM_STREAMS; ++streamIdx) {
    const cuComplex symbol = input[streamIdx * numSymbols + symbolIndex];

    // QPSK demodulation: determine quadrant based on sign of real and imaginary parts
    uint8_t bits = 0;
    if (symbol.x >= 0.0f) {
      bits = (symbol.y >= 0.0f) ? 0x0 : 0x2;
    } else {
      bits = (symbol.y >= 0.0f) ? 0x1 : 0x3;
    }

    // Pack bits into output array (4 symbols per byte)
    uint8_t* currentOutputBits = outputBits + streamIdx * (numSymbols / 4 + 1);
    const uint32_t byteIndex = symbolIndex / 4;
    const uint32_t bitOffset = (symbolIndex % 4) * 2;
    const uint8_t mask = ~(0x3 << bitOffset);
    currentOutputBits[byteIndex] = (currentOutputBits[byteIndex] & mask) | (bits << bitOffset);
  }
}

// QPSK constellation mapping:
// 00 -> +1 + 1j (45°)
// 01 -> -1 + 1j (135°)
// 11 -> -1 - 1j (225° or -135°)
// 10 -> +1 - 1j (315° or -45°)

__global__ void k_QpskModulate(
    const uint8_t* __restrict__ inputBits,
    cuComplex* __restrict__ output,
    uint32_t numSymbols,
    float amplitude) {
  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Each symbol is 2 bits, so we need to read from inputBits[symbolIndex / 4]
  // and extract the appropriate 2 bits
  const uint32_t byteIndex = symbolIndex / 4;
  const uint32_t bitOffset = (symbolIndex % 4) * 2;
  const uint8_t bits = (inputBits[byteIndex] >> bitOffset) & 0x3;

  // Map bits to QPSK constellation
  cuComplex symbol;
  switch (bits) {
    case 0:  // 00 -> +1 + 1j
      symbol = make_cuComplex(amplitude, amplitude);
      break;
    case 1:  // 01 -> -1 + 1j
      symbol = make_cuComplex(-amplitude, amplitude);
      break;
    case 3:  // 11 -> -1 - 1j
      symbol = make_cuComplex(-amplitude, -amplitude);
      break;
    case 2:  // 10 -> +1 - 1j
      symbol = make_cuComplex(amplitude, -amplitude);
      break;
    default:
      symbol = make_cuComplex(0.0f, 0.0f);
      break;
  }

  output[symbolIndex] = symbol;
}

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
    float amplitude) {
  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Process 4 streams simultaneously
  // Each symbol is 2 bits, so we need to read from inputBits[symbolIndex / 4]
  const uint32_t byteIndex = symbolIndex / 4;
  const uint32_t bitOffset = (symbolIndex % 4) * 2;

  // Extract bits for each stream
  const uint8_t bits0 = (inputBits0[byteIndex] >> bitOffset) & 0x3;
  const uint8_t bits1 = (inputBits1[byteIndex] >> bitOffset) & 0x3;
  const uint8_t bits2 = (inputBits2[byteIndex] >> bitOffset) & 0x3;
  const uint8_t bits3 = (inputBits3[byteIndex] >> bitOffset) & 0x3;

  // Map bits to QPSK constellation for each stream
  cuComplex symbol0, symbol1, symbol2, symbol3;

  // Stream 0
  switch (bits0) {
    case 0: symbol0 = make_cuComplex(amplitude, amplitude); break;
    case 1: symbol0 = make_cuComplex(-amplitude, amplitude); break;
    case 3: symbol0 = make_cuComplex(-amplitude, -amplitude); break;
    case 2: symbol0 = make_cuComplex(amplitude, -amplitude); break;
    default: symbol0 = make_cuComplex(0.0f, 0.0f); break;
  }

  // Stream 1
  switch (bits1) {
    case 0: symbol1 = make_cuComplex(amplitude, amplitude); break;
    case 1: symbol1 = make_cuComplex(-amplitude, amplitude); break;
    case 3: symbol1 = make_cuComplex(-amplitude, -amplitude); break;
    case 2: symbol1 = make_cuComplex(amplitude, -amplitude); break;
    default: symbol1 = make_cuComplex(0.0f, 0.0f); break;
  }

  // Stream 2
  switch (bits2) {
    case 0: symbol2 = make_cuComplex(amplitude, amplitude); break;
    case 1: symbol2 = make_cuComplex(-amplitude, amplitude); break;
    case 3: symbol2 = make_cuComplex(-amplitude, -amplitude); break;
    case 2: symbol2 = make_cuComplex(amplitude, -amplitude); break;
    default: symbol2 = make_cuComplex(0.0f, 0.0f); break;
  }

  // Stream 3
  switch (bits3) {
    case 0: symbol3 = make_cuComplex(amplitude, amplitude); break;
    case 1: symbol3 = make_cuComplex(-amplitude, amplitude); break;
    case 3: symbol3 = make_cuComplex(-amplitude, -amplitude); break;
    case 2: symbol3 = make_cuComplex(amplitude, -amplitude); break;
    default: symbol3 = make_cuComplex(0.0f, 0.0f); break;
  }

  output0[symbolIndex] = symbol0;
  output1[symbolIndex] = symbol1;
  output2[symbolIndex] = symbol2;
  output3[symbolIndex] = symbol3;
}

__global__ void k_QpskDemodulate(
    const cuComplex* __restrict__ input,
    uint8_t* __restrict__ outputBits,
    uint32_t numSymbols) {
  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  const cuComplex symbol = input[symbolIndex];

  // QPSK demodulation: determine quadrant based on sign of real and imaginary parts
  // 00 -> +1 + 1j (45°) -> bits 00
  // 01 -> -1 + 1j (135°) -> bits 01
  // 11 -> -1 - 1j (225°) -> bits 11
  // 10 -> +1 - 1j (315°) -> bits 10

  uint8_t bits = 0;
  if (symbol.x >= 0.0f) {
    // Right half-plane: could be 00 or 10
    if (symbol.y >= 0.0f) {
      bits = 0x0;  // 00
    } else {
      bits = 0x2;  // 10
    }
  } else {
    // Left half-plane: could be 01 or 11
    if (symbol.y >= 0.0f) {
      bits = 0x1;  // 01
    } else {
      bits = 0x3;  // 11
    }
  }

  // Pack bits into output array (4 symbols per byte)
  const uint32_t byteIndex = symbolIndex / 4;
  const uint32_t bitOffset = (symbolIndex % 4) * 2;
  const uint8_t mask = ~(0x3 << bitOffset);  // Clear the 2 bits
  outputBits[byteIndex] = (outputBits[byteIndex] & mask) | (bits << bitOffset);
}

__global__ void k_QpskDemodulate4x(
    const cuComplex* __restrict__ input0,
    const cuComplex* __restrict__ input1,
    const cuComplex* __restrict__ input2,
    const cuComplex* __restrict__ input3,
    uint8_t* __restrict__ outputBits0,
    uint8_t* __restrict__ outputBits1,
    uint8_t* __restrict__ outputBits2,
    uint8_t* __restrict__ outputBits3,
    uint32_t numSymbols) {
  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Demodulate each stream
  const cuComplex symbol0 = input0[symbolIndex];
  const cuComplex symbol1 = input1[symbolIndex];
  const cuComplex symbol2 = input2[symbolIndex];
  const cuComplex symbol3 = input3[symbolIndex];

  // Demodulate stream 0
  uint8_t bits0 = 0;
  if (symbol0.x >= 0.0f) {
    bits0 = (symbol0.y >= 0.0f) ? 0x0 : 0x2;
  } else {
    bits0 = (symbol0.y >= 0.0f) ? 0x1 : 0x3;
  }

  // Demodulate stream 1
  uint8_t bits1 = 0;
  if (symbol1.x >= 0.0f) {
    bits1 = (symbol1.y >= 0.0f) ? 0x0 : 0x2;
  } else {
    bits1 = (symbol1.y >= 0.0f) ? 0x1 : 0x3;
  }

  // Demodulate stream 2
  uint8_t bits2 = 0;
  if (symbol2.x >= 0.0f) {
    bits2 = (symbol2.y >= 0.0f) ? 0x0 : 0x2;
  } else {
    bits2 = (symbol2.y >= 0.0f) ? 0x1 : 0x3;
  }

  // Demodulate stream 3
  uint8_t bits3 = 0;
  if (symbol3.x >= 0.0f) {
    bits3 = (symbol3.y >= 0.0f) ? 0x0 : 0x2;
  } else {
    bits3 = (symbol3.y >= 0.0f) ? 0x1 : 0x3;
  }

  // Pack bits into output arrays (4 symbols per byte)
  const uint32_t byteIndex = symbolIndex / 4;
  const uint32_t bitOffset = (symbolIndex % 4) * 2;

  // Stream 0
  const uint8_t mask0 = ~(0x3 << bitOffset);
  outputBits0[byteIndex] = (outputBits0[byteIndex] & mask0) | (bits0 << bitOffset);

  // Stream 1
  const uint8_t mask1 = ~(0x3 << bitOffset);
  outputBits1[byteIndex] = (outputBits1[byteIndex] & mask1) | (bits1 << bitOffset);

  // Stream 2
  const uint8_t mask2 = ~(0x3 << bitOffset);
  outputBits2[byteIndex] = (outputBits2[byteIndex] & mask2) | (bits2 << bitOffset);

  // Stream 3
  const uint8_t mask3 = ~(0x3 << bitOffset);
  outputBits3[byteIndex] = (outputBits3[byteIndex] & mask3) | (bits3 << bitOffset);
}

GSDR_C_LINKAGE cudaError_t gsdrQpskModulate(
    const uint8_t* inputBits,
    cuComplex* output,
    uint32_t numSymbols,
    float amplitude,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_QpskModulate()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  k_QpskModulate<<<blocks, threads, 0, cudaStream>>>(
      inputBits,
      output,
      numSymbols,
      amplitude);

  SIMPLE_CUDA_FNC_END("k_QpskModulate()")
}

GSDR_C_LINKAGE cudaError_t gsdrQpskDemodulate(
    const cuComplex* input,
    uint8_t* outputBits,
    uint32_t numSymbols,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_QpskDemodulate()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  k_QpskDemodulate<<<blocks, threads, 0, cudaStream>>>(
      input,
      outputBits,
      numSymbols);

  SIMPLE_CUDA_FNC_END("k_QpskDemodulate()")
}

GSDR_C_LINKAGE cudaError_t gsdrQpskModulate4x(
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
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_QpskModulate4x()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  k_QpskModulate4x<<<blocks, threads, 0, cudaStream>>>(
      inputBits0,
      inputBits1,
      inputBits2,
      inputBits3,
      output0,
      output1,
      output2,
      output3,
      numSymbols,
      amplitude);

  SIMPLE_CUDA_FNC_END("k_QpskModulate4x()")
}

GSDR_C_LINKAGE cudaError_t gsdrQpskDemodulate4x(
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
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_QpskDemodulate4x()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  k_QpskDemodulate4x<<<blocks, threads, 0, cudaStream>>>(
      input0,
      input1,
      input2,
      input3,
      outputBits0,
      outputBits1,
      outputBits2,
      outputBits3,
      numSymbols);

  SIMPLE_CUDA_FNC_END("k_QpskDemodulate4x()")
}

// Template instantiations for common stream counts
template __global__ void k_QpskModulateTemplated<1>(const uint8_t* __restrict__ inputBits, cuComplex* __restrict__ output, uint32_t numSymbols, float amplitude);
template __global__ void k_QpskModulateTemplated<2>(const uint8_t* __restrict__ inputBits, cuComplex* __restrict__ output, uint32_t numSymbols, float amplitude);
template __global__ void k_QpskModulateTemplated<4>(const uint8_t* __restrict__ inputBits, cuComplex* __restrict__ output, uint32_t numSymbols, float amplitude);
template __global__ void k_QpskModulateTemplated<8>(const uint8_t* __restrict__ inputBits, cuComplex* __restrict__ output, uint32_t numSymbols, float amplitude);

template __global__ void k_QpskDemodulateTemplated<1>(const cuComplex* __restrict__ input, uint8_t* __restrict__ outputBits, uint32_t numSymbols);
template __global__ void k_QpskDemodulateTemplated<2>(const cuComplex* __restrict__ input, uint8_t* __restrict__ outputBits, uint32_t numSymbols);
template __global__ void k_QpskDemodulateTemplated<4>(const cuComplex* __restrict__ input, uint8_t* __restrict__ outputBits, uint32_t numSymbols);
template __global__ void k_QpskDemodulateTemplated<8>(const cuComplex* __restrict__ input, uint8_t* __restrict__ outputBits, uint32_t numSymbols);

// Optimized QPSK kernels for specific grid sizes using instruction-level parallelism

// 2-stream optimized kernel using vectorized operations
__global__ void k_QpskModulate2x(const uint8_t* __restrict__ inputBits0, const uint8_t* __restrict__ inputBits1,
                                cuComplex* __restrict__ output0, cuComplex* __restrict__ output1,
                                uint32_t numSymbols, float amplitude) {
  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  const uint32_t byteIndex = symbolIndex / 4;
  const uint32_t bitOffset = (symbolIndex % 4) * 2;

  // Process 2 streams with vectorized bit extraction
  const uint8_t bits0 = (inputBits0[byteIndex] >> bitOffset) & 0x3;
  const uint8_t bits1 = (inputBits1[byteIndex] >> bitOffset) & 0x3;

  // Vectorized constellation mapping
  cuComplex symbols[2];
  symbols[0] = (bits0 == 0) ? make_cuComplex(amplitude, amplitude) :
               (bits0 == 1) ? make_cuComplex(-amplitude, amplitude) :
               (bits0 == 3) ? make_cuComplex(-amplitude, -amplitude) :
               make_cuComplex(amplitude, -amplitude);

  symbols[1] = (bits1 == 0) ? make_cuComplex(amplitude, amplitude) :
               (bits1 == 1) ? make_cuComplex(-amplitude, amplitude) :
               (bits1 == 3) ? make_cuComplex(-amplitude, -amplitude) :
               make_cuComplex(amplitude, -amplitude);

  output0[symbolIndex] = symbols[0];
  output1[symbolIndex] = symbols[1];
}

// 8-stream optimized kernel for large grid processing
__global__ void k_QpskModulate8x(const uint8_t* __restrict__ inputBits0, const uint8_t* __restrict__ inputBits1,
                                const uint8_t* __restrict__ inputBits2, const uint8_t* __restrict__ inputBits3,
                                const uint8_t* __restrict__ inputBits4, const uint8_t* __restrict__ inputBits5,
                                const uint8_t* __restrict__ inputBits6, const uint8_t* __restrict__ inputBits7,
                                cuComplex* __restrict__ output0, cuComplex* __restrict__ output1,
                                cuComplex* __restrict__ output2, cuComplex* __restrict__ output3,
                                cuComplex* __restrict__ output4, cuComplex* __restrict__ output5,
                                cuComplex* __restrict__ output6, cuComplex* __restrict__ output7,
                                uint32_t numSymbols, float amplitude) {
  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  const uint32_t byteIndex = symbolIndex / 4;
  const uint32_t bitOffset = (symbolIndex % 4) * 2;

  // Extract bits for all 8 streams
  const uint8_t bits[8] = {
    (inputBits0[byteIndex] >> bitOffset) & 0x3,
    (inputBits1[byteIndex] >> bitOffset) & 0x3,
    (inputBits2[byteIndex] >> bitOffset) & 0x3,
    (inputBits3[byteIndex] >> bitOffset) & 0x3,
    (inputBits4[byteIndex] >> bitOffset) & 0x3,
    (inputBits5[byteIndex] >> bitOffset) & 0x3,
    (inputBits6[byteIndex] >> bitOffset) & 0x3,
    (inputBits7[byteIndex] >> bitOffset) & 0x3
  };

  // Vectorized constellation mapping for 8 streams
  for (int i = 0; i < 8; ++i) {
    cuComplex symbol;
    switch (bits[i]) {
      case 0: symbol = make_cuComplex(amplitude, amplitude); break;
      case 1: symbol = make_cuComplex(-amplitude, amplitude); break;
      case 3: symbol = make_cuComplex(-amplitude, -amplitude); break;
      case 2: symbol = make_cuComplex(amplitude, -amplitude); break;
      default: symbol = make_cuComplex(0.0f, 0.0f); break;
    }

    switch (i) {
      case 0: output0[symbolIndex] = symbol; break;
      case 1: output1[symbolIndex] = symbol; break;
      case 2: output2[symbolIndex] = symbol; break;
      case 3: output3[symbolIndex] = symbol; break;
      case 4: output4[symbolIndex] = symbol; break;
      case 5: output5[symbolIndex] = symbol; break;
      case 6: output6[symbolIndex] = symbol; break;
      case 7: output7[symbolIndex] = symbol; break;
    }
  }
}

// Host wrapper functions for template-based kernels
GSDR_C_LINKAGE cudaError_t gsdrQpskModulateTemplated(
    const uint8_t* inputBits,
    cuComplex* output,
    uint32_t numSymbols,
    float amplitude,
    int numStreams,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_QpskModulateTemplated()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  // Call appropriate template instantiation based on numStreams
  switch (numStreams) {
    case 1:
      k_QpskModulateTemplated<1><<<blocks, threads, 0, cudaStream>>>(inputBits, output, numSymbols, amplitude);
      break;
    case 2:
      k_QpskModulateTemplated<2><<<blocks, threads, 0, cudaStream>>>(inputBits, output, numSymbols, amplitude);
      break;
    case 4:
      k_QpskModulateTemplated<4><<<blocks, threads, 0, cudaStream>>>(inputBits, output, numSymbols, amplitude);
      break;
    case 8:
      k_QpskModulateTemplated<8><<<blocks, threads, 0, cudaStream>>>(inputBits, output, numSymbols, amplitude);
      break;
    default:
      // Fallback to original implementation
      k_QpskModulate<<<blocks, threads, 0, cudaStream>>>(inputBits, output, numSymbols, amplitude);
      break;
  }

  SIMPLE_CUDA_FNC_END("k_QpskModulateTemplated()")
}

GSDR_C_LINKAGE cudaError_t gsdrQpskDemodulateTemplated(
    const cuComplex* input,
    uint8_t* outputBits,
    uint32_t numSymbols,
    int numStreams,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numSymbols;
  SIMPLE_CUDA_FNC_START("k_QpskDemodulateTemplated()")

  const size_t threadsPerWarp = 32;
  const size_t blockCount = (numElements + threadsPerWarp - 1) / threadsPerWarp;

  const dim3 blocks(blockCount);
  const dim3 threads(threadsPerWarp);

  // Call appropriate template instantiation based on numStreams
  switch (numStreams) {
    case 1:
      k_QpskDemodulateTemplated<1><<<blocks, threads, 0, cudaStream>>>(input, outputBits, numSymbols);
      break;
    case 2:
      k_QpskDemodulateTemplated<2><<<blocks, threads, 0, cudaStream>>>(input, outputBits, numSymbols);
      break;
    case 4:
      k_QpskDemodulateTemplated<4><<<blocks, threads, 0, cudaStream>>>(input, outputBits, numSymbols);
      break;
    case 8:
      k_QpskDemodulateTemplated<8><<<blocks, threads, 0, cudaStream>>>(input, outputBits, numSymbols);
      break;
    default:
      // Fallback to original implementation
      k_QpskDemodulate<<<blocks, threads, 0, cudaStream>>>(input, outputBits, numSymbols);
      break;
  }

  SIMPLE_CUDA_FNC_END("k_QpskDemodulateTemplated()")
}