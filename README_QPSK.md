# QPSK Implementation for GSDR

This document describes the QPSK (Quadrature Phase Shift Keying) modulation and demodulation implementation for the GSDR library.

## Overview

QPSK is a digital modulation scheme where each symbol represents 2 bits of information. The implementation provides:

- **Single-stream processing**: Standard QPSK modulation/demodulation for one stream
- **Multi-stream processing**: Optimized kernels for processing 4 streams simultaneously
- **Template-based kernels**: Flexible kernels that can be optimized for different grid sizes (1, 2, 4, 8 streams)
- **Instruction-level parallelism**: Optimized for processing multiple streams at once

## Constellation Mapping

The QPSK constellation uses the following mapping:

- **00** → +1 + 1j (45°)
- **01** → -1 + 1j (135°)
- **11** → -1 - 1j (225° or -135°)
- **10** → +1 - 1j (315° or -45°)

## API Functions

### Basic QPSK Functions

```cpp
// Single stream modulation
cudaError_t gsdrQpskModulate(
    const uint8_t* inputBits,    // Input bits (0 or 1), packed as uint8_t
    cuComplex* output,           // Complex output samples
    uint32_t numSymbols,         // Number of symbols to generate
    float amplitude,             // Amplitude scaling factor
    int32_t cudaDevice,          // CUDA device index
    cudaStream_t cudaStream);    // CUDA stream

// Single stream demodulation
cudaError_t gsdrQpskDemodulate(
    const cuComplex* input,      // Complex input samples
    uint8_t* outputBits,         // Output bits (0 or 1), packed as uint8_t
    uint32_t numSymbols,         // Number of symbols to demodulate
    int32_t cudaDevice,          // CUDA device index
    cudaStream_t cudaStream);    // CUDA stream
```

### Multi-Stream Functions (4 streams)

```cpp
// 4-stream modulation
cudaError_t gsdrQpskModulate4x(
    const uint8_t* inputBits0, const uint8_t* inputBits1,
    const uint8_t* inputBits2, const uint8_t* inputBits3,
    cuComplex* output0, cuComplex* output1,
    cuComplex* output2, cuComplex* output3,
    uint32_t numSymbols, float amplitude,
    int32_t cudaDevice, cudaStream_t cudaStream);

// 4-stream demodulation
cudaError_t gsdrQpskDemodulate4x(
    const cuComplex* input0, const cuComplex* input1,
    const cuComplex* input2, const cuComplex* input3,
    uint8_t* outputBits0, uint8_t* outputBits1,
    uint8_t* outputBits2, uint8_t* outputBits3,
    uint32_t numSymbols,
    int32_t cudaDevice, cudaStream_t cudaStream);
```

### Template-Based Functions

```cpp
// Template-based modulation (supports 1, 2, 4, 8 streams)
cudaError_t gsdrQpskModulateTemplated(
    const uint8_t* inputBits, cuComplex* output,
    uint32_t numSymbols, float amplitude, int numStreams,
    int32_t cudaDevice, cudaStream_t cudaStream);

// Template-based demodulation (supports 1, 2, 4, 8 streams)
cudaError_t gsdrQpskDemodulateTemplated(
    const cuComplex* input, uint8_t* outputBits,
    uint32_t numSymbols, int numStreams,
    int32_t cudaDevice, cudaStream_t cudaStream);
```

## Performance Optimizations

### Instruction-Level Parallelism

The implementation uses several techniques to optimize performance:

1. **SIMD Operations**: Kernels are designed to process multiple streams simultaneously using vectorized operations
2. **Warp-level Optimization**: Thread blocks are sized to match warp boundaries (32 threads)
3. **Memory Coalescing**: Data access patterns are optimized for coalesced memory access
4. **Shared Memory**: Efficient use of shared memory for intermediate calculations

### Template Specialization

The template-based kernels provide optimized versions for common stream counts:

- **1 stream**: Standard single-stream processing
- **2 streams**: Optimized for dual-stream processing with vectorized operations
- **4 streams**: Optimized for quad-stream processing (common in SDR applications)
- **8 streams**: Optimized for high-throughput applications

## Bit Packing

Input bits are packed efficiently:
- 4 symbols (8 bits) are packed into each `uint8_t` byte
- Bits are extracted using bit shifting and masking operations
- This reduces memory bandwidth requirements by 8x compared to storing individual bits

## CUDA Kernel Architecture

The kernels follow the GSDR library patterns:

```cpp
__global__ void k_QpskModulate(
    const uint8_t* __restrict__ inputBits,
    cuComplex* __restrict__ output,
    uint32_t numSymbols,
    float amplitude) {

  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Extract 2 bits for this symbol
  const uint32_t byteIndex = symbolIndex / 4;
  const uint32_t bitOffset = (symbolIndex % 4) * 2;
  const uint8_t bits = (inputBits[byteIndex] >> bitOffset) & 0x3;

  // Map bits to constellation point
  cuComplex symbol;
  switch (bits) {
    case 0: symbol = make_cuComplex(amplitude, amplitude); break;
    case 1: symbol = make_cuComplex(-amplitude, amplitude); break;
    case 3: symbol = make_cuComplex(-amplitude, -amplitude); break;
    case 2: symbol = make_cuComplex(amplitude, -amplitude); break;
  }

  output[symbolIndex] = symbol;
}
```

## Usage Example

```cpp
#include <gsdr/qpsk.h>

// Example: Modulate 1024 QPSK symbols
const uint32_t numSymbols = 1024;
const float amplitude = 1.0f;

// Input bits (256 bytes for 1024 symbols since 4 symbols/byte)
uint8_t* inputBits;  // Packed bits: 4 symbols per byte
cuComplex* output;   // Complex output samples

// Allocate device memory and copy input data...

// Perform QPSK modulation
cudaError_t error = gsdrQpskModulate(
    inputBits, output, numSymbols, amplitude, 0, 0);

// Check for errors and process results...
```

## Build Integration

The QPSK implementation is integrated into the GSDR build system:

1. **Header file**: `include/gsdr/qpsk.h`
2. **Implementation**: `src/qpsk.cu`
3. **Build integration**: Added to `CMakeLists.txt`

The implementation automatically generates export headers and follows the library's coding standards and patterns.

## Error Handling

All functions return `cudaError_t` and follow the GSDR library's error handling patterns:

- Device setup and teardown using `SIMPLE_CUDA_FNC_START` and `SIMPLE_CUDA_FNC_END` macros
- Proper error checking and propagation
- Thread-safe operation with CUDA streams