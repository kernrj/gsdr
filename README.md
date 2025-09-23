# GSDR - GPU Software Defined Radio Library

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/your-username/gsdr/actions/workflows/test.yml)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-blue)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A high-performance CUDA-based Software Defined Radio (SDR) library providing GPU-accelerated digital signal processing operations for communication systems.

## Features

- **High-Performance Modulation/Demodulation**: QPSK, QPSK256, AM, FM
- **Advanced Filtering**: FIR filters with decimation support
- **Arithmetic Operations**: Complex and real number operations on GPU
- **Data Conversion**: Efficient format conversions
- **Comprehensive Testing**: Full test suite with CI/CD integration

## Quick Start

### Prerequisites

- CUDA 11.8 or higher
- CMake 3.17 or higher
- GCC/Clang with C++11 support
- Google Test (for running tests)

### Building

```bash
mkdir build
cd build
cmake .. -DUSE_TESTS=ON
make -j$(nproc)
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

## Components

### Modulation/Demodulation

- **QPSK** - Quadrature Phase Shift Keying (2 bits/symbol)
- **QPSK256** - 256-ary QPSK (8 bits/symbol)
- **AM** - Amplitude Modulation demodulation
- **FM** - Frequency Modulation demodulation

### Signal Processing

- **FIR Filters** - Finite Impulse Response filters with decimation
- **Arithmetic** - Complex and real number operations
- **Conversion** - Data format conversions
- **Trigonometric** - GPU-accelerated trig functions

## Documentation

- [QPSK Implementation Details](README_QPSK.md)
- [QPSK256 Implementation Details](README_QPSK256.md)

## CI/CD

This project uses GitHub Actions for continuous integration:

- **Test Workflow**: Runs on every PR and push to main branches
- **CUDA Matrix**: Tests against multiple CUDA versions (11.8, 12.0)
- **Status Badge**: Automatically updated test status
- **Documentation Check**: Validates README files

### Workflow Triggers

- Pull requests (opened, updated, or synchronized)
- Pushes to main/develop branches
- Manual dispatch with custom CUDA version
- Weekly scheduled runs for compatibility testing

## Performance

All operations are optimized for GPU execution with:

- **Instruction-level parallelism** for multi-stream processing
- **Memory coalescing** for efficient global memory access
- **Warp optimization** for maximum GPU utilization
- **Template-based kernels** for common grid sizes

## Testing

The test suite covers:

- **Unit tests** for all major components
- **Integration tests** for complete workflows
- **Performance tests** with large datasets
- **Edge case testing** with boundary conditions
- **Error handling** validation

Run tests with:
```bash
ctest --output-on-failure --build-config Release
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Licensed under the Apache License, Version 2.0. See [LICENSE.txt](LICENSE.txt) for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review test cases for usage examples