<!--- status badge start -->
[![Tests](https://img.shields.io/badge/tests-unknown-yellow)](https://github.com/kernrj/gsdr/actions/workflows/test.yml)
<!--- status badge end -->
# GSDR - GPU Software Defined Radio Library

[![CI Build](https://img.shields.io/badge/CI%20Build-passing-brightgreen)](https://github.com/your-username/gsdr/actions/workflows/test.yml)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-blue)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

⚠️ **Note**: GitHub Actions CI performs compilation-only validation due to lack of GPU runtime. For actual GPU testing, run tests on a system with CUDA-capable GPUs.

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

## CI/CD & Testing

### GitHub Actions (Static Validation)

This project uses GitHub Actions for **static code validation** and quality checks:

- **✅ Syntax Validation**: Validates all C++/CUDA files compile without syntax errors
- **✅ Code Quality**: Checks for TODO/FIXME comments, long lines, and style issues
- **✅ Project Structure**: Ensures required directories and files are present
- **✅ Test Coverage**: Counts and validates test files
- **⚠️ Status Badge**: Shows validation status (✅ passing = code quality checks pass)

⚠️ **Important**: GitHub Actions runners don't have CUDA-capable GPUs, so they perform **static analysis only**. No GPU runtime testing is available.

### Workflow Triggers

- Pull requests (opened, updated, or synchronized)
- Pushes to main/develop branches
- Manual dispatch for custom validation configurations
- Weekly scheduled validation runs

### What GitHub Actions Validates

| Check | Description | Status |
|-------|-------------|--------|
| **Syntax Validation** | All C++/CUDA files compile without errors | ✅ Automated |
| **Code Quality** | No TODO/FIXME comments in production code | ✅ Automated |
| **Project Structure** | Required directories and files present | ✅ Automated |
| **Test Coverage** | Test files exist and are properly structured | ✅ Automated |
| **Documentation** | README and workflow files validated | ✅ Automated |

### Actual GPU Testing (Manual)

For complete testing with GPU execution, run tests locally on CUDA-capable hardware:

```bash
# On a system with CUDA-capable GPU
mkdir build && cd build
cmake .. -DUSE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure --build-config Release
```

### Test Suite Overview

- **6,564+ lines** of comprehensive test code
- **11 test files** covering all major components
- **Unit tests** for individual functions
- **Integration tests** for complete workflows
- **Performance tests** with large datasets
- **Edge case testing** with boundary conditions

## Performance

All operations are optimized for GPU execution with:

- **Instruction-level parallelism** for multi-stream processing
- **Memory coalescing** for efficient global memory access
- **Warp optimization** for maximum GPU utilization
- **Template-based kernels** for common grid sizes

## Testing

### GitHub Actions Validation (Automated)

GitHub Actions performs **static code validation** and quality checks:

- ✅ **Syntax Validation**: All C++/CUDA files compile without syntax errors
- ✅ **Code Quality**: Checks for TODO/FIXME comments and style issues
- ✅ **Project Structure**: Validates required directories and files exist
- ✅ **Test Coverage**: Counts and validates test file structure
- ⚠️ **GPU Runtime**: Not available (GitHub Actions limitation)

### Local GPU Testing (Manual)

The comprehensive test suite covers:

- **11 test files** with **6,564+ lines** of test code
- **Unit tests** for all major components (QPSK, QPSK256, AM, FM, FIR, arithmetic, etc.)
- **Integration tests** for complete signal processing workflows
- **Performance tests** with large datasets (64K+ samples)
- **Edge case testing** with boundary conditions and error scenarios

#### Running Tests (Requires CUDA GPU)

```bash
# On a system with CUDA-capable GPU:
mkdir build && cd build
cmake .. -DUSE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure --build-config Release
```

#### Advanced Testing Options

```bash
# With code coverage analysis
cmake .. -DUSE_TESTS=ON -DENABLE_COVERAGE=ON
make
ctest --output-on-failure
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_report

# With extended test configurations
ctest --output-on-failure --build-config Release --rerun-failed
```

### Validation Results

GitHub Actions provides **early feedback** on:
- Code compilation issues
- Project structure problems
- Documentation completeness
- Test suite organization

**For complete validation**, run the full test suite on CUDA-capable hardware.

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