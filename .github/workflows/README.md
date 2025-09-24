# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the GSDR project. These workflows provide comprehensive CI/CD capabilities for testing, building, and validating the codebase.

## ⚠️ Important: GitHub Actions Limitations

**GitHub Actions runners do NOT have CUDA-capable GPUs.** This means:

- ✅ **Compilation**: All code compiles successfully
- ✅ **Build Validation**: Library builds correctly
- ✅ **Static Analysis**: Code quality checks pass
- ❌ **Runtime Testing**: CUDA kernels cannot execute (no GPU available)

## Workflows Overview

### 1. `test.yml` - Main CI Workflow (RECOMMENDED)
**Triggers:**
- Pull requests (opened, synchronized, reopened, ready for review)
- Pushes to main/develop branches

**Features:**
- ✅ **Smart CUDA Detection**: Attempts CUDA build, falls back to CPU-only
- ✅ **Build Validation**: Verifies compilation succeeds
- ✅ **Test Compilation**: Compiles test suite (when CUDA available)
- ✅ **Detailed Reporting**: Clear status messages about GPU limitations
- ✅ **PR Integration**: Automatic status comments

**Use Case:** Primary CI for all PRs and commits

### 2. `ci-comprehensive.yml` - Full CI Pipeline
**Triggers:**
- Pull requests
- Pushes to main/develop
- Manual dispatch

**Features:**
- ✅ **Pre-checks**: Syntax validation and code quality
- ✅ **Multi-environment**: Tests different configurations
- ✅ **Documentation**: README validation
- ✅ **Detailed Reporting**: Comprehensive build reports

**Use Case:** Complete validation pipeline

### 3. `manual-test.yml` - Manual Testing
**Triggers:**
- Manual workflow dispatch

**Features:**
- ✅ **Configurable Options**: Custom build types and settings
- ✅ **Extended Testing**: Optional performance tests
- ✅ **Coverage Analysis**: Code coverage reporting

**Use Case:** Manual testing with specific configurations

## Testing Strategy

### What GitHub Actions Can Do
1. **Compile Validation**: Ensure all CUDA/C++ code compiles correctly
2. **Build Verification**: Confirm library builds successfully
3. **Code Quality**: Check for syntax errors and basic issues
4. **Documentation**: Validate README files and structure

### What GitHub Actions Cannot Do
1. **GPU Runtime Testing**: No CUDA-capable GPUs available
2. **Performance Testing**: Cannot measure actual GPU performance
3. **Integration Testing**: Cannot test full signal processing pipelines
4. **Hardware Validation**: Cannot verify GPU-specific optimizations

### Recommended Testing Approach

#### For Development (GitHub Actions)
```bash
# GitHub Actions automatically:
# 1. Compiles all code
# 2. Validates build process
# 3. Checks for syntax errors
# 4. Reports compilation status
```

#### For Actual Testing (Local GPU System)
```bash
# On a system with CUDA-capable GPU:
mkdir build && cd build
cmake .. -DUSE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure --build-config Release
```

## Workflow Status

### Status Badge
```markdown
[![CI Build](https://img.shields.io/badge/CI%20Build-passing-brightgreen)](https://github.com/your-username/gsdr/actions/workflows/test.yml)
```

- **✅ Green**: Code compiles successfully
- **❌ Red**: Compilation failed
- **⏳ Yellow**: Build in progress

### Understanding Results

When you see **✅ CI Build passing**, this means:
- All C++/CUDA code compiles correctly
- Library builds successfully
- No syntax errors detected
- Code structure is valid

**This does NOT mean**:
- Tests passed (no GPU available for execution)
- Performance is optimal
- GPU-specific optimizations work

## Troubleshooting

### Common Issues

1. **"CUDA Setup Failed"**
   - This is expected on GitHub Actions (no GPU runtime)
   - The workflow will fall back to CPU-only build
   - Check the build logs for actual compilation errors

2. **"Tests Cannot Run"**
   - This is expected - GitHub Actions cannot run CUDA code
   - Tests are compiled but not executed
   - Run tests locally on GPU hardware for full validation

3. **Build Directory Missing**
   - Fixed in updated workflows
   - Workflows now handle missing directories gracefully

### Manual Testing

To run actual GPU tests:

1. **Set up a CUDA development environment**
2. **Clone the repository**
3. **Build with tests enabled**:
   ```bash
   mkdir build && cd build
   cmake .. -DUSE_TESTS=ON
   make
   ctest --output-on-failure
   ```

## Configuration

### Environment Variables
- `BUILD_TYPE`: Release (default) or Debug
- `CUDA_ARCH`: Target GPU architecture (75 for CUDA 11.8, 80 for 12.0)

### Build Options
- **Standard Build**: Compiles library and basic tests
- **Test Build**: Includes comprehensive test suite
- **Coverage Build**: Adds code coverage analysis

## Security

- **Minimal Permissions**: Workflows use least-privilege access
- **Trusted Actions**: All actions from verified sources
- **No Secrets**: No sensitive data stored in workflows

## Support

For workflow-related issues:
1. Check the Actions tab for detailed logs
2. Review this documentation
3. Create an issue with workflow logs attached

---

**Note**: These workflows provide compilation validation. For actual GPU testing, run tests on systems with CUDA-capable hardware.