# GPU CI Setup Guide

This guide explains how to configure and use the GPU testing workflow for the GSDR project.

## Overview

The GPU CI workflow runs CUDA tests on GitHub-hosted GPU runners, providing comprehensive testing of GPU-accelerated signal processing functions. The workflow includes fallback to CPU testing when GPU hardware is not available.

## Prerequisites

### 1. Enable GPU Runners

To use GPU runners, you need to enable **Larger runners → GPU-powered** at the organization or repository level:

1. Go to **Settings → Actions → Runners**
2. Click **New runner** → **Larger runners**
3. Select **GPU-powered** from the runner type dropdown
4. Choose your preferred OS (Ubuntu 20.04 recommended for CUDA compatibility)
5. Copy the generated runner label (e.g., `linux-nvidia-gpu`)

### 2. Configure Repository Variables

Set the following repository variables in **Settings → Secrets and variables → Variables**:

#### Required
- `GPU_RUNNER_LABEL`: The GPU runner label from step 1 (e.g., `linux-nvidia-gpu`)

#### Optional
- `DOCKERFILE_GPU`: Path to a Dockerfile for container-based GPU testing (e.g., `.github/Dockerfile.gpu`)

### 3. Billing Considerations

GPU runners incur additional costs beyond standard GitHub Actions minutes:

- **Pricing**: See [GitHub Actions billing](https://docs.github.com/billing/managing-billing-for-github-actions/about-billing-for-github-actions?utm_source=chatgpt.com)
- **Costs**: GPU runners are billed at a higher rate than standard runners
- **Concurrency**: Configure spending limits in organization settings
- **Usage**: Monitor usage in **Settings → Billing and plans → Usage**

## Workflow Features

### Jobs Included

1. **check-gpu**: Validates GPU hardware availability and CUDA setup
2. **gpu-tests**: Runs full GPU test suite with CMake and GTest
3. **gpu-tests-in-container**: Optional container-based testing (requires `DOCKERFILE_GPU`)
4. **cpu-fallback**: CPU-only testing when GPU is unavailable
5. **summary**: Comprehensive test results and health summary

### Triggers

The workflow runs on:
- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop`
- Manual dispatch with optional CPU fallback

## Configuration Options

### Repository Variables

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `GPU_RUNNER_LABEL` | GPU runner label from Settings → Actions → Runners | ✅ | `linux-nvidia-gpu` |
| `DOCKERFILE_GPU` | Path to GPU-enabled Dockerfile | ❌ | `.github/Dockerfile.gpu` |

### Workflow Dispatch

You can manually trigger the workflow with CPU fallback:

```yaml
workflow_dispatch:
  inputs:
    run_cpu_fallback:
      description: 'Run CPU fallback tests'
      required: false
      default: false
      type: boolean
```

## Testing and Verification

### 1. One-off Test Run

To test the configuration:

1. Go to **Actions** tab
2. Click **GPU Tests with CUDA** workflow
3. Click **Run workflow**
4. Check "Run CPU fallback tests" if needed
5. Click **Run workflow**

### 2. Verify GPU Hardware

The workflow performs several hardware checks:

```bash
# Check GPU availability
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

# Verify CUDA toolkit
nvcc --version

# Test CUDA runtime
# (Compiles and runs a simple CUDA program)
```

### 3. Monitor Test Results

- **Artifacts**: Test results are uploaded as workflow artifacts
- **Job Summary**: Detailed summary available in the workflow run
- **Logs**: Full logs with grouped output for easy debugging

## Troubleshooting

### GPU Runner Not Available

**Symptoms**:
- Workflow fails with "GPU_RUNNER_LABEL repo variable is not set"
- Jobs run on `ubuntu-latest` instead of GPU runner

**Solutions**:
1. Verify `GPU_RUNNER_LABEL` variable is set correctly
2. Ensure GPU runners are enabled in organization settings
3. Check runner status in **Settings → Actions → Runners**

### CUDA Toolkit Issues

**Symptoms**:
- `nvcc: command not found`
- CUDA runtime tests fail

**Solutions**:
1. The workflow automatically installs CUDA toolkit if missing
2. Verify CUDA version compatibility with your code
3. Check CUDA architecture settings in CMake (currently set to 75)

### Container Issues

**Symptoms**:
- Container tests fail or are skipped
- Docker build errors

**Solutions**:
1. Ensure `DOCKERFILE_GPU` variable points to a valid Dockerfile
2. Verify Dockerfile has proper CUDA setup
3. Check that NVIDIA Container Toolkit is available

### Performance Issues

**Symptoms**:
- Tests run slowly
- Memory errors during testing

**Solutions**:
1. Monitor GPU memory usage with `nvidia-smi`
2. Consider using `ccache` (already configured in workflow)
3. Review CUDA memory management in your code

## Cost Optimization

### 1. Usage Controls

- Set spending limits in organization billing settings
- Use manual dispatch only when needed
- Monitor usage in billing dashboard

### 2. Efficient Testing

- Tests run only on relevant file changes
- Caching reduces build times
- CPU fallback avoids unnecessary GPU costs

### 3. Concurrency Management

The workflow includes concurrency controls to cancel superseded runs:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

## Support and Resources

### Documentation Links

- **GPU Runners GA**: [The GitHub Blog](https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/?utm_source=chatgpt.com)
- **Managing Larger Runners**: [GitHub Docs](https://docs.github.com/actions/using-github-hosted-runners/managing-larger-runners?utm_source=chatgpt.com)
- **About Larger Runners**: [GitHub Docs](https://docs.github.com/actions/using-github-hosted-runners/about-larger-runners/about-larger-runners?utm_source=chatgpt.com)
- **Billing Information**: [GitHub Docs](https://docs.github.com/billing/managing-billing-for-github-actions/about-billing-for-github-actions?utm_source=chatgpt.com)
- **Choosing Runners**: [GitHub Docs](https://docs.github.com/actions/using-jobs/choosing-the-runner-for-a-job?utm_source=chatgpt.com)

### Getting Help

- Check workflow logs for detailed error messages
- Review GitHub Actions documentation for runner setup
- Contact GitHub support for billing and runner configuration issues

## Local Development

For local GPU testing, ensure you have:

```bash
# Required packages
sudo apt-get install nvidia-cuda-toolkit

# Verify setup
nvidia-smi
nvcc --version

# Build and test locally
mkdir build && cd build
cmake .. -DUSE_TESTS=ON
make
ctest --output-on-failure
```

---

*This documentation was generated for the GSDR project GPU CI setup. Last updated: $(date)*