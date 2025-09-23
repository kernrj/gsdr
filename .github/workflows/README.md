# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the GSDR project. These workflows provide comprehensive CI/CD capabilities for testing, building, and validating the codebase.

## Workflows Overview

### 1. `test.yml` - Main Test Workflow
**Triggers:**
- Pull requests (opened, synchronized, reopened, ready for review)
- Pushes to main/develop branches

**Features:**
- ✅ CUDA 11.8 setup
- ✅ Full test suite execution
- ✅ Automatic PR commenting with results
- ✅ Test artifact uploads

**Use Case:** Primary CI for all PRs and commits

### 2. `ci-comprehensive.yml` - Comprehensive CI
**Triggers:**
- Pull requests
- Pushes to main/develop
- Manual dispatch

**Features:**
- ✅ Pre-checks (syntax validation, code quality)
- ✅ Multi-CUDA version testing (11.8, 12.0)
- ✅ Documentation validation
- ✅ Detailed reporting and status checks

**Use Case:** Full CI pipeline with multiple checks

### 3. `ci.yml` - Basic CI
**Triggers:**
- Pull requests
- Pushes to main/develop

**Features:**
- ✅ Simple CUDA setup
- ✅ Basic build and test
- ✅ Test result validation

**Use Case:** Lightweight CI for quick feedback

### 4. `cuda-matrix.yml` - CUDA Compatibility
**Triggers:**
- Pushes to main
- Pull requests
- Weekly schedule (Sundays at 2 AM UTC)

**Features:**
- ✅ Multiple CUDA versions (11.8, 12.0)
- ✅ Different CUDA architectures
- ✅ Compatibility validation

**Use Case:** Ensuring compatibility across CUDA versions

### 5. `manual-test.yml` - Manual Testing
**Triggers:**
- Manual workflow dispatch

**Features:**
- ✅ Configurable build type (Debug/Release)
- ✅ Configurable CUDA version
- ✅ Optional coverage reporting
- ✅ Optional extended tests

**Use Case:** Manual testing with custom configurations

### 6. `status.yml` - Status Badge Updates
**Triggers:**
- Pushes to main
- Completed test workflows

**Features:**
- ✅ Automatic status badge generation
- ✅ README.md updates with build status
- ✅ Badge color coding (green/red/yellow)

**Use Case:** Keeping repository status badges current

### 7. `pr-comment.yml` - PR Comment Trigger
**Triggers:**
- Issue comments containing `/test`

**Features:**
- ✅ Manual test triggering via PR comments
- ✅ Automatic result commenting
- ✅ Permission-based access control

**Use Case:** Manual test execution from PR comments

## Workflow Triggers

### Automatic Triggers
- **Pull Requests**: All PR events (opened, updated, synchronized, ready_for_review)
- **Main Branch Pushes**: Automatic testing on main branch updates
- **Scheduled**: Weekly CUDA compatibility testing
- **Workflow Dependencies**: Status updates when other workflows complete

### Manual Triggers
- **Workflow Dispatch**: Manual test execution with custom parameters
- **PR Comments**: `/test` command in PR comments (requires write permissions)

## Configuration Options

### CUDA Versions Supported
- **11.8**: Compatible with CUDA architectures 75, 80, 86
- **12.0**: Compatible with CUDA architectures 80, 86, 89

### Build Types
- **Release**: Optimized build with full testing
- **Debug**: Debug build with additional checks

### Test Coverage
- **Standard Tests**: All unit and integration tests
- **Extended Tests**: Performance tests and edge cases
- **Coverage Reports**: Code coverage analysis (when enabled)

## Usage Examples

### For Maintainers
1. **Regular Development**: PR workflow runs automatically
2. **Release Testing**: Use manual workflow with extended tests
3. **Compatibility**: Weekly scheduled runs ensure CUDA compatibility

### For Contributors
1. **PR Creation**: Automatic testing on PR creation
2. **PR Updates**: Automatic re-testing on every push to PR branch
3. **Manual Testing**: Use `/test` comment to trigger additional runs

### For Reviewers
1. **Status Badge**: Quick visual indication of test status
2. **Workflow Results**: Detailed test reports in PR comments
3. **Manual Triggers**: Ability to re-run tests if needed

## Status Badges

The repository includes automatically updated status badges:

```markdown
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/your-username/gsdr/actions/workflows/test.yml)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-blue)](https://developer.nvidia.com/cuda-toolkit)
```

## Troubleshooting

### Common Issues

1. **CUDA Setup Failures**
   - Ensure Ubuntu 20.04 runner is used
   - Check CUDA version compatibility

2. **Test Timeouts**
   - Extended tests may take longer
   - Check for infinite loops in test code

3. **Memory Issues**
   - Large test arrays may require more memory
   - Monitor GPU memory usage

### Workflow Logs

All workflows provide detailed logs:
- **Actions Tab**: Real-time workflow execution
- **Artifacts**: Test results and reports
- **PR Comments**: Summary of test results

## Performance

### Execution Times
- **Standard Tests**: ~5-10 minutes
- **Extended Tests**: ~15-20 minutes
- **Full CI**: ~10-15 minutes (parallel execution)

### Resource Usage
- **CPU**: 2-4 cores
- **Memory**: 4-8 GB RAM
- **GPU**: CUDA-compatible GPU (if available)

## Security

- **Permissions**: Workflows use minimal required permissions
- **External Actions**: All actions are from trusted sources
- **Artifact Retention**: 30 days for test results, 7 days for manual tests

## Contributing to Workflows

When adding new workflows:

1. Follow the existing naming convention
2. Include proper error handling
3. Add documentation in this README
4. Test workflows thoroughly
5. Use the most recent action versions

## Support

For workflow-related issues:
1. Check the Actions tab for detailed logs
2. Review workflow documentation above
3. Create an issue with workflow logs attached