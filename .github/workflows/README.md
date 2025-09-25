# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the GSDR project. These workflows provide **static code validation** and quality checks for the codebase.

## ✅ What GitHub Actions Does

**GitHub Actions runners perform static analysis and validation:**

- ✅ **Syntax Validation**: All C++/CUDA files compile without syntax errors
- ✅ **Code Quality**: Checks for TODO/FIXME comments, long lines, and style issues
- ✅ **Project Structure**: Ensures required directories and files are present
- ✅ **Test Coverage**: Counts and validates test file structure
- ✅ **Documentation**: Validates README and workflow files

**No GPU runtime testing is available** since GitHub Actions runners lack CUDA-capable hardware.

## Workflows Overview

### 1. `test.yml` - Code Quality Check (RECOMMENDED)
**Triggers:**
- Pull requests (opened, synchronized, reopened, ready for review)
- Pushes to main/develop branches

**Features:**
- ✅ **Syntax Validation**: Comprehensive C++/CUDA syntax checking
- ✅ **Code Quality**: TODO/FIXME detection, line length validation
- ✅ **Project Structure**: Directory and file existence checks
- ✅ **Test Coverage**: Test file counting and validation
- ✅ **PR Integration**: Automatic status comments with detailed reports

**Use Case:** Primary validation for all PRs and commits

### 2. `ci-comprehensive.yml` - Comprehensive Validation
**Triggers:**
- Pull requests
- Pushes to main/develop
- Manual dispatch

**Features:**
- ✅ **Static Analysis**: Advanced syntax and quality checks
- ✅ **Multi-check**: Comprehensive validation pipeline
- ✅ **Documentation**: README and structure validation
- ✅ **Detailed Reporting**: Comprehensive validation reports

**Use Case:** Complete static validation pipeline

### 3. `manual-test.yml` - Manual Validation Run
**Triggers:**
- Manual workflow dispatch

**Features:**
- ✅ **Configurable Options**: Custom validation settings
- ✅ **Extended Checks**: Additional quality validations
- ✅ **Coverage Analysis**: Placeholder for future coverage analysis
- ✅ **Detailed Reporting**: Custom validation reports

**Use Case:** Manual validation with specific configurations

### 4. `cuda-matrix.yml` - Multi-Platform Validation
**Triggers:**
- Push to main
- Pull requests
- Weekly schedule

**Features:**
- ✅ **Platform Testing**: Validation across different environments
- ✅ **Consistency Checks**: Ensures code works across platforms
- ✅ **Scheduled Runs**: Weekly validation to catch drift

**Use Case:** Platform compatibility validation

### 5. `gpu-tests.yml` - AWS CodeBuild GPU Testing
**Triggers:**
- Push to main/develop branches
- Pull requests to main/develop

**Features:**
- 🚀 **AWS CodeBuild Integration**: Runs on GPU-enabled CodeBuild instances
- 🚀 **Full CUDA Testing**: Complete build and test execution with GPU hardware
- 🚀 **Comprehensive Reporting**: Detailed test results and environment information
- 🚀 **PR Status Updates**: Automatic status checks for pull requests

**Use Case:** Full GPU runtime testing with actual CUDA hardware

### 6. `gpu-tests-simple.yml` - Simple GPU Testing
**Triggers:**
- Push to main/develop branches
- Pull requests to main/develop
- Manual dispatch

**Features:**
- 🚀 **AWS CodeBuild Integration**: Uses pre-configured CodeBuild project
- 🚀 **Streamlined**: Minimal configuration, uses existing build setup
- 🚀 **Quick Results**: Fast execution for existing CodeBuild projects

**Use Case:** GPU testing with pre-configured AWS CodeBuild project

## Validation Strategy

### What GitHub Actions Validates
1. **Syntax Validation**: Ensure all C++/CUDA code compiles correctly
2. **Code Quality**: Check for TODO/FIXME comments and style issues
3. **Project Structure**: Verify required directories and files exist
4. **Test Coverage**: Count and validate test files
5. **Documentation**: Validate README files and project structure

### Validation Results
- ✅ **PASS**: All static checks completed successfully
- ⚠️ **WARN**: Minor issues found (e.g., TODO comments)
- ❌ **FAIL**: Critical issues found (e.g., syntax errors, missing files)

### Status Badge
```markdown
[![CI Build](https://img.shields.io/badge/CI%20Validation-passing-brightgreen)](https://github.com/your-username/gsdr/actions/workflows/test.yml)
```

- **✅ Green**: All validation checks passed
- **❌ Red**: Critical validation issues found
- **⏳ Yellow**: Validation in progress

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
- `AWS_REGION`: AWS region for CodeBuild (default: us-east-1)
- `CODEBUILD_PROJECT_NAME`: Name of the AWS CodeBuild project

### Build Options
- **Standard Build**: Compiles library and basic tests
- **Test Build**: Includes comprehensive test suite
- **Coverage Build**: Adds code coverage analysis

### AWS CodeBuild Setup
For GPU testing workflows to work, you need:

1. **AWS Credentials**: Choose one of the authentication methods below
2. **CodeBuild Project**: GPU-enabled project named "gsdr" with appropriate instance types

#### Authentication Methods

**Option 1: AWS Access Keys (Recommended)**
Add these secrets to your GitHub repository:
- `AWS_ACCESS_KEY_ID`: Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key

**Option 2: AWS Role ARN**
Add this secret to your GitHub repository:
- `AWS_CODEBUILD_ROLE_ARN`: ARN of an IAM role with CodeBuild permissions

#### Required IAM Permissions (for Role ARN method)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "codebuild:StartBuild",
        "codebuild:BatchGetBuilds",
        "logs:GetLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Required IAM Permissions (for Access Keys method)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "codebuild:StartBuild",
        "codebuild:BatchGetBuilds",
        "logs:GetLogEvents",
        "logs:DescribeLogStreams",
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Debug Mode
To enable debug mode and see detailed AWS configuration information:
1. Go to your repository Settings → Variables and secrets → Variables
2. Add a new variable: `DEBUG_AWS_CONFIG` with value `true`
3. This will show credential configuration details in the workflow logs

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