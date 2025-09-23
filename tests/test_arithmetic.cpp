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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include <random>

#include "test_main.cpp"
#include <gsdr/arithmetic.h>

class ArithmeticTest : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();
    }

    cudaError_t error = cudaSuccess;
    const size_t testSize = 1024;

    std::vector<float> generateTestData(size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        std::vector<float> data(size);
        std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
        return data;
    }

    std::vector<cuComplex> generateComplexTestData(size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        std::vector<cuComplex> data(size);
        std::generate(data.begin(), data.end(), [&]() {
            return make_cuComplex(dis(gen), dis(gen));
        });
        return data;
    }
};

TEST_F(ArithmeticTest, AddConstFloatFloatTest) {
    auto input = generateTestData(testSize);
    auto output = std::vector<float>(testSize);
    const float addConst = 3.14f;

    error = gsdrAddConstFF(input.data(), addConst, output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AddConstFF failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        float expected = input[i] + addConst;
        EXPECT_NEAR(output[i], expected, 1e-6f) << "Incorrect result at index " << i;
    }
}

TEST_F(ArithmeticTest, AddConstComplexComplexTest) {
    auto input = generateComplexTestData(testSize);
    auto output = std::vector<cuComplex>(testSize);
    const cuComplex addConst = make_cuComplex(1.5f, -2.5f);

    error = gsdrAddConstCC(input.data(), addConst, output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AddConstCC failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        cuComplex expected = make_cuComplex(input[i].x + addConst.x, input[i].y + addConst.y);
        EXPECT_NEAR(output[i].x, expected.x, 1e-6f) << "Incorrect real result at index " << i;
        EXPECT_NEAR(output[i].y, expected.y, 1e-6f) << "Incorrect imaginary result at index " << i;
    }
}

TEST_F(ArithmeticTest, AddConstComplexFloatTest) {
    auto input = generateComplexTestData(testSize);
    auto output = std::vector<cuComplex>(testSize);
    const float addConst = -1.23f;

    error = gsdrAddConstCF(input.data(), addConst, output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AddConstCF failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        cuComplex expected = make_cuComplex(input[i].x + addConst, input[i].y + addConst);
        EXPECT_NEAR(output[i].x, expected.x, 1e-6f) << "Incorrect real result at index " << i;
        EXPECT_NEAR(output[i].y, expected.y, 1e-6f) << "Incorrect imaginary result at index " << i;
    }
}

TEST_F(ArithmeticTest, AddConstFloatComplexTest) {
    auto input = generateTestData(testSize);
    auto output = std::vector<cuComplex>(testSize);
    const cuComplex addConst = make_cuComplex(2.5f, -1.5f);

    error = gsdrAddConstFC(input.data(), addConst, output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AddConstFC failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        cuComplex expected = make_cuComplex(input[i] + addConst.x, input[i] + addConst.y);
        EXPECT_NEAR(output[i].x, expected.x, 1e-6f) << "Incorrect real result at index " << i;
        EXPECT_NEAR(output[i].y, expected.y, 1e-6f) << "Incorrect imaginary result at index " << i;
    }
}

TEST_F(ArithmeticTest, MultiplyComplexComplexTest) {
    auto input1 = generateComplexTestData(testSize);
    auto input2 = generateComplexTestData(testSize);
    auto output = std::vector<cuComplex>(testSize);

    error = gsdrMultiplyCC(input1.data(), input2.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "MultiplyCC failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        cuComplex expected = make_cuComplex(
            input1[i].x * input2[i].x - input1[i].y * input2[i].y,
            input1[i].x * input2[i].y + input1[i].y * input2[i].x
        );
        EXPECT_NEAR(output[i].x, expected.x, 1e-5f) << "Incorrect real result at index " << i;
        EXPECT_NEAR(output[i].y, expected.y, 1e-5f) << "Incorrect imaginary result at index " << i;
    }
}

TEST_F(ArithmeticTest, MultiplyFloatFloatTest) {
    auto input1 = generateTestData(testSize);
    auto input2 = generateTestData(testSize);
    auto output = std::vector<float>(testSize);

    error = gsdrMultiplyFF(input1.data(), input2.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "MultiplyFF failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        float expected = input1[i] * input2[i];
        EXPECT_NEAR(output[i], expected, 1e-6f) << "Incorrect result at index " << i;
    }
}

TEST_F(ArithmeticTest, MultiplyComplexFloatTest) {
    auto input1 = generateComplexTestData(testSize);
    auto input2 = generateTestData(testSize);
    auto output = std::vector<cuComplex>(testSize);

    error = gsdrMultiplyCF(input1.data(), input2.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "MultiplyCF failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        cuComplex expected = make_cuComplex(
            input1[i].x * input2[i],
            input1[i].y * input2[i]
        );
        EXPECT_NEAR(output[i].x, expected.x, 1e-6f) << "Incorrect real result at index " << i;
        EXPECT_NEAR(output[i].y, expected.y, 1e-6f) << "Incorrect imaginary result at index " << i;
    }
}

TEST_F(ArithmeticTest, MagnitudeTest) {
    auto input = generateComplexTestData(testSize);
    auto output = std::vector<float>(testSize);

    error = gsdrMagnitude(input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Magnitude failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        float expected = sqrtf(input[i].x * input[i].x + input[i].y * input[i].y);
        EXPECT_NEAR(output[i], expected, 1e-6f) << "Incorrect magnitude at index " << i;
    }
}

TEST_F(ArithmeticTest, AbsTest) {
    auto input = generateTestData(testSize);
    auto output = std::vector<float>(testSize);

    // Make some negative values
    for (size_t i = 0; i < testSize; i += 2) {
        input[i] = -std::abs(input[i]);
    }

    error = gsdrAbs(input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Abs failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        float expected = std::abs(input[i]);
        EXPECT_NEAR(output[i], expected, 1e-6f) << "Incorrect absolute value at index " << i;
    }
}

TEST_F(ArithmeticTest, AddToMagnitudeTest) {
    auto input = generateComplexTestData(testSize);
    auto output = std::vector<cuComplex>(testSize);
    const float addValue = 2.5f;

    error = gsdrAddToMagnitude(input.data(), addValue, output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AddToMagnitude failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        float originalMag = sqrtf(input[i].x * input[i].x + input[i].y * input[i].y);
        float newMag = originalMag + addValue;

        // Check that magnitude is increased by addValue
        float actualNewMag = sqrtf(output[i].x * output[i].x + output[i].y * output[i].y);
        EXPECT_NEAR(actualNewMag, newMag, 1e-5f) << "Incorrect magnitude at index " << i;

        // Check that direction is preserved (phase should be same)
        if (originalMag > 1e-6f) {
            float originalPhase = atan2f(input[i].y, input[i].x);
            float newPhase = atan2f(output[i].y, output[i].x);
            EXPECT_NEAR(originalPhase, newPhase, 1e-5f) << "Phase changed at index " << i;
        }
    }
}

TEST_F(ArithmeticTest, ZeroInputTest) {
    std::vector<float> zeroInput(testSize, 0.0f);
    std::vector<float> output(testSize);
    const float addConst = 5.0f;

    // Test AddConstFF with zero input
    error = gsdrAddConstFF(zeroInput.data(), addConst, output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AddConstFF with zero input failed";

    for (size_t i = 0; i < testSize; ++i) {
        EXPECT_EQ(output[i], addConst) << "Incorrect result for zero input at index " << i;
    }

    // Test MultiplyFF with zero input
    error = gsdrMultiplyFF(zeroInput.data(), zeroInput.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "MultiplyFF with zero input failed";

    for (size_t i = 0; i < testSize; ++i) {
        EXPECT_EQ(output[i], 0.0f) << "Incorrect result for zero multiplication at index " << i;
    }
}

TEST_F(ArithmeticTest, EdgeCasesTest) {
    // Test with different sizes
    const std::vector<size_t> testSizes = {1, 2, 3, 4, 31, 32, 33, 1023, 1024, 1025};

    for (size_t size : testSizes) {
        auto input1 = generateTestData(size);
        auto input2 = generateTestData(size);
        auto output = std::vector<float>(size);

        // Test AddConstFF
        error = gsdrAddConstFF(input1.data(), 1.0f, output.data(), size, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "AddConstFF failed for size " << size;

        // Test MultiplyFF
        error = gsdrMultiplyFF(input1.data(), input2.data(), output.data(), size, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "MultiplyFF failed for size " << size;
    }
}

TEST_F(ArithmeticTest, LargeNumbersTest) {
    // Test with large numbers
    std::vector<float> largeInput(testSize);
    std::fill(largeInput.begin(), largeInput.end(), 1e6f);
    auto output = std::vector<float>(testSize);

    error = gsdrMultiplyFF(largeInput.data(), largeInput.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "MultiplyFF with large numbers failed";

    for (size_t i = 0; i < testSize; ++i) {
        EXPECT_NEAR(output[i], 1e12f, 1e6f) << "Incorrect result for large numbers at index " << i;
    }
}

TEST_F(ArithmeticTest, SpecialValuesTest) {
    std::vector<float> specialInput = {0.0f, 1.0f, -1.0f, INFINITY, -INFINITY, NAN};
    auto output = std::vector<float>(specialInput.size());

    // Test with special values
    error = gsdrAbs(specialInput.data(), output.data(), specialInput.size(), 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Abs with special values failed";

    EXPECT_EQ(output[0], 0.0f) << "Abs(0) should be 0";
    EXPECT_EQ(output[1], 1.0f) << "Abs(1) should be 1";
    EXPECT_EQ(output[2], 1.0f) << "Abs(-1) should be 1";
    // INFINITY and NAN behavior may vary, but should not crash
}