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
#include <cmath>

#include "test_main.cpp"
#include <gsdr/trig.h>

class TrigTest : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();
    }

    cudaError_t error = cudaSuccess;
    const size_t testSize = 1024;

    std::vector<float> generatePhaseData(size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PIf);

        std::vector<float> data(size);
        std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
        return data;
    }
};

TEST_F(TrigTest, CosineFloatTest) {
    auto input = generatePhaseData(testSize);
    auto output = std::vector<float>(testSize);

    error = gsdrCosineF(0.0f, 2.0f * M_PIf, output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "CosineF failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        float expected = cosf(input[i]);
        EXPECT_NEAR(output[i], expected, 1e-6f) << "Incorrect cosine value at index " << i;
    }
}

TEST_F(TrigTest, CosineComplexTest) {
    auto input = generatePhaseData(testSize);
    auto output = std::vector<cuComplex>(testSize);

    error = gsdrCosineC(0.0f, 2.0f * M_PIf, output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "CosineC failed";

    // Verify results
    for (size_t i = 0; i < testSize; ++i) {
        cuComplex expected = make_cuComplex(cosf(input[i]), sinf(input[i])); // e^(j*phase) = cos + j*sin
        EXPECT_NEAR(output[i].x, expected.x, 1e-6f) << "Incorrect real cosine value at index " << i;
        EXPECT_NEAR(output[i].y, expected.y, 1e-6f) << "Incorrect imaginary cosine value at index " << i;
    }
}

TEST_F(TrigTest, CosinePhaseRangeTest) {
    // Test different phase ranges
    const std::vector<std::pair<float, float>> phaseRanges = {
        {0.0f, 2.0f * M_PIf},     // Full cycle
        {0.0f, M_PIf},            // Half cycle
        {0.0f, M_PIf / 2.0f},     // Quarter cycle
        {-M_PIf, M_PIf},          // Symmetric around zero
        {M_PIf, 3.0f * M_PIf / 2.0f} // Partial range
    };

    for (const auto& range : phaseRanges) {
        auto output = std::vector<float>(testSize);

        error = gsdrCosineF(range.first, range.second, output.data(), testSize, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "CosineF failed for range [" << range.first << ", " << range.second << "]";

        // Verify output range
        for (float val : output) {
            EXPECT_GE(val, -1.0f) << "Cosine value too negative for range [" << range.first << ", " << range.second << "]";
            EXPECT_LE(val, 1.0f) << "Cosine value too positive for range [" << range.first << ", " << range.second << "]";
        }
    }
}

TEST_F(TrigTest, CosineEdgeCasesTest) {
    const std::vector<size_t> testSizes = {1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025};

    for (size_t size : testSizes) {
        auto output = std::vector<float>(size);

        error = gsdrCosineF(0.0f, 2.0f * M_PIf, output.data(), size, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "CosineF failed for size " << size;

        // Verify output size
        EXPECT_EQ(output.size(), size) << "Incorrect output size for size " << size;
    }
}

TEST_F(TrigTest, CosineKnownValuesTest) {
    // Test with known phase values
    std::vector<float> knownPhases = {0.0f, M_PIf / 2.0f, M_PIf, 3.0f * M_PIf / 2.0f, 2.0f * M_PIf};
    auto output = std::vector<float>(knownPhases.size());

    error = gsdrCosineF(0.0f, 2.0f * M_PIf, output.data(), knownPhases.size(), 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "CosineF failed for known values";

    // Check specific known values
    EXPECT_NEAR(output[0], 1.0f, 1e-6f) << "cos(0) should be 1";
    EXPECT_NEAR(output[1], 0.0f, 1e-6f) << "cos(π/2) should be 0";
    EXPECT_NEAR(output[2], -1.0f, 1e-6f) << "cos(π) should be -1";
    EXPECT_NEAR(output[3], 0.0f, 1e-6f) << "cos(3π/2) should be 0";
    EXPECT_NEAR(output[4], 1.0f, 1e-6f) << "cos(2π) should be 1";
}

TEST_F(TrigTest, CosineConsistencyTest) {
    auto output1 = std::vector<float>(testSize);
    auto output2 = std::vector<float>(testSize);

    // Run twice with same parameters
    error = gsdrCosineF(0.0f, 2.0f * M_PIf, output1.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "First CosineF failed";

    error = gsdrCosineF(0.0f, 2.0f * M_PIf, output2.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Second CosineF failed";

    // Results should be identical
    for (size_t i = 0; i < testSize; ++i) {
        EXPECT_EQ(output1[i], output2[i]) << "Non-consistent result at index " << i;
    }
}

TEST_F(TrigTest, CosineLargeRangeTest) {
    // Test with large arrays
    const size_t largeSize = 65536; // 64K elements
    auto output = std::vector<float>(largeSize);

    error = gsdrCosineF(0.0f, 2.0f * M_PIf, output.data(), largeSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "CosineF failed for large array";

    // Verify a few random samples
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < 100; ++i) {
        size_t randomIndex = gen() % largeSize;
        float expectedPhase = 2.0f * M_PIf * randomIndex / largeSize;
        float expected = cosf(expectedPhase);
        EXPECT_NEAR(output[randomIndex], expected, 1e-5f)
            << "Incorrect value at random index " << randomIndex;
    }
}

TEST_F(TrigTest, CosinePrecisionTest) {
    // Test precision with specific values
    std::vector<float> precisePhases = {0.0f, M_PIf / 6.0f, M_PIf / 4.0f, M_PIf / 3.0f, M_PIf / 2.0f};
    auto output = std::vector<float>(precisePhases.size());

    error = gsdrCosineF(0.0f, 2.0f * M_PIf, output.data(), precisePhases.size(), 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "CosineF failed for precision test";

    // Check precise values
    EXPECT_NEAR(output[0], 1.0f, 1e-6f);
    EXPECT_NEAR(output[1], cosf(M_PIf / 6.0f), 1e-6f); // √3/2 ≈ 0.866
    EXPECT_NEAR(output[2], cosf(M_PIf / 4.0f), 1e-6f); // √2/2 ≈ 0.707
    EXPECT_NEAR(output[3], 0.5f, 1e-6f);               // cos(π/3) = 0.5
    EXPECT_NEAR(output[4], 0.0f, 1e-6f);
}

TEST_F(TrigTest, CosineNegativePhasesTest) {
    // Test with negative phases
    std::vector<float> negativePhases = {-2.0f * M_PIf, -M_PIf, -M_PIf / 2.0f, 0.0f};
    auto output = std::vector<float>(negativePhases.size());

    error = gsdrCosineF(-2.0f * M_PIf, 2.0f * M_PIf, output.data(), negativePhases.size(), 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "CosineF failed for negative phases";

    // Cosine is even function: cos(-x) = cos(x)
    EXPECT_NEAR(output[0], 1.0f, 1e-6f);  // cos(-2π) = cos(2π) = 1
    EXPECT_NEAR(output[1], -1.0f, 1e-6f); // cos(-π) = cos(π) = -1
    EXPECT_NEAR(output[2], 0.0f, 1e-6f);  // cos(-π/2) = cos(π/2) = 0
    EXPECT_NEAR(output[3], 1.0f, 1e-6f);  // cos(0) = 1
}

TEST_F(TrigTest, CosineComplexLargeTest) {
    const size_t largeSize = 32768; // 32K elements
    auto output = std::vector<cuComplex>(largeSize);

    error = gsdrCosineC(0.0f, 2.0f * M_PIf, output.data(), largeSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "CosineC failed for large array";

    // Verify a few samples
    for (size_t i = 0; i < 10; ++i) {
        float phase = 2.0f * M_PIf * i / largeSize;
        cuComplex expected = make_cuComplex(cosf(phase), sinf(phase));
        EXPECT_NEAR(output[i].x, expected.x, 1e-5f) << "Incorrect real value at index " << i;
        EXPECT_NEAR(output[i].y, expected.y, 1e-5f) << "Incorrect imaginary value at index " << i;
    }
}

TEST_F(TrigTest, CosineComplexUnitCircleTest) {
    const size_t numPoints = 8;
    auto output = std::vector<cuComplex>(numPoints);

    // Generate points around unit circle
    error = gsdrCosineC(0.0f, 2.0f * M_PIf, output.data(), numPoints, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "CosineC failed for unit circle test";

    // All points should be on unit circle
    for (size_t i = 0; i < numPoints; ++i) {
        float magnitude = sqrtf(output[i].x * output[i].x + output[i].y * output[i].y);
        EXPECT_NEAR(magnitude, 1.0f, 1e-5f) << "Point not on unit circle at index " << i;
    }

    // Check specific points
    EXPECT_NEAR(output[0].x, 1.0f, 1e-5f) << "First point should be (1, 0)";
    EXPECT_NEAR(output[0].y, 0.0f, 1e-5f);

    EXPECT_NEAR(output[numPoints/4].x, 0.0f, 1e-5f) << "Quarter point should be (0, 1)";
    EXPECT_NEAR(output[numPoints/4].y, 1.0f, 1e-5f);

    EXPECT_NEAR(output[numPoints/2].x, -1.0f, 1e-5f) << "Half point should be (-1, 0)";
    EXPECT_NEAR(output[numPoints/2].y, 0.0f, 1e-5f);

    EXPECT_NEAR(output[3*numPoints/4].x, 0.0f, 1e-5f) << "Three-quarter point should be (0, -1)";
    EXPECT_NEAR(output[3*numPoints/4].y, -1.0f, 1e-5f);
}