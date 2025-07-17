
#include "FLController.h"

#include <gtest/gtest.h>

#include <memory>

// --- Common data type for test
struct TestData {
    TestData(float a, float b) 
        : input(a), expected(b) {}
    float input, expected; 
};

// --- Fuzzy Logic Controller Fuzzy Data Tests ---
class FLCFuzzyDataTests : public ::testing::Test {
protected:
    std::shared_ptr<FuzzyDataP> m_fuzzyDataP;
    std::shared_ptr<FuzzyDataI> m_fuzzyDataI;
    std::shared_ptr<FuzzyDataD> m_fuzzyDataD;
    FLCData m_data;


    float m_dt = 1.0f;

    void SetUp() override {
        m_fuzzyDataP = std::make_shared<FuzzyDataP>();
        m_fuzzyDataI = std::make_shared<FuzzyDataI>();
        m_fuzzyDataD = std::make_shared<FuzzyDataD>();
    }

    void TearDown() override {
        m_data.reset();
        m_fuzzyDataD.reset();
        m_fuzzyDataI.reset();
        m_fuzzyDataP.reset();
    }
};

TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataPTest) {
    std::array<TestData, 8> testInput = {
        TestData(0.0f,    0.0f),
        TestData(25.0f,   0.25f),
        TestData(60.0f,   0.6f),
        TestData(80.0f,   0.8f),
        TestData(100.0f,  1.0f),
        TestData(110.0f,  1.0f),
        TestData(-10.0f, -0.1f),
        TestData(-50.0f, -0.5f),
    };

    for (auto& d : testInput) {
        m_data.set(d.input, m_dt, -100, 100);
        float result = m_fuzzyDataP->calculate(m_data, m_dt);
        EXPECT_FLOAT_EQ(result, d.expected); // Normalized value should be
    }
}

TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataITest) {
    std::array<TestData, 7> testInput = {
        TestData(0.0f,   0.0f),
        TestData(10.0f,  0.1f),
        TestData(25.0f,  0.34999999f),
        TestData(50.0f,  0.85000002f),
        TestData(20.0f,  1.0f),
        TestData(-30.0f, 0.75f),
        TestData(-10.0f, 0.64999998f),
    };

    for (auto& d : testInput) {
        m_data.set(d.input, m_dt, -100, 100);
        float result = m_fuzzyDataI->calculate(m_data, m_dt);
        EXPECT_FLOAT_EQ(result, d.expected); // Normalized value should be
    }
}

TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataDTest) {
    std::array<TestData, 7> testInput = {
        TestData(0.0f,    0.0f),
        TestData(10.0f,   0.1f),
        TestData(25.0f,   0.15f),
        TestData(50.0f,   0.25f),
        TestData(20.0f,  -0.3f),
        TestData(-30.0f, -0.5f),
        TestData(-10.0f,  0.2f),
    };
    
    for (auto& d : testInput) {
        m_data.set(d.input, m_dt, -100, 100);
        float result = m_fuzzyDataD->calculate(m_data, m_dt);
        EXPECT_FLOAT_EQ(result, d.expected); // Normalized value should be
    }
}

// ---// Fuzzy Logic Controller Membership Functions Tests ---
class FLCMembershipFunctionsTests : public ::testing::Test {
protected:
    std::shared_ptr<LinearNMF>  m_linearNMF;
    std::shared_ptr<LinearPMF>  m_linearPMF;
    std::shared_ptr<GaussianMF> m_gaussianMF;
    std::shared_ptr<NonLinearPMF> m_nonLinearPMF;
    std::shared_ptr<NoneLinearNMF> m_nonLinearNMF;

    void SetUp() override {
        m_linearNMF  = std::make_shared<LinearNMF>();
        m_linearPMF  = std::make_shared<LinearPMF>();
        m_gaussianMF = std::make_shared<GaussianMF>();
        m_nonLinearPMF = std::make_shared<NonLinearPMF>();
        m_nonLinearNMF = std::make_shared<NoneLinearNMF>();
    }

    void TearDown() override {
        m_linearNMF.reset();
        m_linearPMF.reset();
        m_gaussianMF.reset();
        m_nonLinearPMF.reset();
        m_nonLinearNMF.reset();
    }
};

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearNMFPosTest) {
    TestData tData(0.5f, 0.25f);
    EXPECT_FLOAT_EQ(m_linearNMF->evaluate(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearNMFNegTest) {
    TestData tData(-0.5f, 0.75f);
    EXPECT_FLOAT_EQ(m_linearNMF->evaluate(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearPMFPosTest) {
    TestData tData(0.5f, 0.75f);
    EXPECT_FLOAT_EQ(m_linearPMF->evaluate(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearPMFNegTest) {
    TestData tData(-0.5f, 0.25f);
    EXPECT_FLOAT_EQ(m_linearPMF->evaluate(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFTest) {
    TestData tData(0.0f, 1.0f);
    EXPECT_FLOAT_EQ(m_gaussianMF->evaluate(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFNegativeInputTest) {
    TestData tData(-1.0f, expf(-0.5f));
    EXPECT_FLOAT_EQ(m_gaussianMF->evaluate(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFZeroInputTest) {
    TestData tData(0.0f, 1.0f);
    EXPECT_FLOAT_EQ(m_gaussianMF->evaluate(tData.input), tData.expected);
}


// --- FLController Tests ---
class FLControllerTests : public ::testing::Test {
protected:
    std::shared_ptr<FLController> m_controller;

    void SetUp() override {
        std::array<float, 4> weights = {1.0f, 1.0f, 1.0f, 1.0f};
        m_controller = std::make_shared<FLController>(
            -100.0f,     // min
            100.0f,     // min
            1.0f,       // outputGain
            weights);
    }

    void TearDown() override {
        m_controller.reset();
    }
};

TEST_F(FLControllerTests, FLControllerEvaluateZeroTest) {
    float output = m_controller->evaluate(0.0f, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(output, (0.0f)); // Adjust expected value based on actual implementation
}

// TEST_F(FLControllerTests, FLControllerEvaluateTest) {
//     float setPoint = 50.0f;
//     std::array<TestData, 7> testInput = {
//         TestData(0.0f,  0.0f),
//         TestData(10.0f,  0.1f),
//         TestData(15.0f,  0.15f),
//         TestData(30.0f,  0.25f),
//         TestData(40.0f, -0.3f),
//         TestData(50.0f, -0.5f),
//         TestData(60.0f,  0.2f),
//     };

//     for (auto& data : testInput) {
//         float output = m_controller->evaluate(data.input, setPoint, 1.0f);
//         EXPECT_FLOAT_EQ(output, data.expected); // Adjust expected value based on actual implementation
//     }
// }

TEST_F(FLControllerTests, FLControllerEvaluateNegativeTest) {
    float output = m_controller->evaluate(100.0f, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(output, 0.4990); // Adjust expected value based on actual implementation
}

TEST_F(FLControllerTests, FLControllerEvaluateLimitPositiveTest) {
    float output = m_controller->evaluate(0.0f, 100.0f, 0.1f);
    EXPECT_FLOAT_EQ(output, -(0.4990)); // Adjust expected value based on actual implementation
}

// TEST_F(FLControllerTests, FLControllerEvaluateLimitNegativeTest) {
//     float output = m_controller->evaluate(2000.0f, 50.0f, 0.1f);
//     EXPECT_FLOAT_EQ(output, -(1.0f * m_controller->getOutputGain())); // Adjust expected value based on actual implementation
// }