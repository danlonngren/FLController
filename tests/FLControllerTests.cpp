
#include "FLController.h"

#include <gtest/gtest.h>

#include <memory>

// --- Fuzzy Logic Controller Fuzzy Data Tests ---
class FLCFuzzyDataTests : public ::testing::Test {
protected:
    std::shared_ptr<FuzzyDataP> m_fuzzyDataP;
    std::shared_ptr<FuzzyDataI> m_fuzzyDataI;
    std::shared_ptr<FuzzyDataD> m_fuzzyDataD;
    FLCData m_data;

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
    float dt = 1.0f;
    m_data.setLimits(100.0f, 100.0f, 100.0f);
    m_data.set(50.0f, dt);
    float result = m_fuzzyDataP->calculate(m_data, dt);
    EXPECT_FLOAT_EQ(result, 0.5f); // Normalized value should be
}

TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataPLimTest) {
    float dt = 1.0f;
    m_data.setLimits(100.0f, 100.0f, 100.0f);
    m_data.set(200.0f, dt);
    float result = m_fuzzyDataD->calculate(m_data, dt);
    EXPECT_FLOAT_EQ(result, 1.0f); // Normalized value should be
}

TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataITest) {
    float dt = 1.0f;
    m_data.setLimits(100.0f, 100.0f, 100.0f);
    m_data.set(50.0f, dt);
    float result = m_fuzzyDataI->calculate(m_data, dt);
    EXPECT_FLOAT_EQ(result, 0.5f); // Normalized value should be
}

TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataDTest) {
    float dt = 1.0f;
    m_data.setLimits(100.0f, 100.0f, 100.0f);
    m_data.set(50.0f, dt);
    float result = m_fuzzyDataD->calculate(m_data, dt);
    EXPECT_FLOAT_EQ(result, 0.5f); // Normalized value should be
}

// ---// Fuzzy Logic Controller Membership Functions Tests ---
class FLCMembershipFunctionsTests : public ::testing::Test {
protected:
    std::shared_ptr<LinearNMF>  m_linearNMF;
    std::shared_ptr<LinearPMF>  m_linearPMF;
    std::shared_ptr<GaussianMF> m_gaussianMF;
    std::shared_ptr<NonLinearPMF> m_nonLinearPMF;
    std::shared_ptr<NoneLinearNMF> m_noneLinearNMF;

    void SetUp() override {
        m_linearNMF  = std::make_shared<LinearNMF>();
        m_linearPMF  = std::make_shared<LinearPMF>();
        m_gaussianMF = std::make_shared<GaussianMF>();
        m_nonLinearPMF = std::make_shared<NonLinearPMF>();
        m_noneLinearNMF = std::make_shared<NoneLinearNMF>();
    }

    void TearDown() override {
        m_linearNMF.reset();
        m_linearPMF.reset();
        m_gaussianMF.reset();
    }
};

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearNMFPosTest) {
    float input = 0.5f;
    float expectedOutput = 0.25f; // (1.0 - 0.5) / 2.0

    EXPECT_FLOAT_EQ(m_linearNMF->evaluate(input), expectedOutput);
    EXPECT_FLOAT_EQ(m_linearNMF->evaluateAndNormalise(input, 1.0f), expectedOutput);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearNMFNegTest) {
    float input = -0.5f;
    float expectedOutput = 0.75f; // (1.0 - 0.5) / 2.0

    EXPECT_FLOAT_EQ(m_linearNMF->evaluate(input), expectedOutput);
    EXPECT_FLOAT_EQ(m_linearNMF->evaluateAndNormalise(input, 1.0f), expectedOutput);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearPMFPosTest) {
    float input = 0.5f;
    float expectedOutput = 0.75f; // (0.5 + 1.0) / 2.0

    EXPECT_FLOAT_EQ(m_linearPMF->evaluate(input), expectedOutput);
    EXPECT_FLOAT_EQ(m_linearPMF->evaluateAndNormalise(input, 1.0f), expectedOutput);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearPMFNegTest) {
    float input = -0.5f;
    float expectedOutput = 0.25f; // (0.5 + 1.0) / 2.0

    EXPECT_FLOAT_EQ(m_linearPMF->evaluate(input), expectedOutput);
    EXPECT_FLOAT_EQ(m_linearPMF->evaluateAndNormalise(input, 1.0f), expectedOutput);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFTest) {
    float input = 0.0f; // Center of Gaussian
    float expectedOutput = 1.0f; // exp(0) = 1

    EXPECT_FLOAT_EQ(m_gaussianMF->evaluate(input), expectedOutput);
    EXPECT_FLOAT_EQ(m_gaussianMF->evaluateAndNormalise(input, 1.0f), expectedOutput);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFNegativeInputTest) {
    float input = -1.0f; // Negative input
    float expectedOutput = expf(-0.5f); // Gaussian evaluation

    EXPECT_FLOAT_EQ(m_gaussianMF->evaluate(input), expectedOutput);
    EXPECT_FLOAT_EQ(m_gaussianMF->evaluateAndNormalise(input, 1.0f), expectedOutput);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFZeroInputTest) {
    float input = 0.0f; // Zero input
    float expectedOutput = 1.0f; // Gaussian evaluation at zero

    EXPECT_FLOAT_EQ(m_gaussianMF->evaluate(input), expectedOutput);
    EXPECT_FLOAT_EQ(m_gaussianMF->evaluateAndNormalise(input, 1.0f), expectedOutput);
}


// --- FLController Tests ---
class FLControllerTests : public ::testing::Test {
protected:
    std::shared_ptr<FLController> m_controller;

    void SetUp() override {
        std::array<float, 4> weights = {1.0f, 1.0f, 1.0f, 1.0f};
        m_controller = std::make_shared<FLController>(
            200.0f,     // eLim
            200.0f,     // dLim
            200.0f,     // iLim
            10.0f,      // outputGain
            100.0f,    // outputMax
            weights);
    }

    void TearDown() override {
        m_controller.reset();
    }
};

TEST_F(FLControllerTests, FLControllerZeroTest) {
    float output = m_controller->evaluate(50.0f, 50.0f, 1.0f);
    EXPECT_FLOAT_EQ(output, 0.0f); // Adjust expected value based on actual implementation
}

TEST_F(FLControllerTests, FLControllerEvaluatePositiveTest) {
    float output = m_controller->evaluate(0.0f, 100.0f, 1.0f);
    FLCData data = m_controller->getData();
    EXPECT_FLOAT_EQ(output, (0.57477574f * data.getOutputGain())); // Adjust expected value based on actual implementation
}

TEST_F(FLControllerTests, FLControllerEvaluateNegativeTest) {
    float output = m_controller->evaluate(100.0f, 0.0f, 1.0f);
    FLCData data = m_controller->getData();
    EXPECT_FLOAT_EQ(output, -(0.57477574f * data.getOutputGain())); // Adjust expected value based on actual implementation
}

TEST_F(FLControllerTests, FLControllerEvaluateLimitPositiveTest) {
    float output = m_controller->evaluate(50.0f, 2000.0f, 0.1f);
    FLCData data = m_controller->getData();
    EXPECT_FLOAT_EQ(output, (1.0f * data.getOutputGain())); // Adjust expected value based on actual implementation
}
TEST_F(FLControllerTests, FLControllerEvaluateLimitNegativeTest) {
    float output = m_controller->evaluate(2000.0f, 50.0f, 0.1f);
    FLCData data = m_controller->getData();
    EXPECT_FLOAT_EQ(output, -(1.0f * data.getOutputGain())); // Adjust expected value based on actual implementation
}

// --- FLController Tests ---
class FLController2Tests : public ::testing::Test {
protected:
    std::shared_ptr<FLController> m_controller;

    void SetUp() override {
        std::array<float, 4> weights = {1.0f, 1.0f, 1.0f, 1.0f};
        m_controller = std::make_shared<FLController>(
            2000.0f,   // eLim
            2000.0f,   // dLim
            2000.0f,   // iLim
            100.0f,    // outputGain
            100.0f,    // outputMax
            weights);
    }

    void TearDown() override {
        m_controller.reset();
    }
};

TEST_F(FLController2Tests, FLControllerEvaluateLimitNegativeTest) {
    float output = m_controller->evaluate(4000.0f, 50.0f, 0.1f);
    FLCData data = m_controller->getData();
    EXPECT_FLOAT_EQ(output, -(100.0f)); // Adjust expected value based on actual implementation
}