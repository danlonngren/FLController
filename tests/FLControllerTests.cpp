#include <gtest/gtest.h>

#include <memory>

#include "FLController.h"

using namespace std;

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

TEST_F(FLCFuzzyDataTests, FLControllerNormaliseTest) {
    std::array<TestData, 4> testInput = {
        TestData(0.0f,   0.0f),
        TestData(10.0f,  0.10f),
        TestData(25.0f,  0.25f),
        TestData(-50.0f,  -0.50f),
    };

    for (auto& data : testInput) {
        EXPECT_FLOAT_EQ(normaliseVal(data.input, -100.0f, 100.0f), data.expected); // Normalized value should be
    }
}

TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataPTest) {
    std::array<TestData, 4> testInput = {
        TestData(0.0f,   0.0f),
        TestData(10.0f,  10.0f),
        TestData(25.0f,  25.0f),
        TestData(-50.0f,  -50.0f),
    };

    for (auto& data : testInput) {
        m_data.set(data.input, 0.0f, 0.0f, -100.0f, 100.0f);
        float result = m_fuzzyDataP->getData(m_data);
        EXPECT_FLOAT_EQ(result, data.expected); // Normalized value should be
    }
}

TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataITest) {
    std::array<TestData, 4> testInput = {
        TestData(0.0f,   0.0f),
        TestData(10.0f,  10.0f),
        TestData(25.0f,  25.0f),
        TestData(-50.0f,  -50.0f),
    };

    for (auto& data : testInput) {
        m_data.set(0.0f, data.input, 0.0f, -100, 100);
        float result = m_fuzzyDataI->getData(m_data);
        EXPECT_FLOAT_EQ(result, data.expected); // Normalized value should be
    }
}

TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataDTest) {
    std::array<TestData, 4> testInput = {
        TestData(0.0f,    0.0f),
        TestData(10.0f,   10.0f),
        TestData(25.0f,   25.0f),
        TestData(-50.0f,   -50.0f),
    };
    
    for (auto& data : testInput) {
        m_data.set(0.0f, 0.0f, data.input, -100, 100);
        float result = m_fuzzyDataD->getData(m_data);
        EXPECT_FLOAT_EQ(result, data.expected); // Normalized value should be
    }
}

// ---// Fuzzy Logic Controller Membership Functions Tests ---
class FLCMembershipFunctionsTests : public ::testing::Test {
protected:
    std::shared_ptr<LinearCenterNMF>    m_linearNMF;
    std::shared_ptr<LinearCenterPMF>    m_linearPMF;
    std::shared_ptr<GaussianMF>         m_gaussianMF;
    std::shared_ptr<NonLinearPMF>       m_nonLinearPMF;
    std::shared_ptr<NoneLinearNMF>      m_nonLinearNMF;

    void SetUp() override {
        m_linearNMF     = std::make_shared<LinearCenterNMF>();
        m_linearPMF     = std::make_shared<LinearCenterPMF>();
        m_gaussianMF    = std::make_shared<GaussianMF>();
        m_nonLinearPMF  = std::make_shared<NonLinearPMF>();
        m_nonLinearNMF  = std::make_shared<NoneLinearNMF>();
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

TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFZeroInputTest) {
    TestData tData(0.0f, 1.0f);
    EXPECT_FLOAT_EQ(m_gaussianMF->evaluate(tData.input), tData.expected);
}

// --- FLController Tests ---
class FLControllerTests : public ::testing::Test {
protected:
    std::shared_ptr<FLController> m_controller;

    void SetUp() override {
        m_controller = std::make_shared<FLController>(
                        -100.0f, 100.0f);
    }

    void TearDown() override {
        m_controller.reset();
    }
};

TEST_F(FLControllerTests, FLControllerEvaluateZeroTest) {
    float weight[4] {1.0f, 1.0f, 1.0f, 1.0f};
    
    auto pDada  = std::make_shared<FuzzyDataP>();
    auto IData  = std::make_shared<FuzzyDataI>();
    auto dData  = std::make_shared<FuzzyDataD>();
    auto prodOperator = std::make_shared<FLCProdOperator>();

    FLCSet pPosSet(std::make_shared<LinearCenterPMF>(), pDada);
    FLCSet pNegSet(std::make_shared<LinearCenterNMF>(), pDada);
    
    FLCSet dPosSet(std::make_shared<LinearCenterPMF>(), dData);
    FLCSet dNegSet(std::make_shared<LinearCenterNMF>(), dData);

    FLCSet iPosSet(std::make_shared<LinearCenterPMF>(), IData);
    FLCSet iNegSet(std::make_shared<LinearCenterNMF>(), IData);

    FLCSet pGausSet(std::make_shared<GaussianMF>(), IData);
    FLCSet pGausNegSet(std::make_shared<GaussianMF>(), IData);

    auto fuzzyRules = {
        // P+ and P- rules
        FLCRule(pPosSet, dPosSet, prodOperator, FLCRule::POS, weight[0]), // P+
        FLCRule(pNegSet, dNegSet, prodOperator, FLCRule::NEG, weight[0]), // P-
        // D+ and D- rules
        FLCRule(dPosSet, pNegSet, prodOperator, FLCRule::POS, weight[1]), // D+
        FLCRule(dNegSet, pPosSet, prodOperator, FLCRule::NEG, weight[1]), // D-
        // I+ and I- rules
        FLCRule(pPosSet, iPosSet, prodOperator, FLCRule::POS, weight[2]), // I+
        FLCRule(pNegSet, iNegSet, prodOperator, FLCRule::NEG, weight[2]), // I-
        // Gaussian rule for reducing overshoot
        FLCRule(dPosSet, pGausSet, prodOperator, FLCRule::POS, weight[3]), // G+
        FLCRule(dNegSet, pGausSet, prodOperator, FLCRule::NEG, weight[3])  // G-
    };

    m_controller->setRules(fuzzyRules);
    auto result = m_controller->evaluate(0.1f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(result, 0.0f); // Adjust expected value based on actual implementation
}