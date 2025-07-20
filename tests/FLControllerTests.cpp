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
struct TestDataPID {
    TestDataPID(float _p, float _i, float _d, float b) 
        : p(_p), i(_i), d(_d), expected(b) {}
    float p, i, d, expected; 
};

// --- Fuzzy Logic Controller Fuzzy Data Tests ---
class FLCFuzzyDataTests : public ::testing::Test {
protected:
    std::shared_ptr<FuzzyData> m_fuzzyDataP;
    std::shared_ptr<FuzzyData> m_fuzzyDataI;
    std::shared_ptr<FuzzyData> m_fuzzyDataD;

    float m_dt = 1.0f;

    void SetUp() override {
        m_fuzzyDataP = std::make_shared<FuzzyData>();
        m_fuzzyDataI = std::make_shared<FuzzyData>();
        m_fuzzyDataD = std::make_shared<FuzzyData>();
    }

    void TearDown() override {
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
        EXPECT_FLOAT_EQ(normalizeToMinus1To1(data.input, -100.0f, 100.0f), data.expected); // Normalized value should be
    }
}

// ---// Fuzzy Logic Controller Membership Functions Tests ---
class FLCMembershipFunctionsTests : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearNMFPosTest) {
    TestData tData(0.5f, 0.0f);
    EXPECT_FLOAT_EQ(FuzzyMF::LinearNMF(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearNMFNegTest) {
    TestData tData(-0.5f, 0.5f);
    EXPECT_FLOAT_EQ(FuzzyMF::LinearNMF(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearPMFPosTest) {
    TestData tData(0.5f, 0.5f);
    EXPECT_FLOAT_EQ(FuzzyMF::LinearPMF(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerLinearPMFNegTest) {
    TestData tData(-0.5f, 0.0f);
    EXPECT_FLOAT_EQ(FuzzyMF::LinearPMF(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFTest) {
    TestData tData(0.0f, 1.0f);
    EXPECT_FLOAT_EQ(FuzzyMF::GaussianMF(tData.input), tData.expected);
}

TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFZeroInputTest) {
    TestData tData(0.0f, 1.0f);
    EXPECT_FLOAT_EQ(FuzzyMF::GaussianMF(tData.input), tData.expected);
}

// --- FLController Tests ---
class FLControllerTests : public ::testing::Test {
protected:
    std::shared_ptr<FLController> m_controller;
    FuzzyData m_pData;
    FuzzyData m_iData;
    FuzzyData m_dData;

    void SetUp() override {
        m_controller = std::make_shared<FLController>(
                        -100.0f, 100.0f);
    }

    void TearDown() override {
        m_pData.setData(0.0f);
        m_iData.setData(0.0f);
        m_dData.setData(0.0f);
        m_controller.reset();
    }

};

TEST_F(FLControllerTests, FLControllerEvaluateZeroTest) {
    float weight[4] {1.0f, 1.0f, 1.0f, 1.0f};
    
    FuzzyCondition  pPosSet(FuzzyMF::LinearCenterPMF, m_pData);
    FuzzyCondition  pNegSet(FuzzyMF::LinearCenterNMF, m_pData);
    
    FuzzyCondition  dPosSet(FuzzyMF::LinearCenterPMF, m_dData);
    FuzzyCondition  dNegSet(FuzzyMF::LinearCenterNMF, m_dData);

    FuzzyCondition  iPosSet(FuzzyMF::LinearCenterPMF, m_pData);
    FuzzyCondition  iNegSet(FuzzyMF::LinearCenterNMF, m_iData);

    FuzzyCondition  pGausSet(FuzzyMF::GaussianMF, m_iData);
    FuzzyCondition  pGausNegSet(FuzzyMF::GaussianMF, m_iData);

    std::vector<FuzzyRule> fuzzyRules = {
        // P+ and P- rules
        FuzzyRule(pPosSet, dPosSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[0]), // P+
        FuzzyRule(pNegSet, dNegSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[0]), // P-
        // D+ and D- rules
        FuzzyRule(dPosSet, pNegSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[1]), // D+
        FuzzyRule(dNegSet, pPosSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[1]), // D-
        // I+ and I- rules
        FuzzyRule(pPosSet, iPosSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[2]), // I+
        FuzzyRule(pNegSet, iNegSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[2]), // I-
        // Gaussian rule for reducing overshoot
        FuzzyRule(dPosSet, pGausSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[3]), // G+
        FuzzyRule(dNegSet, pGausSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[3])  // G-
    };

    m_controller->setRules(fuzzyRules);
    auto result = m_controller->evaluate();
    EXPECT_FLOAT_EQ(result, 0.0f); // Adjust expected value based on actual implementation
}

TEST_F(FLControllerTests, FLControllerEvaluatePTest) {
    float weight[4] {1.0f, 1.0f, 1.0f, 1.0f};
    
    FuzzyCondition  pPosSet(FuzzyMF::LinearCenterPMF, m_pData);
    FuzzyCondition  pNegSet(FuzzyMF::LinearCenterNMF, m_pData);
    
    FuzzyCondition  dPosSet(FuzzyMF::LinearCenterPMF, m_dData);
    FuzzyCondition  dNegSet(FuzzyMF::LinearCenterNMF, m_dData);

    std::vector<FuzzyRule> fuzzyRules = {
        // P+ and P- rules
        FuzzyRule(pPosSet, dPosSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[0]), // P+
        FuzzyRule(pNegSet, dNegSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[0]), // P-
    };
    m_controller->setRules(fuzzyRules);

    float result = 0.0f;

    std::array<TestData, 4> testInput = {
        TestData(0.2f,   0.2f),
        TestData(-0.2f, -0.2f),
        TestData(1.0f,   1.0f),
        TestData(-1.0f, -1.0f),
    };

    for (const auto& data : testInput) {
        m_pData.setData(data.input);
        result = m_controller->evaluate();
        EXPECT_FLOAT_EQ(result, data.expected);
    }
}

TEST_F(FLControllerTests, FLControllerEvaluateITest) {
    float weight[4] {1.0f, 1.0f, 1.0f, 1.0f};
    
    FuzzyCondition  pPosSet(FuzzyMF::LinearCenterPMF, m_pData);
    FuzzyCondition  pNegSet(FuzzyMF::LinearCenterNMF, m_pData);

    FuzzyCondition  iPosSet(FuzzyMF::LinearCenterPMF, m_iData);
    FuzzyCondition  iNegSet(FuzzyMF::LinearCenterNMF, m_iData);

    std::vector<FuzzyRule> fuzzyRules = {
        // // I+ and I- rules
        FuzzyRule(pPosSet, iPosSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[2]), // I+
        FuzzyRule(pNegSet, iNegSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[2]), // I-
    };
    m_controller->setRules(fuzzyRules);

    float result = 0.0f;

    std::array<TestData, 4> testInput = {
        TestData(0.2f,   0.2f),
        TestData(-0.2f, -0.2f),
        TestData(1.0f,   1.0f),
        TestData(-1.0f, -1.0f),
    };

    
    for (const auto& data : testInput) {
        m_iData.setData(data.input);
        result = m_controller->evaluate();
        EXPECT_FLOAT_EQ(result, data.expected);
    }
}

TEST_F(FLControllerTests, FLControllerEvaluateDTest) {
    float weight[4] {1.0f, 1.0f, 1.0f, 1.0f};
    
    FuzzyCondition  pPosSet(FuzzyMF::LinearCenterPMF, m_pData);
    FuzzyCondition  pNegSet(FuzzyMF::LinearCenterNMF, m_pData);
    
    FuzzyCondition  dPosSet(FuzzyMF::LinearCenterPMF, m_dData);
    FuzzyCondition  dNegSet(FuzzyMF::LinearCenterNMF, m_dData);


    std::vector<FuzzyRule> fuzzyRules = {
        // P+ and P- rules
        FuzzyRule(pPosSet, dPosSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[0]), // P+
        FuzzyRule(pNegSet, dNegSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[0]), // P-
    };
    m_controller->setRules(fuzzyRules);

    float result = 0.0f;

    std::array<TestData, 4> testInput = {
        TestData(0.2f,   0.63636369f),
        TestData(-0.2f,  0.33333334f),
        TestData(1.0f,   1.0f),
        TestData(-1.0f, -1.0f),
    };

    
    for (const auto& data : testInput) {
        m_pData.setData(0.5f);
        m_dData.setData(data.input);
        result = m_controller->evaluate();
        EXPECT_FLOAT_EQ(result, data.expected);
    }
}

TEST_F(FLControllerTests, FLControllerEvaluateGTest) {
 
    FuzzyCondition  dPosSet(FuzzyMF::LinearCenterPMF, m_dData);
    FuzzyCondition  dNegSet(FuzzyMF::LinearCenterNMF, m_dData);
    FuzzyCondition  pGausSet(FuzzyMF::GaussianMF, m_pData);

    std::vector<FuzzyRule> fuzzyRules = {
        // Gaussian rule for reducing overshoot
        FuzzyRule(dPosSet, pGausSet, FuzzyOps::Sum, FuzzyMF::OutputPos, 1.0f), // G+
        FuzzyRule(dNegSet, pGausSet, FuzzyOps::Sum, FuzzyMF::OutputNeg, 1.0f)  // G-
    };
    m_controller->setRules(fuzzyRules);

    std::array<TestData, 4> testInput = {
        TestData(0.1f,  0.17289558f),
        TestData(0.2f,  0.19219868f),
        TestData(0.3f,  0.22593138f),
        TestData(0.4f,  0.27438989f)
    };
    
    float result = 0.0f;
    for (const auto& data : testInput) {
        m_pData.setData(data.input);
        m_iData.setData(0.0f);
        m_dData.setData(0.5f);
        result = m_controller->evaluate();
        EXPECT_FLOAT_EQ(result, data.expected);
    }
}

TEST_F(FLControllerTests, FLControllerEvaluatePIDTest) {
    float weight[4] {1.3f, 0.3f, 0.7f, 1.0f};
    
    FuzzyCondition  pPosSet(FuzzyMF::LinearCenterPMF, m_pData);
    FuzzyCondition  pNegSet(FuzzyMF::LinearCenterNMF, m_pData);
    
    FuzzyCondition  dPosSet(FuzzyMF::LinearCenterPMF, m_dData);
    FuzzyCondition  dNegSet(FuzzyMF::LinearCenterNMF, m_dData);

    FuzzyCondition  iPosSet(FuzzyMF::LinearCenterPMF, m_iData);
    FuzzyCondition  iNegSet(FuzzyMF::LinearCenterNMF, m_iData);

    FuzzyCondition  pGausSet(FuzzyMF::GaussianMF, m_iData);

    std::vector<FuzzyRule> fuzzyRules = {
        // P+ and P- rules
        FuzzyRule(pPosSet, dPosSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[0]), // P+
        FuzzyRule(pNegSet, dNegSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[0]), // P-
        // D+ and D- rules
        FuzzyRule(dPosSet, pNegSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[1]), // D+
        FuzzyRule(dNegSet, pPosSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[1]), // D-
        // I+ and I- rules
        FuzzyRule(pPosSet, iPosSet, FuzzyOps::Product, FuzzyMF::OutputPos, weight[2]), // I+
        FuzzyRule(pNegSet, iNegSet, FuzzyOps::Product, FuzzyMF::OutputNeg, weight[2]), // I-
        // Gaussian rule for reducing overshoot
        FuzzyRule(dPosSet, pGausSet, FuzzyOps::Sum, FuzzyMF::OutputPos, weight[3]), // G+
        FuzzyRule(dNegSet, pGausSet, FuzzyOps::Sum, FuzzyMF::OutputNeg, weight[3])  // G-
    };
    m_controller->setRules(fuzzyRules);

    std::array<TestData, 11> testInput = {
        TestData(100.0f,  18.894609f),
        TestData(90.0f,   13.012421f),
        TestData(80.0f,   11.12964f),
        TestData(70.0f,   9.2464838f),
        TestData(60.0f,   7.3627725f),
        TestData(50.0f,   5.4783454f),
        TestData(40.0f,   3.5930924f),
        TestData(30.0f,   1.7069294f),
        TestData(20.0f,   -0.1802146f),
        TestData(10.0f,   -2.0683622f),
        TestData(0.0f,    -3.9575171f),
    };
    
    float delta = 100.0f;
    float i = 0.0f;
    float d = 0.0f; 
    float dt = 0.001f;
    for (const auto& data : testInput) {
        i += data.input * dt;
        d = data.input - delta;
        delta = data.input;

        m_pData.setData(normalizeToMinus1To1(data.input, -100.0f, 100.0f));
        m_iData.setData(normalizeToMinus1To1(i, -100.0f, 100.0f));
        m_dData.setData(normalizeToMinus1To1(d, -100.0f, 100.0f));
        float result = m_controller->evaluate();
        EXPECT_FLOAT_EQ(result * 100.0f, data.expected);
    }
}