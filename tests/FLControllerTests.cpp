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
        m_fuzzyDataP = std::make_shared<FuzzyDataBasic>();
        m_fuzzyDataI = std::make_shared<FuzzyDataBasic>();
        m_fuzzyDataD = std::make_shared<FuzzyDataBasic>();
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
        EXPECT_FLOAT_EQ(normaliseVal(data.input, -100.0f, 100.0f), data.expected); // Normalized value should be
    }
}

// TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataPTest) {
//     std::array<TestData, 4> testInput = {
//         TestData(0.0f,   0.0f),
//         TestData(10.0f,  10.0f),
//         TestData(25.0f,  25.0f),
//         TestData(-50.0f,  -50.0f),
//     };

//     for (auto& data : testInput) {
//         float result = m_fuzzyDataP->getData(data.input);
//         EXPECT_FLOAT_EQ(result, data.expected); // Normalized value should be
//     }
// }

// TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataITest) {
//     std::array<TestData, 4> testInput = {
//         TestData(0.0f,   0.0f),
//         TestData(10.0f,  10.0f),
//         TestData(25.0f,  25.0f),
//         TestData(-50.0f,  -50.0f),
//     };

//     for (auto& data : testInput) {
//         m_data.set(0.0f, data.input, 0.0f);
//         float result = m_fuzzyDataI->getData(m_data);
//         EXPECT_FLOAT_EQ(result, data.expected); // Normalized value should be
//     }
// }

// TEST_F(FLCFuzzyDataTests, FLControllerFuzzyDataDTest) {
//     std::array<TestData, 4> testInput = {
//         TestData(0.0f,    0.0f),
//         TestData(10.0f,   10.0f),
//         TestData(25.0f,   25.0f),
//         TestData(-50.0f,   -50.0f),
//     };
    
//     for (auto& data : testInput) {
//         m_data.set(0.0f, 0.0f, data.input);
//         float result = m_fuzzyDataD->getData(m_data);
//         EXPECT_FLOAT_EQ(result, data.expected); // Normalized value should be
//     }
// }

// ---// Fuzzy Logic Controller Membership Functions Tests ---
class FLCMembershipFunctionsTests : public ::testing::Test {
protected:
    std::shared_ptr<LinearCenterNMF>    m_linearNMF;
    std::shared_ptr<LinearCenterPMF>    m_linearPMF;
    std::shared_ptr<NonLinearPMF>       m_nonLinearPMF;
    std::shared_ptr<NoneLinearNMF>      m_nonLinearNMF;
    std::shared_ptr<GaussianMF>         m_gaussianMF;

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
// TEST_F(FLCMembershipFunctionsTests, FLControllerGaussianMFZeroInput2Test) {
//     TestData tData(0.1f, 1.0f);
//     EXPECT_FLOAT_EQ(m_gaussianMF->evaluate(tData.input), tData.expected);
// }

// --- FLController Tests ---
class FLControllerTests : public ::testing::Test {
protected:
    std::shared_ptr<FLController> m_controller;
    std::shared_ptr<FuzzyData> m_pData;
    std::shared_ptr<FuzzyData> m_iData;
    std::shared_ptr<FuzzyData> m_dData;

    void SetUp() override {
        m_pData  = std::make_shared<FuzzyDataBasic>();
        m_iData  = std::make_shared<FuzzyDataBasic>();
        m_dData  = std::make_shared<FuzzyDataBasic>();
        m_controller = std::make_shared<FLController>(
                        -100.0f, 100.0f);
    }

    void TearDown() override {
        m_pData->setData(0.0f);
        m_iData->setData(0.0f);
        m_dData->setData(0.0f);
        m_controller.reset();
    }
};

TEST_F(FLControllerTests, FLControllerEvaluateZeroTest) {
    float weight[4] {1.0f, 1.0f, 1.0f, 1.0f};
    
    FLCSet pPosSet(std::make_shared<LinearCenterPMF>(), m_pData);
    FLCSet pNegSet(std::make_shared<LinearCenterNMF>(), m_pData);
    
    FLCSet dPosSet(std::make_shared<LinearCenterPMF>(), m_dData);
    FLCSet dNegSet(std::make_shared<LinearCenterNMF>(), m_dData);

    FLCSet iPosSet(std::make_shared<LinearCenterPMF>(), m_pData);
    FLCSet iNegSet(std::make_shared<LinearCenterNMF>(), m_iData);

    FLCSet pGausSet(std::make_shared<GaussianMF>(), m_iData);
    FLCSet pGausNegSet(std::make_shared<GaussianMF>(), m_iData);

    auto fuzzyRules = {
        // P+ and P- rules
        FLCRule(pPosSet, dPosSet, FLCRule::PROD, FLCRule::POS, weight[0]), // P+
        FLCRule(pNegSet, dNegSet, FLCRule::PROD, FLCRule::NEG, weight[0]), // P-
        // D+ and D- rules
        FLCRule(dPosSet, pNegSet, FLCRule::PROD, FLCRule::POS, weight[1]), // D+
        FLCRule(dNegSet, pPosSet, FLCRule::PROD, FLCRule::NEG, weight[1]), // D-
        // I+ and I- rules
        FLCRule(pPosSet, iPosSet, FLCRule::PROD, FLCRule::POS, weight[2]), // I+
        FLCRule(pNegSet, iNegSet, FLCRule::PROD, FLCRule::NEG, weight[2]), // I-
        // Gaussian rule for reducing overshoot
        FLCRule(dPosSet, pGausSet, FLCRule::PROD, FLCRule::POS, weight[3]), // G+
        FLCRule(dNegSet, pGausSet, FLCRule::PROD, FLCRule::NEG, weight[3])  // G-
    };

    m_controller->setRules(fuzzyRules);
    auto result = m_controller->evaluate();
    EXPECT_FLOAT_EQ(result, 0.0f); // Adjust expected value based on actual implementation
}

TEST_F(FLControllerTests, FLControllerEvaluatePTest) {
    float weight[4] {1.0f, 1.0f, 1.0f, 1.0f};
    
    FLCSet pPosSet(std::make_shared<LinearCenterPMF>(), m_pData);
    FLCSet pNegSet(std::make_shared<LinearCenterNMF>(), m_pData);
    
    FLCSet dPosSet(std::make_shared<LinearCenterPMF>(), m_dData);
    FLCSet dNegSet(std::make_shared<LinearCenterNMF>(), m_dData);

    auto fuzzyRules = {
        // P+ and P- rules
        FLCRule(pPosSet, dPosSet, FLCRule::PROD, FLCRule::POS, weight[0]), // P+
        FLCRule(pNegSet, dNegSet, FLCRule::PROD, FLCRule::NEG, weight[0]), // P-
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
        m_pData->setData(data.input);
        result = m_controller->evaluate();
        EXPECT_FLOAT_EQ(result, data.expected);
    }
}

TEST_F(FLControllerTests, FLControllerEvaluateITest) {
    float weight[4] {1.0f, 1.0f, 1.0f, 1.0f};
    
    FLCSet pPosSet(std::make_shared<LinearCenterPMF>(), m_pData);
    FLCSet pNegSet(std::make_shared<LinearCenterNMF>(), m_pData);

    FLCSet iPosSet(std::make_shared<LinearCenterPMF>(), m_iData);
    FLCSet iNegSet(std::make_shared<LinearCenterNMF>(), m_iData);

    auto fuzzyRules = {
        // // I+ and I- rules
        FLCRule(pPosSet, iPosSet, FLCRule::PROD, FLCRule::POS, weight[2]), // I+
        FLCRule(pNegSet, iNegSet, FLCRule::PROD, FLCRule::NEG, weight[2]), // I-
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
        m_iData->setData(data.input);
        result = m_controller->evaluate();
        EXPECT_FLOAT_EQ(result, data.expected);
    }
}

TEST_F(FLControllerTests, FLControllerEvaluateDTest) {
    float weight[4] {1.0f, 1.0f, 1.0f, 1.0f};
    
    FLCSet pPosSet(std::make_shared<LinearCenterPMF>(), m_pData);
    FLCSet pNegSet(std::make_shared<LinearCenterNMF>(), m_pData);
    
    FLCSet dPosSet(std::make_shared<LinearCenterPMF>(), m_dData);
    FLCSet dNegSet(std::make_shared<LinearCenterNMF>(), m_dData);

    FLCSet iPosSet(std::make_shared<LinearCenterPMF>(), m_iData);
    FLCSet iNegSet(std::make_shared<LinearCenterNMF>(), m_iData);

    FLCSet pGausSet(std::make_shared<GaussianMF>(), m_iData);
    FLCSet pGausNegSet(std::make_shared<GaussianMF>(), m_iData);

    auto fuzzyRules = {
        // P+ and P- rules
        FLCRule(pPosSet, dPosSet, FLCRule::PROD, FLCRule::POS, weight[0]), // P+
        FLCRule(pNegSet, dNegSet, FLCRule::PROD, FLCRule::NEG, weight[0]), // P-
        // D+ and D- rules
        // FLCRule(dPosSet, pNegSet, FLCRule::PROD, FLCRule::POS, weight[1]), // D+
        // FLCRule(dNegSet, pPosSet, FLCRule::PROD, FLCRule::NEG, weight[1]), // D-
        // // I+ and I- rules
        // FLCRule(pPosSet, iPosSet, FLCRule::PROD, FLCRule::POS, weight[2]), // I+
        // FLCRule(pNegSet, iNegSet, FLCRule::PROD, FLCRule::NEG, weight[2]), // I-
        // // Gaussian rule for reducing overshoot
        // FLCRule(dPosSet, pGausSet, FLCRule::PROD, FLCRule::POS, weight[3]), // G+
        // FLCRule(dNegSet, pGausSet, FLCRule::PROD, FLCRule::NEG, weight[3])  // G-
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
        m_pData->setData(0.5f);
        m_dData->setData(data.input);
        result = m_controller->evaluate();
        EXPECT_FLOAT_EQ(result, data.expected);
    }
}

TEST_F(FLControllerTests, FLControllerEvaluateGTest) {       
    FLCSet dPosSet(std::make_shared<LinearCenterPMF>(), m_dData);
    FLCSet dNegSet(std::make_shared<LinearCenterNMF>(), m_dData);
    FLCSet pGausSet(std::make_shared<GaussianMF>(), m_pData);
    // FLCSet pNGausSet(std::make_shared<GaussianNMF>(), m_pData);

    auto fuzzyRules = {
        // Gaussian rule for reducing overshoot
        FLCRule(dPosSet, pGausSet, FLCRule::SUM, FLCRule::POS, 1.0f), // G+
        FLCRule(dNegSet, pGausSet, FLCRule::SUM, FLCRule::NEG, 1.0f)  // G-
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
        m_pData->setData(data.input);
        m_iData->setData(0.0f);
        m_dData->setData(0.5f);
        result = m_controller->evaluate();
        EXPECT_FLOAT_EQ(result, data.expected);
    }
}

// TEST_F(FLControllerTests, FLControllerEvaluatePIDTest) {
//     float weight[4] {1.3f, 0.3f, 1.0f, 0.3f};
    
//     FLCSet pPosSet(std::make_shared<LinearCenterPMF>(), m_pData);
//     FLCSet pNegSet(std::make_shared<LinearCenterNMF>(), m_pData);
    
//     FLCSet dPosSet(std::make_shared<LinearCenterPMF>(), m_dData);
//     FLCSet dNegSet(std::make_shared<LinearCenterNMF>(), m_dData);

//     FLCSet iPosSet(std::make_shared<LinearCenterPMF>(), m_iData);
//     FLCSet iNegSet(std::make_shared<LinearCenterNMF>(), m_iData);

//     FLCSet pGausSet(std::make_shared<GaussianMF>(), m_iData);

//     auto fuzzyRules = {
//         // P+ and P- rules
//         FLCRule(pPosSet, dPosSet, FLCRule::PROD, FLCRule::POS, weight[0]), // P+
//         FLCRule(pNegSet, dNegSet, FLCRule::PROD, FLCRule::NEG, weight[0]), // P-
//         // D+ and D- rules
//         FLCRule(dPosSet, pNegSet, FLCRule::PROD, FLCRule::POS, weight[1]), // D+
//         FLCRule(dNegSet, pPosSet, FLCRule::PROD, FLCRule::NEG, weight[1]), // D-
//         // I+ and I- rules
//         FLCRule(pPosSet, iPosSet, FLCRule::PROD, FLCRule::POS, weight[2]), // I+
//         FLCRule(pNegSet, iNegSet, FLCRule::PROD, FLCRule::NEG, weight[2]), // I-
//         // Gaussian rule for reducing overshoot
//         FLCRule(dPosSet, pGausSet, FLCRule::SUM, FLCRule::POS, weight[3]), // G+
//         FLCRule(dNegSet, pGausSet, FLCRule::SUM, FLCRule::NEG, weight[3])  // G-
//     };
//     m_controller->setRules(fuzzyRules);

//     float result = 0.0f;

//     std::array<TestDataPID, 2> tData = {
//         TestDataPID(0.1f, 0.0f, 0.0f, 10.0f),
//         TestDataPID(0.2f, 0.0f, 0.0f, 10.0f)
//     };
    
//     for (const auto& data : tData) {
//         m_pData->setData(data.p);
//         m_iData->setData(data.i);
//         m_dData->setData(data.d);
//         result = m_controller->evaluate();
//         EXPECT_FLOAT_EQ(result, data.expected);
//     }
// }