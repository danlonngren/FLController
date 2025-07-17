#include "FLController.h"

#include <math.h>
#include <memory>
#include <array>
#include <algorithm>
#include <utility>

FLController::FLController( float normalizationMin,
					        float normalizationMax,
                            float outputGain, 
                            std::array<float, 4> weights) :
        m_normMin(normalizationMin),
        m_normMax(normalizationMax),
        m_outputGain(outputGain),
        m_w(weights),
        m_fuzzyData(), 
        m_fcRules(),
        m_FLCRules() {
    m_fuzzyData.reset();
    m_fuzzyData.setMinMax(m_normMin, m_normMax);
    std::cout << std::fixed << std::setprecision(3);

    // m_w[0] = 1.0;
    // m_w[1] = 0.35;
    // m_w[2] = 1.0;
    // m_w[3] = 0.7;

    auto negLMF = std::make_shared<LinearNMF>();
    auto posLMF = std::make_shared<LinearPMF>();    
    auto gauMF  = std::make_shared<GaussianMF>();
    
    auto pDada  = std::make_shared<FuzzyDataP>();
    auto IData  = std::make_shared<FuzzyDataI>();
    auto dData  = std::make_shared<FuzzyDataD>();

    auto prodOperator = std::make_shared<FLCProductOperator>();

    float outGain = 1.0f;

    m_FLCRules = {
        // P+ and P- rules
        FLCRule(FLCSet(posLMF, pDada), FLCSet(posLMF, dData), prodOperator, m_w[0], outGain, FLCRule::POSITIVE), // P+
        FLCRule(FLCSet(negLMF, pDada), FLCSet(negLMF, dData), prodOperator, m_w[0], outGain, FLCRule::NEGATIVE), // P-
        // D+ and D- rules
        FLCRule(FLCSet(posLMF, dData), FLCSet(negLMF, pDada), prodOperator, m_w[1], outGain, FLCRule::POSITIVE), // D+
        FLCRule(FLCSet(negLMF, dData), FLCSet(posLMF, pDada), prodOperator, m_w[1], outGain, FLCRule::NEGATIVE), // D-
        // I+ and I- rules
        FLCRule(FLCSet(posLMF, pDada), FLCSet(posLMF, IData), prodOperator, m_w[2], outGain, FLCRule::POSITIVE), // I+
        FLCRule(FLCSet(negLMF, pDada), FLCSet(negLMF, IData), prodOperator, m_w[2], outGain, FLCRule::NEGATIVE), // I-
        // Gaussian rule for reducing overshoot
        FLCRule(FLCSet(posLMF, dData), FLCSet(gauMF, pDada), prodOperator, m_w[3], outGain, FLCRule::POSITIVE), // G+
        FLCRule(FLCSet(negLMF, dData), FLCSet(gauMF, pDada), prodOperator, m_w[3], outGain, FLCRule::NEGATIVE), // G-
    };
}

void FLController::reset() {
	m_fuzzyData.reset();
}

float FLController::evaluate(float input, float setpoint, float dt) { 
    std::cout << "input: " << input << ", setpoint: " << setpoint << std::endl;
    
    // --- Compute error terms ---
	m_fuzzyData.set(setpoint - input, dt, m_normMax,  m_normMin);

    // --- Update fuzzy rules or weights if need ---

    // --- Defuzzification using CoG method---
    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (const auto& rule : m_FLCRules) {
        auto [n, d] = rule.evaluate(m_fuzzyData, dt); // Evaluate the rule with normalized error and derivative error
        numerator += n * d;
        denominator += d;
    }
    
    // --- Normalize output ---
    float output = (denominator != 0.0f) ? numerator / denominator : 0.0f;
    std::cout << "Numerator: " << numerator << ", Denominator: " << denominator << ", out: " << output << std::endl;
    m_fuzzyData.fuzzyOutput = output;

	return m_fuzzyData.fuzzyOutput;
}

void FLController::setLimits(float eLim, float dLim, float iLim, float outputGain) {
    m_outputGain = outputGain;
}

void FLController::setRules(const std::vector<FLCRule>& rules) {
    m_FLCRules = rules;
} 