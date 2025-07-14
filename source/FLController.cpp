#include "FLController.h"

#include <math.h>
#include <memory>
#include <array>
#include <algorithm>
#include <utility>

FLController::FLController(float eLim, 
                            float dLim, 
                            float iLim, 
                            float outputGain, 
                            float outputMax,
                            std::array<float, 4> weights) :
        m_fuzzyData(), 
        m_fcRules(), 
        m_w(weights) {
    m_fuzzyData.reset();
    m_fuzzyData.setLimits(eLim, iLim, dLim);
    m_fuzzyData.setOutputGain(outputGain);
    m_fuzzyData.setMaxOutput(outputMax);

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
    
    m_fcRules = {
        FLCRuleSet(posLMF, pDada, posLMF, dData, m_w[0],  1.0f), // P+ rule
        FLCRuleSet(negLMF, pDada, negLMF, dData, m_w[0], -1.0f), // P- rule

        FLCRuleSet(posLMF, dData, negLMF, pDada, m_w[1],  1.0f), // D+ rule
        FLCRuleSet(negLMF, dData, posLMF, pDada, m_w[1], -1.0f), // D- rule

        FLCRuleSet(posLMF, pDada, posLMF, IData, m_w[2],  1.0f), // I+ rule
        FLCRuleSet(negLMF, pDada, negLMF, IData, m_w[2], -1.0f), // I- rule

        FLCRuleSet(posLMF, dData, gauMF,  pDada, m_w[3],  1.0f), // Gaussian rule for reducing overshoot
        FLCRuleSet(negLMF, dData, gauMF,  pDada, m_w[3], -1.0f)  // Gaussian rule for reducing overshoot
    };
}

void FLController::reset() {
	m_fuzzyData.reset();
}

float FLController::evaluate(float input, float setpoint, float dt) {   
    // --- Compute error terms ---
	m_fuzzyData.set(setpoint - input, dt);

    // --- Update fuzzy rules or weights if need ---

    // --- Defuzzification ---
    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (const auto& rule : m_fcRules) {
        auto [n, d] = rule.evaluate(m_fuzzyData, dt); // Evaluate the rule with normalized error and derivative error
        numerator += d;
        denominator += n;
    }

    std::cout << "Numerator: " << numerator << ", Denominator: " << denominator << std::endl;

    // --- Normalize output ---
    float output = (denominator != 0.0f) ? numerator / denominator : 0.0f;
    output *= m_fuzzyData.getOutputGain();
    output = clamp(output, -m_fuzzyData.getMaxOutput(), m_fuzzyData.getMaxOutput());
    m_fuzzyData.fuzzyOutput = output;

	return m_fuzzyData.fuzzyOutput;
}

void FLController::setLimits(float eLim, float dLim, float iLim, float outputGain, float outputMax) {
    m_fuzzyData.setLimits(eLim, iLim, dLim);
	m_fuzzyData.setOutputGain(outputGain);
    m_fuzzyData.setMaxOutput(outputMax);
}

void FLController::setOutputMax(float outputMax) {
    m_fuzzyData.setMaxOutput(outputMax);
}

void FLController::setRules(const std::vector<FLCRuleSet>& rules) {
    m_fcRules = rules;
}  

float FLController::normalize(float x, float limit) const {
    return clamp(x / std::abs(limit), -1.0f, 1.0f);
}