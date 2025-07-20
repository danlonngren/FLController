#include "FLController.h"

#include <memory>
#include <array>
#include <algorithm>
#include <utility>


// --- FLController implementation ---
FLController::FLController( float normalizationMin, float normalizationMax) :
        m_FLCRules(),
        m_fuzzyOutput(0.0f) {
}

float FLController::evaluate() { 
    // --- Defuzzification using CoG method ---
    m_fuzzyOutput = defuzzifyWeightedAvg(m_FLCRules);

	return m_fuzzyOutput;
}

float FLController::defuzzifyWeightedAvg(std::vector<FuzzyRule>& rules) {
    float weightedSum = 0.0f;
    float totalWeight = 0.0f;

    for (const auto& rule : m_FLCRules) {
        auto [output, weight] = rule.evaluate(); // Evaluate the rules
        weightedSum += output;
        totalWeight += weight;
    }

    float output = (totalWeight != 0.0f) ? weightedSum / totalWeight : 0.0f;
    if constexpr (ENABLE_LOGGING)
        LOG("weightedSum: " + toStr(weightedSum) + ", totalWeight: " + toStr(totalWeight) + ", output: " + toStr(output));
    return output;
}

void FLController::setRules(std::vector<FuzzyRule> rules) {
    m_FLCRules = std::move(rules);
} 

void FLController::reset() {
	m_fuzzyOutput = 0.0f;
    m_FLCRules.clear();
}


// --- FLCRule Implementation ---
FuzzyRule::RuleResult FuzzyRule::evaluate() const {
    float a = m_inputA.evaluate();
    float b = m_inputB.evaluate();
    const float membership = m_operator ? m_operator(a, b) : 0.0f;

    float output = 0.0;
    output = m_outputMf ? m_outputMf(membership * m_weight) : 0.0f;

    if constexpr (ENABLE_LOGGING)
        LOG("membership: "+ toStr(membership) + ", output: "+ toStr(output));
    return {output, membership};
}