#include "FLController.h"

#include <math.h>
#include <memory>
#include <array>
#include <algorithm>
#include <utility>


// --- FLController implementation ---
FLController::FLController( float normalizationMin, float normalizationMax) :
        m_normMin(normalizationMin),
        m_normMax(normalizationMax),
        m_fuzzyOutput(0.0f), 
        m_FLCRules() {
    std::cout << std::fixed << std::setprecision(3);
}

float FLController::evaluate() { 
    // --- Defuzzification using CoG method ---
    m_fuzzyOutput = defuzzifyWeightedAvg(m_FLCRules);

	return m_fuzzyOutput;
}

float FLController::defuzzifyWeightedAvg(std::vector<FLCRule>& rules) {
    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (const auto& rule : m_FLCRules) {
        auto [n, d] = rule.evaluate(); // Evaluate the rules
        numerator += n;
        denominator += d;
    }
    float output = (denominator != 0.0f) ? numerator / denominator : 0.0f;
    if constexpr (ENABLE_LOGGING)
        log("numerator: " + STR(numerator) + ", denominator: " + STR(denominator) + ", output: " + STR(output));
    return output;
}

void FLController::setRules(const std::vector<FLCRule>& rules) {
    m_FLCRules = rules;
} 

void FLController::reset() {
	m_fuzzyOutput = 0.0f;
}


// --- FLCRule Implementation ---
std::pair<float, float> FLCRule::evaluate() const {
    float mf1 = m_conditions[0].evaluate();
    float mf2 = m_conditions[1].evaluate();
    float membership = flcOperator(mf1, mf2, m_operator);
    float output = membership * m_weight;

    // Evaluate output. This should be a membership function
    if (m_type == FLCType::NEG)
        output *= -1.0f;
    else if (m_type == FLCType::ZERO)
        output *= 0.0f;
    if constexpr (ENABLE_LOGGING)
        log("mf1: "+ STR(mf1) + ", mf2: "+ STR(mf2) +", membership: "+ STR(membership) + ", output: "+ STR(output));

    return std::make_pair(output, membership);
}

float FLCRule::flcOperator(float a, float b, FLCOperatorsType type) const {
    switch (type) {
        case FLCOperatorsType::PROD:        return a * b;
        case FLCOperatorsType::AND:         return std::min(a, b);
        case FLCOperatorsType::OR:          return std::max(a, b);
        case FLCOperatorsType::SUM:         return a + b;
        case FLCOperatorsType::BOUNDEDSUM:  return std::min(1.0f, a + b);
        case FLCOperatorsType::BOUNDEDDIFF: return std::max(0.0f, a + b - 1.0f);
        default:    return 0.0f;
    }
}