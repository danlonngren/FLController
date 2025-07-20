#include "FLController.h"

#include <memory>
#include <array>
#include <algorithm>
#include <utility>

// --- Membership functions ---
float MF::GaussianMF(float x) {
    float mean = 0.0f;
    float sigma = 0.3f;
    float diff = x - mean;
    return std::exp(-(diff * diff) / (2.0f * sigma * sigma));
}

float MF::LinearCenterPMF(float x) {
    return (x + 1.0f) / 2.0f;  // x=-1 → 0, x=0 → 0.5, x=1 → 1
}

float MF::LinearCenterNMF(float x) {
    return (1.0f - x) / 2.0f;  // x=-1 → 1, x=0 → 0.5, x=1 → 0
}

float MF::LinearPMF(float x) {
    float start = 0.0f;
    float end = 1.0f;
    if (x <= start) return 0.0f;
    else if (x >= end) return 1.0f;
    else return (x - start) / (end - start);  // linear ramp from 0 to 1
}

float MF::LinearNMF(float x) {
    float start = -1.0f;
    float end = 0.0f;
    if (x <= start) return 1.0f;
    else if (x >= end) return 0.0f;
    else return (end - x) / (end - start);
}

float MF::NonLinearPMF(float x) {
    return (((-x * x * x - x) / 2020.0f) + 0.5f);
}

float MF::NonLinearNMF(float x) {
    return ((((x * x * x) + x) / 2020.0f) + 0.5f);
}

float MF::outputPos(float x) {
    return x; // Do nothing
}

float MF::outputNeg(float x) {
    return x * -1.0f;
} 


// --- FLController implementation ---
FLController::FLController( float normalizationMin, float normalizationMax) :
        m_normMin(normalizationMin),
        m_normMax(normalizationMax),
        m_FLCRules(),
        m_fuzzyOutput(0.0f) {
    std::cout << std::fixed << std::setprecision(3);
}

float FLController::evaluate() { 
    // --- Defuzzification using CoG method ---
    m_fuzzyOutput = defuzzifyWeightedAvg(m_FLCRules);

	return m_fuzzyOutput;
}

float FLController::defuzzifyWeightedAvg(std::vector<FLCRule>& rules) {
    float weightedSum = 0.0f;
    float totalWeight = 0.0f;

    for (const auto& rule : m_FLCRules) {
        auto [output, weight] = rule.evaluate(); // Evaluate the rules
        weightedSum += output;
        totalWeight += weight;
    }
    float output = (totalWeight != 0.0f) ? weightedSum / totalWeight : 0.0f;
    if constexpr (ENABLE_LOGGING)
        log("weightedSum: " + toStr(weightedSum) + ", totalWeight: " + toStr(totalWeight) + ", output: " + toStr(output));
    return output;
}

void FLController::setRules(std::vector<FLCRule> rules) {
    m_FLCRules = std::move(rules);
} 

void FLController::reset() {
	m_fuzzyOutput = 0.0f;
    m_FLCRules.clear();
}


// --- FLCRule Implementation ---
std::pair<float, float> FLCRule::evaluate() const {
    float a = cond1.evaluate();
    float b = cond2.evaluate();
    const float membership = m_operator ? m_operator(a, b) : 0.0f;

    float output = membership * m_weight;
    // Evaluate output. This should be a membership function
    switch (m_type) {
        case FLCType::Positive: break;
        case FLCType::Negative: output *= -1.0; break;
        case FLCType::Zero:     output = 0.0f; break;
    }

    if constexpr (ENABLE_LOGGING)
        log("membership: "+ toStr(membership) + ", output: "+ toStr(output));

    return {output, membership};
}