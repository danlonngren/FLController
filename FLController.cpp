#include "FLController.h"

// --- FLController implementation ---
FLController::FLController( float normalizationMin, float normalizationMax) :
		m_FLCRules(nullptr),
		m_rulesCount(0),
		m_fuzzyOutput(0.0f) {
}

float FLController::evaluate() { 
	// Defuzzification using CoG method
	if (!m_FLCRules)
		return 0;

	m_fuzzyOutput = defuzzifyWeightedAvg(m_FLCRules, m_rulesCount);

	return m_fuzzyOutput;
}

float FLController::defuzzifyWeightedAvg(const FuzzyRule* rules, uint32_t rulesCount) {
	float weightedSum = 0.0f;
	float totalWeight = 0.0f;

	if (!rules)
		return 0;

	for (uint32_t i = 0; i < rulesCount; i++) {
		auto [output, weight] = rules[i].evaluate(); // Evaluate the rules
		weightedSum += output;
		totalWeight += weight;
	}

	float output = (totalWeight != 0.0f) ? weightedSum / totalWeight : 0.0f;
	return output;
}

void FLController::setRules(FuzzyRule* rules, uint32_t rulesCount) {
	if (!rules)
		return;

	m_FLCRules = rules;
	m_rulesCount = rulesCount;
} 

void FLController::reset() {
	m_fuzzyOutput = 0.0f;
	m_FLCRules = nullptr;
}

// --- FLCRule Implementation ---
FuzzyRule::RuleResult FuzzyRule::evaluate() const {
	float a = m_inputA.evaluate();
	float b = m_inputB.evaluate();
	const float membership = m_operator ? m_operator(a, b) : 0.0f;

	float output = 0.0;
	output = m_outputMf ? m_outputMf(membership * m_weight) : 0.0f;

	return {output, membership};
}