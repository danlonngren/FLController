#include "FLController.h"

#include <math.h>
#include <memory>
#include <array>
#include <algorithm>
#include <utility>

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

void FLController::setRules(const std::vector<FLCRule>& rules) {
    m_FLCRules = rules;
} 

void FLController::reset() {
	m_fuzzyOutput = 0.0f;
}