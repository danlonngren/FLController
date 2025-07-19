#include "FLController.h"

#include <math.h>
#include <memory>
#include <array>
#include <algorithm>
#include <utility>

FLController::FLController( float normalizationMin, float normalizationMax) :
        m_normMin(normalizationMin),
        m_normMax(normalizationMax),
        m_fuzzyData(), 
        m_FLCRules() {
    m_fuzzyData.reset();
    m_fuzzyData.setMinMax(m_normMin, m_normMax);
    std::cout << std::fixed << std::setprecision(3);
}

float FLController::evaluate(float pError, float iError, float dError) { 
    std::cout 
        << "evaluate[pError: " << pError 
        << ", iError: " << iError 
        << ", dError: " << dError 
        << "]" << std::endl;
    
    // --- Compute error terms ---
	m_fuzzyData.set(pError, iError, dError, m_normMax, m_normMin);

    // --- Defuzzification using CoG method ---
    m_fuzzyData.fuzzyOutput = defuzzifyWeightedAvg(m_FLCRules);

    std::cout << "m_fuzzyData.fuzzyOutput: " << m_fuzzyData.fuzzyOutput << std::endl;

	return m_fuzzyData.fuzzyOutput;
}

void FLController::setRules(const std::vector<FLCRule>& rules) {
    m_FLCRules = rules;
} 

void FLController::reset() {
	m_fuzzyData.reset();
}