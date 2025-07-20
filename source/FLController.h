#pragma once

#include "FLControllerInterface.h"

#include <functional>
#include <cmath>


// --- Logging extraction ---
#include <iostream>

inline std::string toStr(float x) {
    return std::to_string(x);
}

#define ENABLE_LOGGING 1

#if ENABLE_LOGGING
    #define LOG(msg) do { std::cout << msg << std::endl; } while(0)
#else
    #define LOG(msg) do {} while(0)
#endif
 

// --- Utility ---
inline float clamp(float val, float minVal, float maxVal) {
    if (val < minVal) return minVal;
    else if (val > maxVal) return maxVal;
    else return val;
}

inline float normalizeToMinus1To1(float x, float x_min, float x_max) {
    float b;
    float a = (x - x_min) / (x_max - x_min); // normalize to [0, 1]    
    b = a * 2.0f - 1.0f;                // scale to [-1, 1]
    return clamp(b, -1.0f, 1.0f);
}

inline float normalizeTo0To1(float x, float x_min, float x_max) {
    float a = (x - x_min) / (x_max - x_min); // normalize to [0, 1]
    return clamp(a, 0.0f, 1.0f);
}


// --- Membership functions interface ---
namespace FuzzyMF {
    inline float GaussianMF(float x) {
        float mean = 0.0f;
        float sigma = 0.3f;
        float diff = x - mean;
        return std::exp(-(diff * diff) / (2.0f * sigma * sigma));
    }
    inline float LinearCenterPMF(float x) {
        return (x + 1.0f) / 2.0f;  // x=-1 → 0, x=0 → 0.5, x=1 → 1
    }
    inline float LinearCenterNMF(float x) {
        return (1.0f - x) / 2.0f;  // x=-1 → 1, x=0 → 0.5, x=1 → 0
    }
    inline float LinearPMF(float x) {
        float start = 0.0f;
        float end = 1.0f;
        if (x <= start) return 0.0f;
        else if (x >= end) return 1.0f;
        else return (x - start) / (end - start);  // linear ramp from 0 to 1
    }
    inline float LinearNMF(float x) {
        float start = -1.0f;
        float end = 0.0f;
        if (x <= start) return 1.0f;
        else if (x >= end) return 0.0f;
        else return (end - x) / (end - start);
    }
    inline float NonLinearPMF(float x) {
        return (((-x * x * x - x) / 2020.0f) + 0.5f);
    }
    inline float NonLinearNMF(float x) {
        return ((((x * x * x) + x) / 2020.0f) + 0.5f);
    }

    // Sugeno-Style Output
    inline float OutputPos(float x) { return x; }
    inline float OutputNeg(float x) { return x * -1.0f; } 
}; // End MF


// --- Fuzzy operators ---
namespace FuzzyOps {
    inline float Product(float a, float b)      { return a * b; }
    inline float And(float a, float b)          { return std::min(a, b); }
    inline float Or(float a, float b)           { return std::max(a, b); }
    inline float Sum(float a, float b)          { return a + b; }
    inline float BoundedSum(float a, float b)   { return std::min(1.0f, a + b); }
    inline float BoundedDiff(float a, float b)  { return std::max(0.0f, a + b - 1.0f); }
}; // End FuzzyOps


// --- Fuzzy Logic data types ---
struct FuzzyData {
    float value;

    FuzzyData(float val = 0.0f) : value(val) {};
    void setData(float val) { value = val; }

    // Callable interface
    float operator()() const {
        return value;
    }
}; // End FuzzyData


// --- Fuzzy binding Sets With MF and Data ---
struct FuzzyCondition  {
    using PtrMembershipFunction = float (*)(float);

	FuzzyCondition (PtrMembershipFunction _mf, FuzzyData& _data) 
		: pMf(_mf), data(_data) {}
    
    float evaluate() const {           
        return pMf ? pMf(data()) : 0.0f;
    }

    PtrMembershipFunction pMf;
	FuzzyData& data; // Reference to externally managed data
}; // End FuzzyCondition 

// --- Fuzzy Logic Controller rules ---
class FuzzyRule {
public:
    using PtrFuzzyOperator = float (*)(float, float);
    using PtrMembershipFunction = float (*)(float);

    struct RuleResult {
        float output = 0.0f; 
        float weight = 0.0f; 
    };

	FuzzyRule(FuzzyCondition inputMF1, 
            FuzzyCondition inputMF2, 
            PtrFuzzyOperator op,
			PtrMembershipFunction type, 
            float weight) :
		m_inputA(inputMF1), 
        m_inputB(inputMF2), 
		m_operator(op), 
		m_weight(weight),
		m_outputMf(type) {}

	FuzzyRule::RuleResult evaluate() const;

private:
	FuzzyCondition m_inputA, m_inputB;
    PtrFuzzyOperator m_operator;
	PtrMembershipFunction m_outputMf;
	float m_weight;
}; // End FuzzyRule


// --- Fuzzy Logic Controller Set ---
class FLController : public FLControllerInterface {
public:
	FLController(float normalizationMin, float normalizationMax );
	float evaluate();
	void setRules(std::vector<FuzzyRule> rules);
	void reset();
private:
    float defuzzifyWeightedAvg(std::vector<FuzzyRule>& rules);

	std::vector<FuzzyRule> m_FLCRules;
    float m_fuzzyOutput;
}; // End FLController