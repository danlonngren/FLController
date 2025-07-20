#pragma once

#include "FLControllerInterface.h"

#include <functional>
#include <cmath>


// --- Logging extraction ---
#include <iostream>
#include <iomanip>
constexpr bool ENABLE_LOGGING = true;

inline std::string toStr(float x) {
    return std::to_string(x);
}

inline void log(const std::string& msg) {
    if constexpr (ENABLE_LOGGING)
        std::cout << msg << std::endl;
}
 

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
namespace MF {
    float GaussianMF(float x);
    float LinearCenterPMF(float x);
    float LinearCenterNMF(float x);
    float LinearPMF(float x);
    float LinearNMF(float x);
    float NonLinearPMF(float x);
    float NonLinearNMF(float x);
    float outputPos(float x);
    float outputNeg(float x);
}; // End MF


// --- Fuzzy operators ---
namespace FuzzyOps {
    inline float Product(float a, float b)      { return a * b; }
    inline float And(float a, float b)          { return std::min(a, b); }
    inline float Or(float a, float b)           { return std::max(a, b); }
    inline float Sum(float a, float b)          { return a + b; }
    inline float BoundedSum(float a, float b)   { return std::min(1.0f, a + b); }
    inline float BoundedDiff(float a, float b)  { return std::max(0.0f, a + b - 1.0f); }
};


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
class FLCBindingSet {
public:
    using PtrMembershipFunction = float (*)(float);

	FLCBindingSet(PtrMembershipFunction _mf, FuzzyData& d) 
		: pMf(_mf), data(d) {}
    
    float evaluate() const {           
        return pMf ? pMf(data()) : 0.0f;
    }

private:
    PtrMembershipFunction pMf;
	FuzzyData& data; // Reference to externally managed data
}; // End FLCBindingSet


// --- Fuzzy Logic Controller rules ---
class FLCRule {
public:
    using PtrFuzzyOperator = float (*)(float, float);

	enum FLCType { // Sugeno-Style Output
        Positive, 
        Negative, 
        Zero };

    struct FLCRuleOutput {
        float n; 
        float d; 
    };

	FLCRule(const FLCBindingSet& mfSet1, 
            const FLCBindingSet& mfSet2, 
            PtrFuzzyOperator op,
			FLCType type, 
            float weight) :
		cond1(mfSet1), 
        cond2(mfSet2), 
		m_operator(op), 
		m_weight(weight),
		m_type(type) {}

	std::pair<float, float> evaluate() const;

private:
	const FLCBindingSet& cond1;
	const FLCBindingSet& cond2;
    PtrFuzzyOperator m_operator;
	FLCType m_type;
	float m_weight;
}; // End FLCRule


// --- Fuzzy Logic Controller Set ---
class FLController : public FLControllerInterface {
public:
	FLController(float normalizationMin, float normalizationMax );
	float evaluate();
	void setRules(std::vector<FLCRule> rules);
	void reset();
private:
    float defuzzifyWeightedAvg(std::vector<FLCRule>& rules);

	float m_normMin, m_normMax;
	std::vector<FLCRule> m_FLCRules;
    float m_fuzzyOutput;
}; // End FLController