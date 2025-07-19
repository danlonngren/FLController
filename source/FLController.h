#pragma once

#include "FLControllerInterface.h"

#include <functional>
#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <optional>
#include <map>

#include <iostream>
#include <iomanip>


// --- Logging extraction ---
constexpr bool ENABLE_LOGGING = true;

inline void log(const std::string& msg) {
    if constexpr (ENABLE_LOGGING)
        std::cout << msg << std::endl;
}

inline float clamp(float val, float minVal, float maxVal) {
    if (val < minVal) return minVal;
    else if (val > maxVal) return maxVal;
    else return val;
}

inline float normaliseVal(float x, float x_min, float x_max) {
    float b;
    float a = (x - x_min) / (x_max - x_min); // normalize to [0, 1]
    std::cout 
        << "x: " << x << ", a: " << a 
        << ", x_min: " << x_min << ", x_max: " << x_max 
        << std::endl;
    b = a * 2.0f - 1.0f;                // scale to [-1, 1]
    return clamp(b, -1.0f, 1.0f);  // no clamp needed unless you want to guard against overshoot
}

inline float normZeroToOne(float x, float x_min, float x_max) {
    float a = (x - x_min) / (x_max - x_min); // normalize to [0, 1]
    return clamp(a, -1.0f, 1.0f);  // no clamp needed unless you want to guard against overshoot
}

// --- Fuzzy Logic data types ---
class FuzzyData {
public:
    virtual float getData() const = 0;
    virtual void setData(float value) = 0;
    virtual ~FuzzyData() = default;
};

class FuzzyDataBasic : public FuzzyData {
    float value;
public:
    FuzzyDataBasic(float val = 0.0f) : value(val) {};
    float getData() const override { return value; }
    void setData(float val) {
        value = val;
    }
};

// --- Membership functions interface ---
class MembershipFunction {
public:
    // Evaluate the membership function at a given normaliseVald point x
    virtual float evaluate(float x) const = 0;
    virtual ~MembershipFunction() = default;
};

class TriangularMF : public MembershipFunction {
    float a, b, c;
public:
    TriangularMF(float a, float b, float c) : 
		a(a), b(b), c(c) {}

    float evaluate(float x) const override {
        if (x <= a || x >= c) return 0.0f;
        else if (x == b) return 1.0f;
        else if (x < b) return (x - a) / (b - a);
        else return (c - x) / (c - b);
    }
};

class GaussianMF : public MembershipFunction {
    float mean;
    float sigma;
public:
    GaussianMF(float mean = 0.0f, float sigma = 0.3f) : 
		mean(mean), sigma(sigma) {}

    float evaluate(float x) const override {
        float diff = x - mean;
        return std::exp(-(diff * diff) / (2.0f * sigma * sigma));
    }
};
class GaussianNMF : public MembershipFunction {
    float mean;
    float sigma;
public:
    GaussianNMF(float mean = 0.0f, float sigma = 0.3f) : 
		mean(mean), sigma(sigma) {}

    float evaluate(float x) const override {
        float diff = x - mean;
        return -std::exp(-(diff * diff) / (2.0f * sigma * sigma));
    }
};

class LinearCenterPMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return (x + 1.0f) / 2.0f;  // x=-1 → 0, x=0 → 0.5, x=1 → 1
    }
};

class LinearCenterNMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return (1.0f - x) / 2.0f;  // x=-1 → 1, x=0 → 0.5, x=1 → 0
    }
};

class LinearPMF : public MembershipFunction {
private:
    float start, end;
public:
    LinearPMF(float startRange = 0.0f, float endRange = 1.0f)
        : start(startRange), end(endRange) {}

    float evaluate(float x) const override {
        if (x <= start) return 0.0f;
        else if (x >= end) return 1.0f;
        else return (x - start) / (end - start);  // linear ramp from 0 to 1
    }
};

class LinearNMF : public MembershipFunction {
private:
    float start, end;
public:
    LinearNMF(float startRange = -1.0f, float endRange = 0.0f)
        : start(startRange), end(endRange) {}

    float evaluate(float x) const override {
        if (x <= start) return 1.0f;
        else if (x >= end) return 0.0f;
        else return (end - x) / (end - start);  // linear ramp from 1 to 0
    }
};

class NonLinearPMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return (((-x * x * x - x) / 2020.0f) + 0.5f);
    }
};

class NoneLinearNMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return ((((x * x * x) + x) / 2020.0f) + 0.5f);
    }
};

class outputPos : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return x; // Do nothing
    }
};

class outputNeg : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return x * -1.0f;
    }
};

// --- Fuzzy Sets With MF and Data ---
struct FLCSet {
	std::shared_ptr<MembershipFunction> mf;
	std::shared_ptr<FuzzyData> data;
	FLCSet(	std::shared_ptr<MembershipFunction> mf, std::shared_ptr<FuzzyData> data) 
		: mf(mf), data(data) {}
    
    float evaluate() const {
        return mf->evaluate(data->getData());
    }
};

// --- Fuzzy Logic Controller Set ---
class FLCRule {
public:
	enum FLCType { POS, NEG, ZERO }; // Sugeno-Style Output
    enum FLCOperatorsType { PROD, AND, OR, SUM, BOUNDEDSUM, BOUNDEDDIFF };
private:
	std::array<FLCSet, 2> m_set;
	FLCOperatorsType m_operator;
	FLCType m_type;
	float m_weight;

public:
	FLCRule(FLCSet set1, FLCSet set2, FLCOperatorsType op,
			FLCType type, float weight) :
		m_set({ set1, set2 }), 
		m_operator(op), 
		m_weight(weight),
		m_type(type) {}

	std::pair<float, float> evaluate() const {
		float mf1 = m_set[0].evaluate();
		float mf2 = m_set[1].evaluate();
		float membership = flcOperator(mf1, mf2, m_operator);
		float output = membership * m_weight;

        // Evaluate output. This should be a membership function
		if (m_type == FLCType::NEG)
			output *= -1.0f;
        else if (m_type == FLCType::ZERO)
            output *= 0.0f;

		std::cout 
			<< "mf1: " << mf1 
			<< ", mf2: " << mf2 
			<< ", membership: " << membership 
			<< ", output: " << output << std::endl;
		return std::make_pair(output, membership);
	}

private:
    float flcOperator(float a, float b, FLCOperatorsType type) const {
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
}; // End FLCRule

class FLController : public FLControllerInterface {
private:
	float m_normMin, m_normMax;
	std::vector<FLCRule> m_FLCRules;
    float m_fuzzyOutput;

public:
	FLController(float normalizationMin, float normalizationMax );
	float evaluate();
	void setRules(const std::vector<FLCRule>& rules);
	void reset();

private:
    float defuzzifyWeightedAvg(std::vector<FLCRule>& rules) {
        float numerator   = 0.0f;
        float denominator = 0.0f;
        for (const auto& rule : m_FLCRules) {
            auto [n, d] = rule.evaluate(); // Evaluate the rules
            numerator += n;
            denominator += d;
        }
        float output = (denominator != 0.0f) ? numerator / denominator : 0.0f;
        return output;
    }
}; // End FLController