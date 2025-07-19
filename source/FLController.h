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

#define ENABLE_LOGGING 1

template<typename T>
T clamp(T val, T minVal, T maxVal) {
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

// --- Fuzzy Logic Controller Data ---
class FLCData {
public:
	float fuzzyOutput;
	float m_p, m_i, m_d;
	float m_min, m_max;

	FLCData() : 
        m_p(0.0f), m_i(0.0f), m_d(0.0f), 
        m_min(0.0f), m_max(0.0f) {}

	void set(float p, float i, float d, float min, float max) {
		m_p = p;
		m_i = i;
		m_d = d;
        m_min = min;
        m_max = max;
		if (ENABLE_LOGGING)
			std::cout 
			<< "Data Set:" 
			<< " m_p: " << m_p << ", m_i: " << m_i << ", m_d: " << m_d 
			<< std::endl;
	}

	void setMinMax(float min, float max) {
		m_min = min; 
        m_max = max;
	}

	void reset() { 
        m_p = m_i = m_d = 0.0f;
        fuzzyOutput = m_min = m_max = 0.0f; 
    }
};

// --- Fuzzy Logic data types ---
class FuzzyData {
public:
    virtual float getData(const FLCData& data) const = 0;
    virtual ~FuzzyData() = default;
};

class FuzzyDataP : public FuzzyData {
public:
    float getData( const FLCData& data) const override 
        { return data.m_p; }
};

class FuzzyDataI : public FuzzyData {
public:
    float getData(const FLCData& data) const override 
        { return data.m_i; }
};

class FuzzyDataD : public FuzzyData {
private:
public:
    float getData( const FLCData& data) const override 
        { return data.m_d; }
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

// --- Fuzzy Logic Operators ---
class FLCOperators {
public:
	virtual ~FLCOperators() = default;
	virtual float operation(float normA, float normB) const = 0;
};

class FLCAndOperator : public FLCOperators {
public:
	float operation(float normA, float normB) const override {
		return std::min(normA, normB); // Minimum for AND operation
	}
};

class FLCOrOperator : public FLCOperators {
public:
	float operation(float normA, float normB) const override {
		return std::max(normA, normB); // Maximum for OR operation
	}
};

class FLCSumOperator : public FLCOperators {
public:
	float operation(float normA, float normB) const override {
		return normA + normB; // Sum for SUM operation
	}
};

class FLCProdOperator : public FLCOperators {
public:
	float operation(float normA, float normB) const override {
		return normA * normB; // Product for PRODUCT operation
	}
};

// --- Fuzzy Sets With MF and Data ---
struct FLCSet {
	std::shared_ptr<MembershipFunction> mf;
	std::shared_ptr<FuzzyData> data;
	FLCSet(	std::shared_ptr<MembershipFunction> mf = nullptr, 
			std::shared_ptr<FuzzyData> data = nullptr) 
		: mf(mf), data(data) {}
    
    float evaluate( FLCData& _data ) const {
        return mf->evaluate(data->getData(_data));
    }
};

// --- Fuzzy Logic Controller Set ---
class FLCRule {
public:
	enum FLCType { POS, NEG, ZERO };

private:
	std::array<FLCSet, 2> m_set;
	std::shared_ptr<FLCOperators> m_operator;
	FLCType m_type;
	float m_weight;

public:
	FLCRule(FLCSet set1, FLCSet set2,
			std::shared_ptr<FLCOperators> op,
			FLCType type, float weight) :
		m_set({ set1, set2 }), 
		m_operator(op), 
		m_weight(weight),
		m_type(type) {}

	std::pair<float, float> evaluate(FLCData& data) const {
		float mfOut1 = m_set[0].evaluate(data);
		float mfOut2 = m_set[1].evaluate(data);
		float membership = m_operator->operation(mfOut1, mfOut2);
		float outScaled = membership * m_weight;

        // Evaluate output fonction
		if (m_type == FLCType::NEG)
			outScaled *= -1.0f;
        else if (m_type == FLCType::ZERO)
            outScaled *= 0.0f;

		std::cout 
			<< "mfOut1: " << mfOut1 
			<< ", mfOut2: " << mfOut2 
			<< ", membership: " << membership 
			<< ", outScaled: " << outScaled << std::endl;
		return std::make_pair(outScaled, membership);
	}
};

class FLController : public FLControllerInterface {
private:
	float m_normMin, m_normMax;
	FLCData m_fuzzyData;
	std::vector<FLCRule> m_FLCRules;

public:
	FLController(float normalizationMin, float normalizationMax );
	float evaluate(float pError, float iError, float dError);
	void setRules(const std::vector<FLCRule>& rules);
	FLCData getData() const { return m_fuzzyData; }
	void reset();

private:
    float defuzzifyWeightedAvg(std::vector<FLCRule>& rules) {
        float numerator   = 0.0f;
        float denominator = 0.0f;
        for (const auto& rule : m_FLCRules) {
            auto [n, d] = rule.evaluate(m_fuzzyData); // Evaluate the rules
            numerator += n;
            denominator += d;
        }
        float output = (denominator != 0.0f) ? numerator / denominator : 0.0f;
        return output;
    }
}; // End FLController




//////////////////////////////////////////

// --- Fuzzy variable that holds MF and label ---
enum class FuzzyLabel{NEG = 0, ZERO = 1, POS = 2, LOW = 3, MEDIUM = 4, HIGH = 5};
struct FuzzyTerm {
	FuzzyLabel label;
	std::shared_ptr<MembershipFunction> mf;
};

class FuzzyVariable {
	std::vector<FuzzyTerm> terms;
public:
	void addTerm(const FuzzyLabel label, std::shared_ptr<MembershipFunction> mf) {
		terms.push_back({label, mf});
	}

	std::map<FuzzyLabel, float> fuzzify(float input) {
		std::map<FuzzyLabel, float> results;
		for (const auto& term : terms) {
			results[term.label] = term.mf->evaluate(input);
		}
		return results;
	}
};

struct FuzzyRule {
    FuzzyLabel e_label;
    FuzzyLabel de_label;
    FuzzyLabel ie_label;
    FuzzyLabel output_label;
    float weight = 1.0f;

	FuzzyRule(FuzzyLabel e, FuzzyLabel de, FuzzyLabel ie, FuzzyLabel out, float w = 1.0f)
        : e_label(e), de_label(de), ie_label(ie), output_label(out), weight(w) {}

	float get(const std::map<FuzzyLabel, float>& m, FuzzyLabel label) const {
		auto it = m.find(label);
		return (it != m.end()) ? it->second : 0.0f;
	}

	float evaluate(const std::map<FuzzyLabel, float>& e,
					const std::map<FuzzyLabel, float>& de,
					const std::map<FuzzyLabel, float>& ie) const {
		return get(e, e_label) * get(de, de_label) * get(ie, ie_label) * weight;
	}
};

class FuzzyEngine {
    FuzzyVariable error, deltaError, integralError, output;
    std::vector<FuzzyRule> rules;
public:
    void setInputVariables(FuzzyVariable e, FuzzyVariable de, FuzzyVariable ie) {
        error = e; deltaError = de; integralError = ie;
    }

    void setOutputVariable(FuzzyVariable out) {
        output = out;
    }

    void addRule(const FuzzyRule& rule) {
        rules.push_back(rule);
    }

    float compute(float eVal, float deVal, float ieVal) {
        auto e_fuzz  = error.fuzzify(eVal);
        auto de_fuzz = deltaError.fuzzify(deVal);
        auto ie_fuzz = integralError.fuzzify(ieVal);

        std::map<FuzzyLabel, float> outputMembership;

        for (const auto& rule : rules) {
            float strength = rule.evaluate(e_fuzz, de_fuzz, ie_fuzz);
            outputMembership[rule.output_label] = std::max(outputMembership[rule.output_label], strength);
        }

        float step = 0.01f;
        float sum_num = 0.0f, sum_den = 0.0f;

        for (float x = 0.0f; x <= 1.0f; x += step) {
            float mu = 0.0f;
            auto outputFuzz = output.fuzzify(x);
            for (const auto& [label, degree] : outputFuzz) {
                float ruleStrength = outputMembership[label];
                mu = std::max(mu, ruleStrength * degree);
            }
            sum_num += x * mu;
            sum_den += mu;
        }
		
		return (sum_den == 0.0f) ? 50.0f : ((sum_num / sum_den + 1.0f) * 50.0f);  // Map [-1, 1] → [0, 100]

        // return (sum_den == 0.0f) ? 0.0f : sum_num / sum_den;
    }
};
