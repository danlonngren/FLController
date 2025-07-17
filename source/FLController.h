#pragma once

#include "FLControllerInterface.h"

#ifndef FUZZYCONTROLLER_H
#define FUZZYCONTROLLER_H

#include <functional>
#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <optional>

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
    float a = (x - x_min) / (x_max - x_min);  // normalize to [0, 1]
    float b = a * 2.0f - 1.0f;                // scale to [-1, 1]
    return clamp(b, -1.0f, 1.0f);  // no clamp needed unless you want to guard against overshoot
}

// --- Fuzzy Logic Controller Data ---
class FLCData {
private:
	float eLast;
	float lastDt;
public:
	float fuzzyOutput;
	float pData, iData, dData;
	float m_min;
	float m_max;
	bool started = true;

	FLCData() : pData(0.0f), iData(0.0f), dData(0.0f), eLast(0.0f) {}

	void set(float e, float dt, float min, float max) {
		if (ENABLE_LOGGING)
			std::cout << "Setting data e: " << e << ", dt: " << dt << std::endl;
		m_min = min; m_max = max;
		pData = e; 
        iData += pData * dt;
        if (started) {
			started = false;
            eLast = pData;
        }
		dData  = (pData - eLast) / dt;
		if (ENABLE_LOGGING)
			std::cout 
			<< "Data - pData: " << e 
			<< ", iData: " << iData 
			<< ", pData: " << pData 
			<< ", eLast: " << eLast 
			<< std::endl;
		eLast = pData;
		lastDt = dt;
	}

	void setMinMax(float min, float max) {
		m_min = min; m_max = max;
	}

	void reset() { 
        pData = iData = dData = fuzzyOutput = eLast = 0.0f; 
        lastDt = 0.0f;
    }
};

// --- Membership functions interface ---
class MembershipFunction {
public:
    // Evaluate the membership function at a given normaliseVald point x
    virtual float evaluate(float x) const = 0;
    virtual ~MembershipFunction() = default;
};

class GaussianMF : public MembershipFunction {
    float mean;
    float sigma;
public:
    GaussianMF(float mean = 0.0f, float sigma = 0.3f) : mean(mean), sigma(sigma) {}

    float evaluate(float x) const override {
        float diff = x - mean;
        return std::exp(-(diff * diff) / (2.0f * sigma * sigma));
    }
};

class LinearPMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return (x + 1.0f) / 2.0f;  // x=-1 → 0, x=0 → 0.5, x=1 → 1
    }
};

class LinearNMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return (1.0f - x) / 2.0f;  // x=-1 → 1, x=0 → 0.5, x=1 → 0
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

// --- Fuzzy Logic data types ---
class FuzzyData {
public:
    virtual float calculate(FLCData& data, float dt) = 0;
    virtual ~FuzzyData() = default;
};

class FuzzyDataP : public FuzzyData {
public:
    float calculate(FLCData& data, float dt) override {
        float normPData = normaliseVal(data.pData, data.m_min, data.m_max);
		std::cout << "normPData: " << normPData << ", data.pData: " << data.pData << std::endl;
        return normPData;
    }
};

class FuzzyDataI : public FuzzyData {
public:
    float calculate(FLCData& data, float dt) override {
        if (dt <= 0.0f) return data.iData;
        float normIData = normaliseVal(data.iData, data.m_min, data.m_max);
		std::cout << "normIData: " << normIData << ", data.iData: " << data.iData << std::endl;
        return normIData;
    }
};

class FuzzyDataD : public FuzzyData {
private:
public:
    float calculate(FLCData& data, float dt) override {
		float normDData = normaliseVal(data.dData, data.m_min, data.m_max);
		std::cout << "normDData: " << normDData << ", data.dData: " << data.dData << std::endl;
        return normDData;
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

class FLCProductOperator : public FLCOperators {
public:
	float operation(float normA, float normB) const override {
		return normA * normB; // Product for PRODUCT operation
	}
};

struct FLCSet {
	FLCSet(	std::shared_ptr<MembershipFunction> mf = nullptr, 
			std::shared_ptr<FuzzyData> data = nullptr) 
		: mf(mf), data(data) {}
	std::shared_ptr<MembershipFunction> mf;
	std::shared_ptr<FuzzyData> data;
};

// --- Fuzzy Logic Controller Set ---
class FLCRule {
public:
	enum FLCType {
		POSITIVE,
		NEGATIVE
	};

private:
	std::array<FLCSet, 2> m_set;
	std::shared_ptr<FLCOperators> m_operator;
	float m_weight;
	float m_crispness;
	FLCType m_type;

public:
	FLCRule(FLCSet set1, 
			FLCSet set2,
			std::shared_ptr<FLCOperators> op,
			float weight,
			float crispness,
			FLCType type) :
		m_set({ set1, set2 }), 
		m_operator(op), m_weight(weight),
		m_crispness(crispness), m_type(type) {}

	std::pair<float, float> evaluate(FLCData& data, float dt) const {
		float mfOut1 = m_set[0].mf->evaluate(m_set[0].data->calculate(data, dt));
		float mfOut2 = m_set[1].mf->evaluate(m_set[1].data->calculate(data, dt));
		float out = m_operator->operation(mfOut1, mfOut2);
		float outScaled = out * m_weight * m_crispness;
		if (m_type == FLCType::NEGATIVE)
			outScaled *= -1.0f; // Negate output for negative rules
		std::cout 
			<< "mfOut1: " << mfOut1 
			<< ", mfOut2: " << mfOut2 
			<< ", out: " << out 
			<< ", outScaled: " << outScaled << std::endl;
		return std::make_pair(outScaled, out);
	}
};

class FLController : public FLControllerInterface {
private:
	float m_normMin;
	float m_normMax;
	float m_outputGain;
	float m_maxOutput;
	std::array<float, 4> m_w;

	FLCData m_fuzzyData;
	std::vector<FLCRule> m_FLCRules;

public:
	FLController(	float normalizationMin,
					float normalizationMax,
					float outputGain, 
					std::array<float, 4> weights);

	void reset();
	void setRules(const std::vector<FLCRule>& rules);
	float evaluate(float input, float setpoint, float dt);
	void setLimits(float eLim, float dLim, float iLim, float outputGain);
	FLCData getData() const { return m_fuzzyData; }

	void setOutputGain(float gain) { m_outputGain = gain; }
	float getOutputGain() const { return m_outputGain; }
};

#endif
