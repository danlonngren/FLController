#include "FLController.h"

#include <math.h>
#include <memory>

template<typename T>
T clamp(T val, T minVal, T maxVal) {
    if (val < minVal) return minVal;
    else if (val > maxVal) return maxVal;
    else return val;
}

// --- Membership functions interface ---
class MembershipFunction {
public:
    // Evaluate the membership function at a given normalized point x
    virtual float evaluate(float x) const = 0;
    virtual float evaluateAndNormalise(float x, float max) const = 0;
    virtual ~MembershipFunction() = default;
};

class GaussianMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return clamp(expf(-(x * x) / 2.0f), 0.0f, 1.0f); // sigma = 1.0 assumed
    }
    float evaluateAndNormalise(float x, float max) const override {
        if (max <= 0.0f) return 0.0f;
        return clamp(expf(-(x * x) / (2.0f * max * max)), 0.0f, 1.0f);
    }
};

class LinearPositiveMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return clamp((x + 1.0f) / 2.0f, 0.0f, 1.0f);
    }
    float evaluateAndNormalise(float x, float max) const override {
        return clamp((x + 1.0f) / (2.0f * max), 0.0f, 1.0f);
    }
};

class LinearNegativeMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
        return clamp((1.0f - x) / 2.0f, 0.0f, 1.0f);
    }
    float evaluateAndNormalise(float x, float max) const override {
        return clamp((1.0f - x) / (2.0f * max), 0.0f, 1.0f);
    }
};

class FuzzyData {
public:
    float m_error;
    float m_dError;
    float m_iError;
    float m_eLast;
    float normalizedError;
    float normalizedDError;
    float normalizedIError;

    FuzzyData(float e) :
        m_error(e), 
        m_dError(0.0f), 
        m_iError(0.0f),
        m_eLast(0.0f)
    {}

    void update(float e, float eLim, float deLim, float ieLim) {
        m_error = e;
        m_dError = m_error - m_eLast; // Derivative error
        m_eLast = m_error;
            // Integral error with anti-windup
        m_iError = clamp(m_iError + m_error, -ieLim, ieLim);

        normalizedError = normalize(m_error, eLim);
        normalizedDError = normalize(m_dError, deLim);
        normalizedIError = normalize(m_iError, ieLim);
    }

    float normalize(float x, float limit) const {
        return clamp(x / std::abs(limit), -1.0f, 1.0f);
    }
};

FLController::FLController(float e_limit, float de_limit, float re_limit, float output_gain, float m_outputMax) :
    m_errorLimit(e_limit), 
    m_dErrorLimit(de_limit), 
    m_integralLimit(re_limit), 
    output_gain(output_gain), 
    m_outputMax(m_outputMax) {
        
    m_integralError = 0.0f;
    m_fuzzyOutput   = 0.0f;
    m_eLast = 0.0f;
}

void FLController::reset() {
	m_integralError = 0;
	m_fuzzyOutput   = 0;
}

void FLController::setLimits(float e_limit, float de_limit, float re_limit, float output_gain, float m_outputMax) {
	m_errorLimit = e_limit;
	m_dErrorLimit = de_limit;
	m_integralLimit = re_limit;
	this->output_gain = output_gain;
	this->m_outputMax = m_outputMax;
}

void FLController::setOutputMax(float m_outputMax) {
	this->m_outputMax = m_outputMax;
}

float FLController::evaluate(float input, float setpoint, bool level) {   
    // --- Compute error terms ---
	float error  = (setpoint - input);  // Error
	float dError = (error - m_eLast);        // Derivative error
    m_eLast = error;

    // Integral error with anti-windup
    m_integralError = clamp(m_integralError + error, -m_integralLimit, m_integralLimit);

        // --- Ser mode specific rule weights ---
    if (level) 	{
		m_w[0] = 1.0;
		m_w[1] = 0.35;
		m_w[2] = 1.0;
		m_w[3] = 0.7;
	} else {
		m_w[0] = 1.3;
		m_w[1] = 0.15;
		m_w[2] = 1.0;
		m_w[3] = 0;
	}

    // --- Fuzzification and rule evaluation ---
    float ne  = normalize(error, m_errorLimit);
    float nde = normalize(dError, m_dErrorLimit);
    float nre = normalize(m_integralError, m_integralLimit);

    auto negLMF = std::make_shared<LinearNegativeMF>();
    auto posLMF = std::make_shared<LinearPositiveMF>();    
    auto gauMF  = std::make_shared<GaussianMF>();

    m_rules = {
        // P- and P+ rules
        { [=]() { return negLMF->evaluate(ne) * negLMF->evaluate(nde); }, m_w[0], -1.0f },
        { [=]() { return posLMF->evaluate(ne) * posLMF->evaluate(nde); }, m_w[0],  1.0f },
        // D+ and D- rules
        { [=]() { return negLMF->evaluate(ne) * posLMF->evaluate(nde); }, m_w[1],  1.0f },
        { [=]() { return posLMF->evaluate(ne) * negLMF->evaluate(nde); }, m_w[1], -1.0f },
        // I- and I+ rules
        { [=]() { return negLMF->evaluate(ne) * negLMF->evaluate(nre); }, m_w[2], -1.0f },
        { [=]() { return posLMF->evaluate(ne) * posLMF->evaluate(nre); }, m_w[2],  1.0f },
        // Gaussian rules for reducing overshoot
        { [=]() { return  gauMF->evaluate(ne) * posLMF->evaluate(nde); }, m_w[3],  1.0f },
        { [=]() { return  gauMF->evaluate(ne) * negLMF->evaluate(nde); }, m_w[3], -1.0f },
    };

    // --- Defuzzification ---
    float numerator   = 0.0f;
    float denominator = 0.0f;

    for (const auto& rule : m_rules) {
        float ruleValue = rule.condition();
        numerator += ruleValue * rule.weight * rule.output;
        denominator += ruleValue;
    }

    float output = (denominator != 0.0f) ? numerator / denominator : 0.0f;
    output *= output_gain;
    output = clamp(output, -m_outputMax, m_outputMax);
    m_fuzzyOutput = output;

	return m_fuzzyOutput;
}

float FLController::normalize(float x, float limit) const {
    return clamp(x / std::abs(limit), -1.0f, 1.0f);
}

float FLController::positiveNL(float x) {
    return (((x * x * x + x) / 2020.0) + 0.5);
}

float FLController::negativeNL(float x) {
    return (((-x * x * x - x) / 2020.0) + 0.5);
}