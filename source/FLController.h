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

template<typename T>
T clamp(T val, T minVal, T maxVal) {
    if (val < minVal) return minVal;
    else if (val > maxVal) return maxVal;
    else return val;
}

// --- Fuzzy Logic Controller Data ---
class FLCData {
private:
	float eLast;
	float lastDt;
    float maxOutput;
    float outputGain;
public:
	float fuzzyOutput;
	float pData, iData, dData;
	float m_pLim, m_iLim, m_dLim;

	FLCData() : pData(0.0f), iData(0.0f), dData(0.0f), eLast(0.0f) {}

	void set(float e, float dt) {
		pData  = e; 
        iData += pData * dt;
        if (lastDt == 0.0f) {
            lastDt = pData;
        }
		dData  = (pData - eLast) / dt;
		eLast = pData;
		lastDt = dt;
	}

	void setLimits(float pLim, float iLim, float dLim) {
		m_pLim = pLim;
		m_iLim = iLim;
		m_dLim = dLim;
	}

    void setMaxOutput(float max) { maxOutput = max; }
    float getMaxOutput() const { return maxOutput; }
    void setOutputGain(float gain) { outputGain = gain; }
    float getOutputGain() const { return outputGain; }

	void reset() { 
        pData = iData = dData = fuzzyOutput = eLast = 0.0f; 
        lastDt = 0.0f;
    }
};

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
        return clamp(expf(-(x * x) / 2.0f), -1.0f, 1.0f); // sigma = 1.0 assumed
    }
    float evaluateAndNormalise(float x, float max) const override {
        if (max <= 0.0f) return 0.0f;
        float normX = clamp(x / max, -1.0f, 1.0f);
        return evaluate(normX);
    }
};

class LinearPMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
		std::cout << "Evaluating LinearPMF with x: " << x << std::endl;
        return clamp((x + 1.0f) / 2.0f, -1.0f, 1.0f); // x ∈ [-1, 1]
    }

    float evaluateAndNormalise(float x, float max) const override {
        float normX = clamp(x / max, -1.0f, 1.0f);
        return evaluate(normX);
    }
};

class LinearNMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
		std::cout << "Evaluating LinearNMF with x: " << x << std::endl;
        return clamp((1.0f - x) / 2.0f, -1.0f, 1.0f); // x ∈ [-1, 1]
    }

    float evaluateAndNormalise(float x, float max) const override {
        float normX = clamp(x / max, -1.0f, 1.0f);
        return evaluate(normX);
    }
};

class NonLinearPMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
		std::cout << "Evaluating NonLinearPMF with x: " << x << std::endl;
        return clamp( (((-x * x * x - x) / 2020.0f) + 0.5f), 0.0f, 1.0f ); // x ∈ [-1, 1]
    }

    float evaluateAndNormalise(float x, float max) const override {
        float normX = clamp(x / max, -1.0f, 1.0f);
        return evaluate(normX);
    }
};

class NoneLinearNMF : public MembershipFunction {
public:
    float evaluate(float x) const override {
		std::cout << "Evaluating NoneLinearNMF with x: " << x << std::endl;
        return clamp(((((x * x * x) + x) / 2020.0f) + 0.5f), 0.0f, 1.0f); // x ∈ [-1, 1]
    }

    float evaluateAndNormalise(float x, float max) const override {
        float normX = clamp(x / max, -1.0f, 1.0f);
        return evaluate(normX);
    }
};

// --- Fuzzy Logic data types ---
class FuzzyData {
public:
    virtual float calculate(FLCData& data, float dt) = 0;
    virtual ~FuzzyData() = default;
	float normalize(float x, float limit) const {
        return clamp(x / std::abs(limit), -1.0f, 1.0f);
    }
};

class FuzzyDataP : public FuzzyData {
public:
    float calculate(FLCData& data, float dt) override {
        float normPData = normalize(data.pData, data.m_pLim);
        std::cout << "Calculating FuzzyDataP with pData: " << data.pData << std::endl;
        std::cout << "Normalized pData: " << normPData << std::endl;
        return normPData;
    }
};

class FuzzyDataI : public FuzzyData {
public:
    float calculate(FLCData& data, float dt) override {
        if (dt <= 0.0f) return data.iData;
        float normIData = normalize(data.iData, data.m_iLim);
        std::cout << "Calculating FuzzyDataI with iData: " << data.iData << std::endl;
        std::cout << "Normalized pData: " << normIData << std::endl;
        return normIData;
    }
};

class FuzzyDataD : public FuzzyData {
public:
    float calculate(FLCData& data, float dt) override {
        std::cout << "Calculating FuzzyDataD with dData: " << data.dData << std::endl;
        float normDData = normalize(data.dData, data.m_dLim);
        std::cout << "Normalized dData: " << normDData << std::endl;
        return normDData;
    }
};

// --- Fuzzy Logic Controller Rule ---
class FLCRuleSet {
public:
    struct RuleOutput {
        float output;
        float scaledOutput;
    };

private:
    std::array<std::shared_ptr<MembershipFunction>, 2> m_MFs;
    std::array<std::shared_ptr<FuzzyData>, 2> m_fuzzyData;
    float m_weight;
    float m_output;

public:
    FLCRuleSet(std::shared_ptr<MembershipFunction> mf1, std::shared_ptr<FuzzyData> fD1,  
				std::shared_ptr<MembershipFunction> mf2, std::shared_ptr<FuzzyData> fD2, 
				float weight, float output) :
        m_MFs{ mf1, mf2 }, 
		m_fuzzyData{ fD1, fD2 }, 
		m_weight(weight), 
		m_output(output) {}

    RuleOutput evaluate(FLCData& data, float dt) const {
		std::cout << "Evaluating FLCRuleSet with data: " << data.pData << " and dt: " << dt << std::endl;

		float fuzzyData1 = m_fuzzyData[0]->calculate(data, dt);
		float fuzzyData2 = m_fuzzyData[1]->calculate(data, dt);

		float fuzzyOutput1 = m_MFs[0]->evaluate(fuzzyData1);
		float fuzzyOutput2 = m_MFs[1]->evaluate(fuzzyData2);

        float output = fuzzyOutput1 * fuzzyOutput2;
        float scaledOutput = output * m_weight * m_output;

		std::cout << "\n Fuzzy Output 1 : " << fuzzyOutput1 
		          << "\n Fuzzy Output 2 : " << fuzzyOutput2 
		          << "\n fuzzyData1     : " << fuzzyData1 
		          << "\n fuzzyData2     : " << fuzzyOutput2 
		          << "\n m_weight       : " << m_weight 
		          << "\n m_output       : " << m_output 
		          << "\n Output	        : " << output 
		          << "\n Scaled Output  : " << scaledOutput << std::endl;

        return RuleOutput({ output, scaledOutput });
    }
    
    void setWeight(float weight) {
        m_weight = weight;
    }

    void setOutput(float output) {
        m_output = output;
    }

    float getWeight() const { return m_weight; }
    float getOutput() const { return m_output; }
};

class FLController : public FLControllerInterface {
	private:
		std::array<float, 4> m_w;
		
		std::vector<FLCRuleSet> m_fcRules;

		FLCData m_fuzzyData;

	public:
		FLController(float eLim, 
					float dLim, 
					float iLim, 
					float outputGain, 
					float outputMax,
                    std::array<float, 4> weights);

		void reset();
		void setRules(const std::vector<FLCRuleSet>& rules);
		float evaluate(float input, float setpoint, float dt);
		void setLimits(float eLim, float dLim, float iLim, float outputGain, float outputMax);
		void setOutputMax(float outputMax);
		FLCData getData() const { return m_fuzzyData; }

	private:
		float normalize(float x, float limit) const;

		// TODO: Fix
		// float positiveNL(float x);
		// float negativeNL(float x);
};

#endif
