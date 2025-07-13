#pragma once

#include "FLControllerInterface.h"

#ifndef FUZZYCONTROLLER_H
#define FUZZYCONTROLLER_H

#include <functional>
#include <vector>

class FLController : public FLControllerInterface {
	private:
		float m_errorLimit;
		float m_dErrorLimit;
		float m_integralLimit;
		float output_gain;
		float m_outputMax;
		float m_integralError;
		float m_eLast;
		float m_fuzzyOutput;
		float m_w[4];
		
		struct Rule {
			std::function<float()> condition; // Activation function
			float weight;
			float output;
		};

		// TODO: Change to array
		std::vector<Rule> m_rules;
	public:
		FLController(float e_limit, float de_limit, float re_limit, float output_gain, float out_max);

		void reset();
		float evaluate(float input, float setpoint, bool Level);
		void setLimits(float e_limit, float de_limit, float re_limit, float output_gain, float out_max);
		void setOutputMax(float out_max);

	private:
		float normalize(float x, float limit) const;

		// TODO: Fix
		float positiveNL(float x);
		float negativeNL(float x);
};

#endif
