#ifndef __CS_PREDICTOR_HPP__ 
#define __CS_PREDICTOR_HPP__ 

#include "model_loader.hxx"
#include "../environment/ale_screen.hpp"
#include "../common/Constants.h"
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

class cs_predictor {
public:
	cs_predictor();
	virtual ~cs_predictor();
	void predict(const ALEScreen* screen);
	const int* get_hidden1(){return hidden1;}

private:
	tensorflow::Session* session;

	//outputs
    int* hidden1;
};

#endif