#ifndef __OS_PREDICTOR_HPP__ 
#define __OS_PREDICTOR_HPP__ 

#include "model_loader.hxx"
#include "../environment/ale_screen.hpp"
#include "../common/Constants.h"
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

class os_predictor {
public:
	os_predictor();
	virtual ~os_predictor();
	void predict(const ALEScreen* screen, int act);

	const int* get_hidden1(){return hidden1;}
	const int* get_hidden2(){return hidden2;}
	
	const int* get_hidden3(){return hidden3;}
	const int* get_pred(){return pred;}

private:
	tensorflow::Session* session;

	//outputs
    int* hidden1;
    int* hidden2;
    
    int* hidden3;
    int* pred;
};

#endif