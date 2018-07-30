#ifndef __CNNAE_HPP__ 
#define __CNNAE_HPP__ 

#include "cnnAE_model_loader.hxx"
#include "../environment/ale_screen.hpp"
#include "../common/Constants.h"
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

class CNNAE {
public:
	CNNAE();
	virtual ~CNNAE();
	const int* get_hidden1(const ALEScreen* k_screens);
	const int* get_hidden2(const ALEScreen* k_screens, int act);
	const int* get_pred(const ALEScreen* k_screens, int act);

private:
	tensorflow::Session* session;

	//outputs
    int* hidden1;
    int* hidden2;
    int* pred;
};

#endif