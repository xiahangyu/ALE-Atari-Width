#ifndef __CNN_AE_HPP__ 
#define __CNN_AE_HPP__ 

#include<Python.h>
#include "../common/Constants.h"

#define AE_HIDDEN_STATE_SIZE 34560

class CNNAE {
public:
	CNNAE();
	virtual ~CNNAE();
	int* predict(const IntMatrix &subtracted_screen);	//Get the hidden state of input subtracted_screen
private:
	PyObject* pMod = NULL;
    PyObject* pFunc = NULL;
    PyObject* pScreen = NULL;
};

#endif