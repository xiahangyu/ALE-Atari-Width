#include "CNN_AE.hpp"

CNNAE::CNNAE(){
	Py_Initialize();
	if( !Py_IsInitialized() ){
		printf("Failed to initialize\n");
		exit(-1);
	}
    
	PyRun_SimpleString("import sys");  
    PyRun_SimpleString("sys.path.append('./python_nn_model/Autoencoder')");

    PyObject* pModuleName = PyUnicode_FromString("predict");
    pMod = PyImport_Import(pModuleName);

    if(!pMod){
		PyErr_Print();
    	exit(-1);
    }

    PyObject* funcName = PyUnicode_FromString("predict");   
	pFunc = PyObject_GetAttr(pMod, funcName);
	if(!pFunc){
        PyErr_Print();
		exit(-1);
	}

	pScreen = PyList_New(SCREEN_SIZE);
}

CNNAE::~CNNAE(){
	Py_Finalize();
}

int* CNNAE::predict(const IntMatrix &subtracted_screen){
    //std::cout << "    predict 1" << std::endl;
    for(int i = 0; i < SCREEN_HEIGHT; i++)
    	for(int j = 0; j < SCREEN_WIDTH; j++)
        	PyList_SetItem(pScreen, i * SCREEN_WIDTH + j, Py_BuildValue("i", subtracted_screen[i][j]));
    //std::cout << "    predict 2" << std::endl;

    PyObject* pParm = PyTuple_New(1);
    //std::cout << "    predict 3" << std::endl;
    PyTuple_SetItem(pParm, 0, pScreen);
    //std::cout << "    predict 4" << std::endl;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

	PyObject* pRetVal = PyEval_CallObject(pFunc, pParm);

    PyGILState_Release(gstate);
    //std::cout << "    predict 5" << std::endl;
    if(!pRetVal){
        PyErr_Print();
        exit(-1);
    }

	static int hidden_state[AE_HIDDEN_STATE_SIZE];
    for(int i = 0; i < AE_HIDDEN_STATE_SIZE; i++){
        PyArg_Parse(PyList_GetItem(pRetVal, i), "i", &hidden_state[i]);
    }
    //std::cout << "    predict 6" << std::endl;

    if(!pParm)
    	delete pParm;
    if(!pRetVal)
    	delete pRetVal;
    //std::cout << "    predict 7" << std::endl;
    return hidden_state;
}