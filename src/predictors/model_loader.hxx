#ifndef MODEL_LOADER_HXX
#define MODEL_LOADER_HXX

#include <vector>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include "../environment/ale_screen.hpp"


class ModelLoader{
public:
	ModelLoader();

	virtual ~ModelLoader();

	int load(tensorflow::Session*, const std::string) ;

	tensorflow::GraphDef graphdef; //Graph Definition for current model
};


#endif