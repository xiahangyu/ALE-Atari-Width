#ifndef CNNAE_MODEL_LOADER_HXX
#define CNNAE_MODEL_LOADER_HXX

#include <vector>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include "../environment/ale_screen.hpp"

namespace cnnAE_model {

	class FeatureAdapter{
	public:
		FeatureAdapter();

		virtual ~FeatureAdapter();

	    void assign_k_screens(const ALEScreen* k_screens) ;

	    void assign_one_step_act(int act) ;

	    std::vector<std::pair<std::string, tensorflow::Tensor> > input;
	};


	class ModelLoader{
	public:
		ModelLoader();

		virtual ~ModelLoader();

	    int load(tensorflow::Session*, const std::string) ;

	    tensorflow::GraphDef graphdef; //Graph Definition for current model
	};

}

#endif