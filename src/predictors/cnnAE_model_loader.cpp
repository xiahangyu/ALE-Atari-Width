#include <iostream>
#include <vector>
#include <map>

#include "cnnAE_model_loader.hxx"
#include "../common/Constants.h"

using namespace tensorflow;

namespace cnnAE_model {

/*FeatureAdapter*/
    FeatureAdapter::FeatureAdapter(){}

    FeatureAdapter::~FeatureAdapter(){}

    void FeatureAdapter::assign_k_screens(const ALEScreen* k_screens) {
        // Create New tensor and set value
        Tensor x(tensorflow::DT_FLOAT, TensorShape({NUM_K, 1, 210*160})); 
        auto x_map = x.tensor<float, 3>();

        for (int i = 0; i < NUM_K; i++) 
            for(int h = 0; h < 210; h++)
                for(int w = 0; w < 160; w++)
                    x_map(i, 0, h*160 + w) = k_screens[i].get(h, w);

        // Append <tname, Tensor> to input
        input.push_back(std::pair<std::string, Tensor>("x", x));
    }

    void FeatureAdapter::assign_one_step_act(int act){
        // Create New tensor and set value
        Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 18})); 

        auto x_map = x.tensor<float, 2>();
        for (int i = 0; i < 18; i++) {
            x_map(0, i) = (i == act) ? 1 : 0;
        }

        // Append <tname, Tensor> to input
        input.push_back(std::pair<std::string, Tensor>("one_step_act", x));
    }


/*ModelLoader*/
    ModelLoader::ModelLoader(){}

    ModelLoader::~ModelLoader(){}
    
    int ModelLoader::load(tensorflow::Session* session, const std::string model_path) {
        //Read the pb file into the grapgdef member
        Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
        if (!status_load.ok()) {
            std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
            std::cout << status_load.ToString() << "\n";
            return -1;
        }

        // Add the graph to the session
        Status status_create = session->Create(graphdef);
        if (!status_create.ok()) {
            std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
            return -1;
        }
        return 0;
    }
}