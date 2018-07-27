#include <iostream>
#include <vector>
#include <map>

#include "cnnAE_model_loader.hxx"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "../common/Constants.h"

using namespace tensorflow;

namespace cnnAE_model {

/*FeatureAdapter*/
    FeatureAdapter::FeatureAdapter(){}

    FeatureAdapter::~FeatureAdapter(){}

    void FeatureAdapter::assign_k_screens(const ALEScreen* k_screens) {
        // Create New tensor and set value
        Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({k, 1, 210*160})); 
        auto x_map = x.tensor<float, 3>();

        for (int i = 0; i < K; i++) 
            for(int h = 0; h < 210; h++)
                for(int w = 0; w < 160; w++)
                    x_map(i, 0, h*160 + w) = k_screens[i].get(h, w);

        // Append <tname, Tensor> to input
        input.push_back(std::pair<std::string, tensorflow::Tensor>("x", x));
    }

    void FeatureAdapter::assign_one_step_act(int act){
        // Create New tensor and set value
        Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 18})); 

        auto x_map = x.tensor<float, 2>();
        for (int i = 0; i < 18; i++) {
            x_map(0, i) = (i == act) ? 1 : 0;
        }

        // Append <tname, Tensor> to input
        input.push_back(std::pair<std::string, tensorflow::Tensor>("one_step_act", x));
    }


/*ModelLoader*/
    ModelLoader::ModelLoader(){}

    ModelLoader::~ModelLoader(){}
    
    int ModelLoader::load(tensorflow::Session* session, const std::string model_path) {
        //Read the pb file into the grapgdef member
        tensorflow::Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
        if (!status_load.ok()) {
            std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
            std::cout << status_load.ToString() << "\n";
            return -1;
        }

        // Add the graph to the session
        tensorflow::Status status_create = session->Create(graphdef);
        if (!status_create.ok()) {
            std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
            return -1;
        }
        return 0;
    }

    int ModelLoader::predict(tensorflow::Session* session, const FeatureAdapterBase& input_feature,
            const std::string output_node, int* prediction) {
        // The session will initialize the outputs
        std::vector<tensorflow::Tensor> outputs;         //shape  [batch_size]
        if(!session) {
            std::cout << "SESSION OK!!!!!!" << std::endl;
            return -1;
        }

        // @input: vector<pair<string, tensor> >, feed_dict
        // @output_node: std::string, name of the output node op, defined in the protobuf file
        tensorflow::Status status = session->Run(input_feature.input, {output_node}, {}, &outputs);
        if (!status.ok()) {
            std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;
            return -1;
        }

        //Fetch output value
        if(outputs.size() != 1){
            std::cout << "Output size Error: src/predictor/ann_model_loader.cpp:predict()..." << std::endl;
            return -1;
        }
        Tensor t = outputs[0];                   // Fetch the first tensor
        if(t.shape().dim_size(2) != 33600){
            std::cout << "Tensor t size Error: src/predictor/ann_model_loader.cpp:predict()..." << std::endl;

        }

        auto tmap = t.tensor<float, 3>();
        for (int i = 0; i < 33600; i++) {
            prediction[i] = tmap[0][0][i];
        }
        return 0;
    }
}