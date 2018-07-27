#include <iostream>
#include "cnnAE.hpp"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

CNNAE::CNNAE(){
    std::string model_path = "./nn_model/seq_nn_model/nn_model_frozen.pb"

    // Create New Session
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        exit(-1);
    }

    // Create prediction demo
    cnnAE_model::ModelLoader model;  //Create demo for prediction
    if (0 != model.load(session, model_path)) {
        std::cout << "Error: Model Loading failed..." << std::endl;
        exit(-1);
    }
}

CNNAE::~CNNAE(){}

const int* CNNAE::get_hidden1(const ALEScreen* k_screens){
    cnnAE_model::FeatureAdapter input_feat;
    input_feat.assign_k_screens(k_screens);

    if(!session) {
        std::cout << "Session Error: src/predictors/cnnAE.cpp:get_hidden1()" << std::endl;
        exit(-1);
    }

    std::vector<tensorflow::Tensor> outputs;  
    tensorflow::Status status = session->Run(input_feat.input, {"hidden1"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;
        exit(-1);
    }

    //Fetch output value
    if(outputs.size() != 1){
        std::cout << "Output size Error: src/predictor/cnnAE.cpp:get_hidden1()..." << std::endl;
        exit(-1);
    }

    Tensor t = outputs[0];                   // Fetch the first tensor
    if(t.shape().dim_size(1) != HIDDEN1_SIZE){
        std::cout << "Tensor t size Error: src/predictor/cnnAE.cpp:get_hidden1()..." << std::endl;
        exit(-1);
    }

    auto tmap = t.tensor<float, 2>();
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        hidden1[i] = tmap[0][i];
    }
    return hidden1;
}

const int* CNNAE::get_hidden2(const ALEScreen* k_screens, int act){
    cnnAE_model::FeatureAdapter input_feat;
    input_feat.assign_k_screens(k_screens);
    input_feat.assign_one_step_act(act);

    if(!session) {
        std::cout << "Session Error: src/predictors/cnnAE.cpp:get_hidden2()" << std::endl;
        exit(-1);
    }

    std::vector<tensorflow::Tensor> outputs;  
    tensorflow::Status status = session->Run(input_feat.input, {"hidden2"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;
        exit(-1);
    }

    //Fetch output value
    if(outputs.size() != 1){
        std::cout << "Output size Error: src/predictor/cnnAE.cpp:get_hidden2()..." << std::endl;
        exit(-1);
    }

    Tensor t = outputs[0];                   // Fetch the first tensor
    if(t.shape().dim_size(1) != HIDDEN2_SIZE){
        std::cout << "Tensor t size Error: src/predictor/cnnAE.cpp:get_hidden2()..." << std::endl;
        exit(-1);
    }

    auto tmap = t.tensor<float, 2>();
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        hidden2[i] = tmap[0][i];
    }
    return hidden2;
}

const int* CNNAE::get_pred(const ALEScreen* k_screens, int act){
    cnnAE_model::FeatureAdapter input_feat;
    input_feat.assign_k_screens(k_screens);
    input_feat.assign_one_step_act(act);

    if(!session) {
        std::cout << "Session Error: src/predictors/cnnAE.cpp:get_hidden2()" << std::endl;
        exit(-1);
    }

    std::vector<tensorflow::Tensor> outputs;  
    tensorflow::Status status = session->Run(input_feat.input, {"pred"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;
        exit(-1);
    }

    //Fetch output value
    if(outputs.size() != 1){
        std::cout << "Output size Error: src/predictor/cnnAE.cpp:get_hidden2()..." << std::endl;
        exit(-1);
    }

    Tensor t = outputs[0];                   // Fetch the first tensor
    if(t.shape().dim_size(2) != 33600){
        std::cout << "Tensor t size Error: src/predictor/cnnAE.cpp:get_hidden2()..." << std::endl;
        exit(-1);
    }

    auto tmap = t.tensor<float, 3>();
    for (int i = 0; i < 33600; i++) {
        pred[i] = tmap[0][0][i];
    }
    return pred;
}
