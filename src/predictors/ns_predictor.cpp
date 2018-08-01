#include <iostream>
#include "ns_predictor.hpp"

using namespace tensorflow;

ns_predictor::ns_predictor(){
    std::string model_path = "./models/freeway/subtracted/nn_model_frozen.pb";

    // Create New Session
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        exit(-1);
    }

    // Create prediction demo
    ModelLoader model;  //Create demo for prediction
    if (0 != model.load(session, model_path)) {
        std::cout << "Error: Model Loading failed..." << std::endl;
        exit(-1);
    }

    // hidden1 = new int [HIDDEN1_SIZE];
    hidden2 = new int [HIDDEN2_SIZE];

    // hidden3 = new int [HIDDEN3_SIZE];
    pred = new int [33600];
}

ns_predictor::~ns_predictor(){
    // delete [] hidden1;
    delete [] hidden2;

    // delete [] hidden3;
    delete pred;
}

void ns_predictor::predict(const ALEScreen* k_screens, int act){
    if(!session) {
        std::cout << "Session Error..." << std::endl;
        exit(-1);
    }

    std::vector<std::pair<std::string, tensorflow::Tensor> > input;
    // Screen tensor
    Tensor x(tensorflow::DT_FLOAT, TensorShape({1, NUM_K, 33600})); 
    auto x_map = x.tensor<float, 3>();
    for(int k = 0; k < NUM_K; k++)
        for(int h = 0; h < 210; h++)
            for(int w = 0; w < 160; w++)
                x_map(0, k, h*160 + w) = k_screens[k].get(h, w);
    input.push_back(std::pair<std::string, Tensor>("x", x));

    // Action tensor
    Tensor t_act(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 18})); 
    auto act_map = t_act.tensor<float, 2>();
    for (int i = 0; i < 18; i++) {
        act_map(0, i) = (i == act) ? 1 : 0;
    }
    input.push_back(std::pair<std::string, Tensor>("one_step_act", t_act));

    std::vector<tensorflow::Tensor> outputs;  
    Status status = session->Run(input, {"hidden2", "pred"}, {}, &outputs);  //"hidden1","hidden2","hidden3","pred"
    if (!status.ok()) {
        std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;
        exit(-1);
    }

    //Fetch output value
    if(outputs.size() != 2){
        std::cout << "Output size Error..." << std::endl;
        exit(-1);
    }

    Tensor t_hidden = outputs[0];   
    Tensor t_pred = outputs[1]; 
    if(t_hidden.shape().dim_size(0)!=1 || t_pred.shape().dim_size(0)!=1 || t_pred.shape().dim_size(1)!=1 ){
        std::cout << "Output tensor size Error..." << std::endl;
        exit(-1);
    }

    auto t_hidden_map = t_hidden.tensor<float, 2>();
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        hidden2[i] = t_hidden_map(0, i);   
    }

    auto t_pred_map = t_pred.tensor<float, 3>();
    for (int i = 0; i < 33600; i++) {
        pred[i] = t_pred_map(0, 0, i);
    }
}
