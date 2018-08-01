#include <iostream>
#include "os_predictor.hpp"

using namespace tensorflow;

os_predictor::os_predictor(){
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
    // pred = new int [33600];
}

os_predictor::~os_predictor(){
    // delete [] hidden1;
    delete [] hidden2;

    // delete [] hidden3;
    // delete pred;
}

void os_predictor::predict(const ALEScreen* screen, int act){
    if(!session) {
        std::cout << "Session Error..." << std::endl;
        exit(-1);
    }

    std::vector<std::pair<std::string, tensorflow::Tensor> > input;
    // Screen tensor
    Tensor x(tensorflow::DT_FLOAT, TensorShape({1, 33600})); 
    auto x_map = x.tensor<float, 2>();
    for(int h = 0; h < 210; h++)
        for(int w = 0; w < 160; w++)
            x_map(0, h*160 + w) = screen->get(h, w);
    input.push_back(std::pair<std::string, Tensor>("x", x));

    // Action tensor
    Tensor t_act(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 18})); 
    auto act_map = t_act.tensor<float, 2>();
    for (int i = 0; i < 18; i++) {
        act_map(0, i) = (i == act) ? 1 : 0;
    }
    input.push_back(std::pair<std::string, Tensor>("act", t_act));

    std::vector<tensorflow::Tensor> outputs;  
    Status status = session->Run(input, {"hidden2"}, {}, &outputs);  //"hidden1","hidden2","hidden3","y_hat"
    if (!status.ok()) {
        std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;
        exit(-1);
    }

    //Fetch output value
    if(outputs.size() != 1){
        std::cout << "Output size Error..." << std::endl;
        exit(-1);
    }

    Tensor t = outputs[0];                   // Fetch the first tensor
    if(t.shape().dim_size(0)!=1 || t.shape().dim_size(1) != HIDDEN2_SIZE){
        std::cout << "Tensor t size Error..." << std::endl;
        exit(-1);
    }

    auto tmap = t.tensor<float, 2>();
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        hidden2[i] = tmap(0, i);
    }
}
