#include <iostream>
#include "cs_predictor.hpp"

using namespace tensorflow;

cs_predictor::cs_predictor(){
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

    hidden1 = new int [HIDDEN1_SIZE];   
}

cs_predictor::~cs_predictor(){
    delete [] hidden1;
}

void cs_predictor::predict(const ALEScreen* screen){
    if(!session) {
        std::cout << "Session Error..." << std::endl;
        exit(-1);
    }


    std::vector<std::pair<std::string, tensorflow::Tensor> > input;
    // Create New tensor and set value
        Tensor x(tensorflow::DT_FLOAT, TensorShape({1, 33600})); 
        auto x_map = x.tensor<float, 2>();

        for(int h = 0; h < 210; h++)
            for(int w = 0; w < 160; w++)
                x_map(0, h*160 + w) = screen->get(h, w);

        // Append <tname, Tensor> to input
        input.push_back(std::pair<std::string, Tensor>("x", x));

    std::vector<tensorflow::Tensor> outputs;  
    Status status = session->Run(input, {"hidden"}, {}, &outputs);
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
    if(t.shape().dim_size(0)!=1 || t.shape().dim_size(1) != HIDDEN1_SIZE){
        std::cout << "Tensor t size Error..." << std::endl;
        exit(-1);
    }

    auto tmap = t.tensor<float, 2>();
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        hidden1[i] = tmap(0, i);
    }
}
