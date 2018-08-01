#include <iostream>
#include <vector>
#include <map>

#include "model_loader.hxx"
#include "../common/Constants.h"

using namespace tensorflow;

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