# freeze the graph and the weights
# output_name_nodes are seperated with ,
python3 freeze_graph.py --input_graph=./action-free/cnn/freeway/ckpt/graph.pbtxt --input_checkpoint=./action-free/cnn/freeway/ckpt/graph.ckpt --output_graph=../../trained_models/freeway/ac_cnn/model_frozen.pb --output_node_names=hidden
