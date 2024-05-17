#!/usr/bin/env python

# author: Christopher Hojny

import argparse
import json

def get_n_layers(gnn_params):
    '''
    returns number of layers in GNN

    gnn_params - dictionary of parameters extracted from JSON format
    '''

    assert 'L' in gnn_params[0]

    return gnn_params[0]['L']

def write_input_layer(params, outbuffer):
    '''
    writes parameters of a sage layer to a file

    params    - dictionary of parameters extracted from JSON format from first layer of GNN
    outbuffer - file buffer used for writing information
    '''

    n_in_features = params["in_channels"]

    # write information about sage layer
    outbuffer.write(f"input\n{n_in_features}\n")

def write_sage_layer(params, outbuffer):
    '''
    writes parameters of a sage layer to a file

    params    - dictionary of parameters extracted from JSON format for a single layer
    outbuffer - file buffer used for writing information
    '''

    n_in_features = params["in_channels"]
    n_out_features = params["out_channels"]
    activation = "none"
    if params["activation"]:
        activation = "relu"

    # write information about sage layer
    outbuffer.write(f"sage\n{n_in_features} {n_out_features} {activation}\n")

    # write information for every out_feature in layer
    outbuffer.write("nodeweights\n")
    for f2 in range(n_out_features):
        for f1 in range(n_in_features):
            outbuffer.write(f"{params['w1'][f2][f1]}")
            if f1 < n_in_features - 1:
                outbuffer.write(" ")
        outbuffer.write("\n")

    outbuffer.write("edgeweights\n")
    for f2 in range(n_out_features):
        for f1 in range(n_in_features):
            outbuffer.write(f"{params['w2'][f2][f1]}")
            if f1 < n_in_features - 1:
                outbuffer.write(" ")
        outbuffer.write("\n")

    outbuffer.write("bias\n")
    for f2 in range(n_out_features):
        outbuffer.write(f"{params['b'][f2]}\n")

def write_addpool_layer(params, outbuffer):
    '''
    writes parameters of an addpool layer to a file

    params    - dictionary of parameters extracted from JSON format for a single layer
    outbuffer - file buffer used for writing information
    '''

    assert params["layer"] == "add_pool"

    # write information about addpool layer
    outbuffer.write("addpool\n")

def write_dense_layer(params, outbuffer):
    '''
    writes parameters of an addpool layer to a file

    params    - dictionary of parameters extracted from JSON format for a single layer
    outbuffer - file buffer used for writing information
    '''

    n_in_features = params["in_channels"]
    n_out_features = params["out_channels"]
    activation = "none"
    if params["activation"]:
        activation = "relu"

    # write information about dense layer
    outbuffer.write(f"dense\n{n_in_features} {n_out_features} {activation}\n")

    # write information for every out_feature in layer
    outbuffer.write("denseweights\n")
    for f2 in range(n_out_features):
        for f1 in range(n_in_features):
            outbuffer.write(f"{params['w'][f2][f1]}")
            if f1 < n_in_features - 1:
                outbuffer.write(" ")
        outbuffer.write("\n")

    outbuffer.write("bias\n")
    for f2 in range(n_out_features):
        outbuffer.write(f"{params['b'][f2]}\n")

def write_layer(params, outbuffer):
    '''
    writes parameters of a GNN layer to a file

    params    - dictionary of parameters extracted from JSON format for a single layer
    outbuffer - file buffer used for writing information
    '''

    if params["layer"] == "sage":
        write_sage_layer(params, outbuffer)
    elif params["layer"] == "add_pool":
        write_addpool_layer(params, outbuffer)
    else:
        assert params["layer"] == "dense"
        write_dense_layer(params, outbuffer)

def parse_and_write_GNN(gnn_path, out_path):
    '''
    gets information of a GNN read from a JSON file and stores it in GNN file format

    gnn_path - path to JSON file
    out_path - path for storing output file
    '''

    gfile = open(gnn_path, "r")
    params = json.load(gfile)
    n_layers = get_n_layers(params)

    # write information about layers
    ofile = open(out_path, "w")
    ofile.write(f"{n_layers + 1}\n") # increment by 1 to take input layer into account
    write_input_layer(params[1], ofile)
    for i in range(n_layers):
        write_layer(params[i+1], ofile)

    ofile.close()
    gfile.close()

if __name__ == "__main__":

    # create a parser for arguments
    parser = argparse.ArgumentParser(description='generates GNN instance file from JSON file')
    parser.add_argument('jsonpath', metavar="json_path", type=str, help='path of JSON file')
    parser.add_argument('gnnpath', metavar="gnn_path", type=str, help='path for storing GNN file')
    parser.add_argument('fname', metavar="f_name", type=str, help='name of file for storing GNN (without file ending)')

    args = parser.parse_args()
    out_path = f"{args.gnnpath}/{args.fname}.gnn"

    parse_and_write_GNN(args.jsonpath, out_path)
