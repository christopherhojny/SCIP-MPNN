#!/usr/bin/env python

# author: Christopher Hojny

import argparse

import json
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import numpy as np
from torch.nn import Linear, ReLU
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F

class MPNN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(dataset.num_features, hidden_channels, aggr="sum")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="sum")
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr="sum")
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

def get_target_graph(model, index, dataset):
    """
    get the information of target graph

    inputs:

    index      -  the index of the target graph, ranging from 0 to len(dataset) - 1

    outputs:
    N          - number of nodes
    X          - node features, shape NxF
    A          - adjacency matrix, shape NxN
    logits     - outputs from the MPNN
    y_original - true label
    delta      - maximal number of modified edges for each node
    """

    data = dataset[index]
    N = data.x.shape[0]
    X = data.x.detach().numpy()
    edge_first = []
    edge_second = []
    for k in range(data.edge_index.shape[1]):
        u = data.edge_index[0, k]
        v = data.edge_index[1, k]
        edge_first.append(u)
        edge_second.append(v)

    logits = model(data.x, data.edge_index, None)
    label = torch.argmax(logits).detach().numpy()
    logits = logits.detach().numpy()
    nedges = data.edge_index.shape[1]

    return N, X, edge_first, edge_second, nedges, logits, label

def parse_and_write_files(datasetname, pt_path, store_path):

    dataset =  TUDataset(root="data/TUDataset", name=datasetname)

    num_graphs = len(dataset)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    hidden_channels = 16

    # load the trained model
    model = MPNN(hidden_channels, dataset)
    model.load_state_dict(torch.load(pt_path))
    model.eval()

    # create instance for each graph in the dataset
    for index in range(num_graphs):
        N, X, edge_first, edge_second, nedges, logits, label = get_target_graph(model, index, dataset)

        # extract all graphs with their associated data
        fout = open(f"{store_path}/graph_{datasetname}_{index}.gcinfo", 'w')

        # write basic information about graph
        fout.write(f"{N} {nedges} {X.shape[1]} {logits.shape[1]} {label}\n")

        # write edge lists
        for i in range(nedges):
            if i < nedges - 1:
                fout.write(f"{edge_first[i]} ")
            else:
                fout.write(f"{edge_first[i]}\n")
        for i in range(nedges):
            if i < nedges - 1:
                fout.write(f"{edge_second[i]} ")
            else:
                fout.write(f"{edge_second[i]}\n")
        
        # write assignment of features per node
        for v in range(N):
            for f in range(X.shape[1]):
                if X[v,f] == 1.0:
                    fout.write(f"{f} ")
                else:
                    assert X[v,f] == 0.0
            fout.write("\n")

        fout.close()


if __name__ == "__main__":

    # create a parser for arguments
    parser = argparse.ArgumentParser(description='generates basic graph instances for graph classification')
    parser.add_argument('ptpath', metavar="pt_path", type=str, help='path for pytorch files for GNN')
    parser.add_argument('storepath', metavar="store_path", type=str, help='path for for storing files containing basic information')
    parser.add_argument('testset', metavar="testset", type=str, help='name of benchmark test set (MUTAG or ENZYME)')

    args = parser.parse_args()

    if not args.testset in ["MUTAG", "ENZYMES"]:
        print(f"unknown test set name, expected MUTAG or ENZYME, but got {args.testset}")

    parse_and_write_files(args.testset, args.ptpath, args.storepath)
