#!/usr/bin/env python

# author: Christopher Hojny

import argparse

import json
import torch
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.loader import DataLoader
import numpy as np
from torch.nn import Linear, ReLU
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

class MPNN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(dataset.num_features, hidden_channels, aggr="sum")
        self.conv2 = SAGEConv(hidden_channels, dataset.num_classes, aggr="sum")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x

def get_target_graph(y, index, data):
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

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        index, 2, data.edge_index, relabel_nodes=True
    )
    new_index = mapping[0].detach().numpy()
    N = subset.shape[0]
    X = data.x[subset, :].detach().numpy()
    edge_first = []
    edge_second = []
    for k in range(edge_index.shape[1]):
        u = edge_index[0, k]
        v = edge_index[1, k]
        edge_first.append(u)
        edge_second.append(v)
    logits = y[index].detach().numpy()
    label = y[index].argmax().detach().numpy()
    nedges = len(edge_first)
    return N, X, edge_first, edge_second, nedges, logits, label, new_index

def parse_and_write_files(datasetname, pt_path, store_path):

    dataset =  AttributedGraphDataset(root="data/AttributedGraphDataset", name=datasetname)
    data = dataset[0]

    num_nodes = data.num_nodes
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    hidden_channels = 32

    # load the trained model
    model = MPNN(hidden_channels, dataset)
    model.load_state_dict(torch.load(pt_path))
    model.eval()
    y = model(data.x, data.edge_index)

    # create instance for each node in the graph
    for index in range(num_nodes):
        N, X, edge_first, edge_second, nedges, logits, label, newindex = get_target_graph(y, index, data)

        # extract all graphs with their associated data
        fout = open(f"{store_path}/graph_{datasetname}_{index}.ncinfo", 'w')

        # write basic information about graph
        fout.write(f"{N} {nedges} {X.shape[1]} {len(logits)} {label} {newindex}\n")

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
    parser = argparse.ArgumentParser(description='generates basic graph instances for node classification')
    parser.add_argument('ptpath', metavar="pt_path", type=str, help='path for pytorch files for GNN')
    parser.add_argument('storepath', metavar="store_path", type=str, help='path for for storing files containing basic information')
    parser.add_argument('testset', metavar="testset", type=str, help='name of benchmark test set (CiteSeer or Cora)')

    args = parser.parse_args()

    if not args.testset in ["CiteSeer", "Cora"]:
        print(f"unknown test set name, expected CiteSeer or Cora, but got {args.testset}")

    parse_and_write_files(args.testset, args.ptpath, args.storepath)
