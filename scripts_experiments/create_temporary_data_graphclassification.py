#!/usr/bin/env python

# author: Christopher Hojny

import argparse
import math

def create_temporary_file(infile, outfile, globalbudgetpercentage, strength, labelshift):

    # read data of instance from file
    fin = open(infile, 'r')

    # read basic information
    line = fin.readline()
    content = line.split()
    assert len(content) == 5

    nnodes = int(content[0])
    nedges = int(content[1])
    nfeatures = int(content[2])
    nlogits = int(content[3])
    label = int(content[4])

    # read edges
    edges_first = []
    edges_second = []
    if nedges > 0:
        line = fin.readline()
        content = line.split()
        assert len(content) == nedges

        for c in content:
            edges_first.append(int(c))

        line = fin.readline()
        content = line.split()
        assert len(content) == nedges

        for c in content:
            edges_second.append(int(c))

    A = [[0 for j in range(nnodes)] for i in range(nnodes)]
    for i in range(nedges):
        A[edges_first[i]][edges_second[i]] = 1
        A[edges_second[i]][edges_first[i]] = 1

    # read feature assignments
    feature_assignments = []
    for v in range(nnodes):
        line = fin.readline()
        content = line.split()
        assert len(content) < nfeatures

        features_v = nfeatures * [0.0]
        for c in content:
            features_v[int(c)] = 1.0
        feature_assignments.append(features_v)

    fin.close()

    # compute degrees of graph
    degrees = [sum(A[v]) for v in range(nnodes)]
    max_degree = max(degrees)

    # compute global budget
    globalbud = math.ceil(nedges * globalbudgetpercentage)

    # write information to file
    fout = open(outfile, 'w')

    fout.write("robustclassification\n")
    fout.write("nodes\n")
    fout.write(f"{nnodes}\n")
    fout.write("adjacencies\n")
    for i in range(nnodes):
        for j in range(nnodes):
            if j < nnodes - 1:
                fout.write(f"{A[i][j]} ")
            else:
                fout.write(f"{A[i][j]}\n")
    fout.write("features\n")
    fout.write(f"{nfeatures}\n")
    fout.write("featureLB\n")
    for v in range(nnodes):
        for f in range(nfeatures):
            if f < nfeatures - 1:
                fout.write(f"{feature_assignments[v][f]} ")
            else:
                fout.write(f"{feature_assignments[v][f]}\n")
    fout.write("featureUB\n")
    for v in range(nnodes):
        for f in range(nfeatures):
            if f < nfeatures - 1:
                fout.write(f"{feature_assignments[v][f]} ")
            else:
                fout.write(f"{feature_assignments[v][f]}\n")
    fout.write(f"global\n{globalbud}\nlocal\n")
    for v in range(nnodes):
        deg = degrees[v]
        bound = max(0, deg - max_degree + strength)
        fout.write(f"{bound}\n")
    fout.write("originalclassification\n")
    fout.write(f"{label}\n")
    fout.write("modifiedclassification\n")
    fout.write(f"{(label + labelshift) % nlogits}\n")

    fout.close()
                
                

if __name__ == "__main__":

    # create a parser for arguments
    parser = argparse.ArgumentParser(description='creates temporary files for graph classification problems')
    parser.add_argument('inputfile', metavar="inputfile", type=str, help='input file containing information about the graph')
    parser.add_argument('outputfile', metavar="outputfile", type=str, help='output file containing SCIP readable data')
    parser.add_argument('globalbudgetpercentage', metavar="globalbudgetpercentage", type=float, help='percentage of edges allowed to be attacked')
    parser.add_argument('strength', metavar="strength", type=int, help='local attack strength')
    parser.add_argument('labelshift', metavar="labelshift", type=int, help='shift to determine attacked label')

    args = parser.parse_args()

    create_temporary_file(args.inputfile, args.outputfile, args.globalbudgetpercentage, args.strength, args.labelshift)
