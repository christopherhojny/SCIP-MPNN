#!/usr/bin/env python

# author: Christopher Hojny

import argparse
import gurobipy as gp
import subprocess

def read_and_solve_instance(instance, solfile, memlimit, timelimit):

    m = gp.read(instance)

    # set parameters
    m.Params.SoftMemLimit = memlimit
    m.Params.TimeLimit = timelimit
    m.Params.BestObjStop = -0.001
    m.Params.BestBdStop = 0.001
    m.Params.Threads = 1

    # add some noise to some constraints
    conss = m.getConstrs()
    for c in conss:
        if c.ConstrName.startswith("reluB"):
            assert c.Sense == '<'
            c.RHS += 1e-11

    m.optimize()

    # find (non-) robustness
    result = 0
    if m.ObjBound > 0:
        result = 1
    elif m.ObjVal < 0:
        result = -1

    # print a solution if it is available
    if m.SolCount > 0:
        f = open(solfile, 'w')
        for v in m.getVars():
            if v.X != 0.0:
                f.write(f"{v.VarName} {v.X}\n")
        f.close()

        return result, True

    return result, False

if __name__ == "__main__":

    # create a parser for arguments
    parser = argparse.ArgumentParser(description='solves a GNN instance using Gurobi and checks solution via SCIP')
    parser.add_argument('scipbinary', metavar="scipbinart", type=str, help='SCIP binary')
    parser.add_argument('gnnfile', metavar="gnnfile", type=str, help='file encoding the GNN')
    parser.add_argument('instance', metavar="instance", type=str, help='instance that needs to be solved')
    parser.add_argument('memlimit', metavar="memlimit", type=float, help='memory limit in GB')
    parser.add_argument('timelimit', metavar="timelimit", type=float, help='time limit in seconds')
    parser.add_argument('bounds', metavar="bounds", type=str, help='description of bounds')
    parser.add_argument('probtype', metavar="probtype", type=str, help='type of problem (graph or node)')

    args = parser.parse_args()

    assert args.bounds == "basic" or args.bounds == "sbt"
    assert args.probtype == "graph" or args.probtype == "node"

    gnn = args.gnnfile.split('/')[-1]
    inst = args.instance.split('/')[-1]
    ptype = "robustclassify"
    if args.probtype == "node":
        ptype = "nodeclassify"

    # create MPS file using SCIP
    write_setting = f"{gnn}_{inst}.set"
    tmp_instance = f"{gnn}_{inst}.mps"
    f = open(write_setting, 'w')
    f.write(f'gnn/onlywritemodel = "{tmp_instance}"\n')
    if args.bounds == "basic":
        f.write(f'gnn/{ptype}/useenhancedbounds = FALSE\n')
    else:
        f.write(f'gnn/{ptype}/useenhancedbounds = TRUE\n')
    f.close()

    subprocess.run([args.scipbinary, args.gnnfile, args.instance, "-t", str(args.timelimit), "-m", str(args.memlimit * 1000), "-s", write_setting])

    tmp_sol_file = f"{gnn}_{inst}.sol"
    result, has_solution = read_and_solve_instance(tmp_instance, tmp_sol_file, args.memlimit, args.timelimit)

    print(f"robustness of instance: {result}")

    subprocess.run(["rm", f"{write_setting}"])

    if has_solution:
        # check solution
        check_setting = f"{gnn}_{inst}.set"
        f = open(check_setting, 'w')
        f.write(f'gnn/onlychecksol = "{tmp_sol_file}"\n')
        if args.bounds == "basic":
            f.write(f'gnn/{ptype}/useenhancedbounds = FALSE\n')
        else:
            f.write(f'gnn/{ptype}/useenhancedbounds = TRUE\n')
        f.close()

        subprocess.run([args.scipbinary, args.gnnfile, args.instance, "-t", str(args.timelimit), "-m", str(args.memlimit * 1000), "-s", check_setting])

        subprocess.run(["rm", f"{check_setting}"])
