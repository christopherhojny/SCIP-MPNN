# SCIP-MPNN

Additional information about the code for the paper

   Verifying message-passing neural networks via topology-based bounds tightening

by Christopher Hojny (\*), Shiqiang Zhang (*), Juan S. Campos, and Ruth Misener,
(\* co-first authors) which has been published at: Proceedings of the 41st International
Conference on Machine Learning, Vienna, Austria, PMLR 235, 2024.

The BibTeX reference is

```
    @inproceedings{HZCM2024,
      title={Verifying message-passing neural networks via topology-based bounds tightening},
      author={Christopher Hojny and Shiqiang Zhang and Juan S. Campos and Ruth Misener},
      booktitle={Proceedings of the 41st International Conference On Machine Learning, Vienna, Austria, PMLR 235},
      year={2024}
    }
```

## Contributors

- [Christopher Hojny](https://github.com/christopherhojny): Python and C code
- [Shiqiang Zhang](https://github.com/zshiqiang): trained GNN models and created instances

Shiqiang Zhang is funded by an Imperial College Hans Rausing PhD Scholarship.


## I Steps of the Code

1. To run the program, enter

    `bin/scipgnn.$(OSTYPE).$(ARCH).$(COMP).$(OPT).$(LPS)`

    (e.g., "bin/scipgnn.linux.x86_64.gnu.opt.spx2"). The first two arguments
    are mandatory and specify (i) the information about the trained GNN,
    (ii) the information about the verification problem that needs to be solved.

    Additional parameters can be:

    -s `<setting file>`

    -t `<time limit>`

    -m `<mem limit>`

    -n `<node limit>`

    -d `<display frequency>`

    We refer to the INSTALL file (Step 6) for a description of how to obtain
    files (i) and (ii).

2. After reading the problem, a MIP for solving the graph or node classification
    problem encoded in (ii) over a trained GNN encoded in (i) is created and solved.
    Depending on the parameter setting, basic bounds, static bound tightening, or
    aggressive bound tightening is used. These settings can be changed via the settings
    files

    settings/{node,graph}classification_basic.set for basic bounds,

    settings/{node,graph}classification_sbt.set for static bound tightening,

    settings/{node,graph}classification_abt.set for aggressive bound tightening.

3. When the binary is executed, log information of SCIP is printed to the terminal.
   This includes information about presolving and the solving process as well as
   statistics of SCIP and the best solution found during the solving process.
   Finally, also the number of local and global cuts of aggressive bound tightening
   is printed.


## II Running the Models via Gurobi

1. In the article, also a comparison with Gurobi is mentioned. To run such experiments,
   we assume that Gurobi's Python interface has been installed and is available in the
   standard Python environment. The experiments can be run via the script

    `scripts_experiments/run_gurobi.py`

    which takes the following input:

    - the SCIP binary, e.g., `bin/scipgnn.$(OSTYPE).$(ARCH).$(COMP).$(OPT).$(LPS)`
    - the file encoding the GNN, e.g., `data_experiments/gnn_instances/model_Cora.gnn`
    - the instance that needs to be solved, e.g., a file generated from
      `scripts_experiments/create_temporary_data_nodeclassification.py` for the file
      `data_experiments/node_classification_instances/graph_Cora_1.ncinfo`
    - the memory limit in GB
    - the time limit in seconds
    - "basic" or "sbt" to determine whether basic bounds or static bound tightening is used
    - "graph" or "node" to encode whether graph or node classification problem is solved

2. When the script is executed, log information of SCIP and Gurobi is printed to the terminal.
   First, information of SCIP about the generation of the input files for Gurobi is printed.
   Second, the Gurobi logs are provided. Finally, the solution found by Gurobi is provided
   to SCIP which checks whether the solution is actually feasible. If this is the case,

    `1/1 feasible solution given by solution candidate storage`

    will be printed to the screen. Otherwise, SCIP will report

    `violation: left hand side is violated by...`

    accompanied by a copy of the constraint that is violated by Gurobi's solution.
