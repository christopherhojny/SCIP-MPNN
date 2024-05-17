/**@file   probdata_robustclassify.h
 * @brief  Problem data for robust classification problems on GNNs
 * @author Christopher Hojny
 *
 * This file handles the main problem data used in robust classification problems on GNNs.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __PROBDATA_ROBUSTCLASSIFY_H__
#define __PROBDATA_ROBUSTCLASSIFY_H__

#include "struct_gnn.h"
#include "struct_problem.h"

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** sets up the problem data */
SCIP_RETCODE SCIPprobdataCreateRobustClassify(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata,        /**< data about optimization problem on GNN */
   SCIP_Real**           lbgnnoutputvars,    /**< array of lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< array of lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< array of upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< array of upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Bool             uselprelax          /**< whether we just want to solve the LP relaxation */
   );

/** returns number of layers in GNN */
GNN_DATA* SCIPgetProbdataRobustClassifyGNNNData(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns adjacency matrix of graph of in robust classification problem */
SCIP_Bool** SCIPgetProbdataRobustClassifyAdjacencyMatrix(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns number of nodes of graph in robust classification problem */
int SCIPgetProbdataRobustClassifyNNodes(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns global attack budget in robust classification problem */
int SCIPgetProbdataRobustClassifyGlobalBudget(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns local attack budget in robust classification problem */
int* SCIPgetProbdataRobustClassifyLocalBudget(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns GNN output variables */
SCIP_VAR*** SCIPgetProbdataRobustClassifyGNNOutputVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns auxiliary variables */
SCIP_VAR*** SCIPgetProbdataRobustClassifyAuxVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns variables modeling activity of activation function */
SCIP_VAR*** SCIPgetProbdataRobustClassifyIsActiveVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns variables modeling adjacency */
SCIP_VAR** SCIPgetProbdataRobustClassifyAdjacencyVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

SCIP_RETCODE SCIPsetOBBTobjective(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nfeatures,          /**< number of features */
   int                   layeridx,           /**< index of layer */
   int                   nodeidx,            /**< index of node of underlying graph */
   int                   featureidx,         /**< index of feature */
   SCIP_Bool             maximize            /**< whether objective sense is maximization */
   );

SCIP_RETCODE SCIPresetOBBTobjective(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nfeatures,          /**< number of features */
   int                   layeridx,           /**< index of layer */
   int                   nodeidx,            /**< index of node of underlying graph */
   int                   featureidx          /**< index of feature */
   );

#ifdef __cplusplus
}
#endif

#endif
