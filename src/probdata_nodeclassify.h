/**@file   probdata_nodeclassify.h
 * @brief  Problem data for node classification problems on GNNs
 * @author Christopher Hojny
 *
 * This file handles the main problem data used in node classification problems on GNNs.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __PROBDATA_NODECLASSIFY_H__
#define __PROBDATA_NODECLASSIFY_H__

#include "struct_gnn.h"
#include "struct_problem.h"

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** sets up the problem data */
SCIP_RETCODE SCIPprobdataCreateNodeClassify(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata,        /**< data about optimization problem on GNN */
   SCIP_Real**           lbgnnoutputvars,    /**< array of lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< array of lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< array of upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< array of upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent       /**< array storing upper bounds on node content before
                                              *   applying an activation function */
   );

/** returns number of layers in GNN */
GNN_DATA* SCIPgetProbdataNodeClassifyGNNNData(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns adjacency matrix of graph of in node classification problem */
SCIP_Bool** SCIPgetProbdataNodeClassifyAdjacencyMatrix(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns number of nodes of graph in node classification problem */
int SCIPgetProbdataNodeClassifyNNodes(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns global attack budget in node classification problem */
int SCIPgetProbdataNodeClassifyGlobalBudget(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns local attack budget in node classification problem */
int SCIPgetProbdataNodeClassifyLocalBudget(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns GNN output variables */
SCIP_VAR*** SCIPgetProbdataNodeClassifyGNNOutputVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns auxiliary variables */
SCIP_VAR*** SCIPgetProbdataNodeClassifyAuxVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns variables modeling activity of activation function */
SCIP_VAR*** SCIPgetProbdataNodeClassifyIsActiveVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

/** returns variables modeling adjacency */
SCIP_VAR** SCIPgetProbdataNodeClassifyAdjacencyVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   );

#ifdef __cplusplus
}
#endif

#endif
