/**@file   struct_problem.h
 * @brief  structs for problems on GNNs
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_STRUCT_PROBLEM_H_
#define __SCIP_STRUCT_PROBLEM_H_

#include "scip/scip.h"
#include "type_gnn.h"
#include "type_problem.h"

#ifdef __cplusplus
extern "C" {
#endif

/** information about robust classification problems on GNNs */
typedef struct GNNProb_Robustclassify
{
   int                   nnodes;             /**< number of nodes of underlying graph */
   int                   nfeatures;          /**< number of node features of graph */
   int                   globalbudget;       /**< global attack budget on graph */
   int*                  localbudget;        /**< array assigning each node an attack budget */
   SCIP_Bool**           adjacencymatrix;    /**< adjacency matrix of underlying graph */
   SCIP_Real**           featurelb;          /**< (nnodes x nfeatures)-matrix of lower bounds on feature assignments */
   SCIP_Real**           featureub;          /**< (nnodes x nfeatures)-matrix of upper bounds on feature assignments */
   int                   graphclassification; /**< index of feature the graph is classified as */
   int                   targetclassification; /**< index of feature a modified graph should be classified as */
} GNNPROB_ROBUSTCLASSIFY;

/** information about node classification problems on GNNs */
typedef struct GNNProb_Nodeclassify
{
   int                   nnodes;             /**< number of nodes of underlying graph */
   int                   nfeatures;          /**< number of node features of graph */
   int                   globalbudget;       /**< global attack budget on graph */
   int                   localbudget;        /**< local attack budget for each node of the graph */
   SCIP_Bool**           adjacencymatrix;    /**< adjacency matrix of underlying directed graph */
   SCIP_Real**           featurelb;          /**< (nnodes x nfeatures)-matrix of lower bounds on feature assignments */
   SCIP_Real**           featureub;          /**< (nnodes x nfeatures)-matrix of upper bounds on feature assignments */
   int                   nodeclassification; /**< node which shall be classified */
   int                   graphclassification; /**< index of feature the graph is classified as */
   int                   targetclassification; /**< index of feature a modified graph should be classified as */
} GNNPROB_NODECLASSIFY;

/** information about a layer */
typedef union GNNProb_Typeinfo
{
   GNNPROB_ROBUSTCLASSIFY robustclassifyinfo; /**< information about robust classification problem */
   GNNPROB_NODECLASSIFY  nodeclassifyinfo;   /**< information about node classification problem */
} GNNPROB_TYPEINFO;

/** data to encode a GNN */
typedef struct GNNProb_Data
{
   GNNPROB_TYPE          probtype;           /**< type of optimization problem on GNN */
   GNNPROB_TYPEINFO*     probtypeinfo;       /**< information about problem */
} GNNPROB_DATA;

#ifdef __cplusplus
}
#endif

#endif
