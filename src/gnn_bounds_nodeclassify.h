/**@file   gnn_bounds_nodeclassify.h
 * @brief  functions to compute bounds on variables and expressions in GNNs
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __GNN_BOUNDS_NODECLASSIFY_H__
#define __GNN_BOUNDS_NODECLASSIFY_H__

#include "struct_gnn.h"
#include "type_gnn.h"

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** computes bounds for all variables and expressions in a layer of a GNN
 *
 *  @pre bounds on previous layers must have been computed and stored in bound arrays
 */
SCIP_RETCODE SCIPcomputeBoundsGNNNodeClassifyLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   int                   globalbudget,       /**< global attack budget */
   int                   localbudget,        /**< local attack budget per node */
   SCIP_VAR**            gnnoutputvars,      /**< output variables of layer (or NULL) */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency in problem (or NULL) */
   SCIP_Real**           lbinput,            /**< lower bounds on input for GNN nodes (or NULL) */
   SCIP_Real**           ubinput,            /**< upper bounds on input for GNN nodes (or NULL) */
   int                   layeridx,           /**< index of layer for which bounds are computed */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing lower bounds on node content before
                                               *   applying an activation function */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            lbauxvarsprev,      /**< lower bounds on auxiliary variables at previous layer */
   SCIP_Real*            ubgnnoutputvarsprev, /**< upper bounds on output at previous GNN layer */
   SCIP_Real*            ubauxvarsprev       /**< upper bounds on auxiliary variables at previous layer */
   );

/** computes bounds for all variables and expressions in GNN */
SCIP_RETCODE SCIPcomputeBoundsGNNNodeClassify(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   int                   globalbudget,       /**< global attack budget */
   int                   localbudget,        /**< local attack budget per node */
   SCIP_VAR***           gnnoutputvars,      /**< output variables of all layers (or NULL) */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency in problem (or NULL) */
   SCIP_Real**           lbinput,            /**< lower bounds on input for GNN nodes (or NULL) */
   SCIP_Real**           ubinput,            /**< upper bounds on input for GNN nodes (or NULL) */
   SCIP_Real***          lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real***          lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real***          lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real***          ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real***          ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real***          ubnodecontent       /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   );

#ifdef __cplusplus
}
#endif

#endif
