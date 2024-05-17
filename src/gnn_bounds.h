/**@file   gnn_bounds.cpp
 * @brief  functions to compute bounds on variables and expressions in GNNs for robust classification problems
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __GNN_BOUNDS_H__
#define __GNN_BOUNDS_H__

#include "struct_gnn.h"
#include "type_gnn.h"

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** computes bounds for variables and expressions in a pooling layer
 *
 *  @pre bounds on previous layers must have been computed and stored in bound arrays
 */
SCIP_RETCODE computeBoundsGNNPoolLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_POOL*   layerinfo,          /**< information about pooling layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            ubgnnoutputvarsprev /**< upper bounds on output at previous GNN layer */
   );

/** computes bounds for variables and expressions in a dense layer
 *
 *  @pre bounds on previous layers must have been computed and stored in bound arrays
 */
SCIP_RETCODE computeBoundsGNNDenseLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_DENSE*  layerinfo,          /**< information about dense layer */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            ubgnnoutputvarsprev /**< upper bounds on output at previous GNN layer */
   );

/** computes bounds for all variables and expressions in a layer of a GNN for robust classification problems
 *
 *  @pre bounds on previous layers must have been computed and stored in bound arrays
 */
SCIP_RETCODE SCIPcomputeBoundsGNNLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             isdirected,         /**< whether underlying  graph is directed */
   GNN_DATA*             gnndata,            /**< data about GNN */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
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

/** computes bounds for all variables and expressions in GNN for robust classification problems */
SCIP_RETCODE SCIPcomputeBoundsGNN(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             isdirected,         /**< whether underlying  graph is directed */
   GNN_DATA*             gnndata,            /**< data about GNN */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
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

/** frees arrays containing bounds for all variables and expressions in a layer of a GNN */
SCIP_RETCODE SCIPfreeBoundsGNN(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   SCIP_Real***          lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real***          lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real***          lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real***          ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real***          ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real***          ubnodecontent       /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   );

/** writes bounds on GNN variables to a file in Python dictionary format */
SCIP_RETCODE SCIPwriteBoundsPython(
   const char*           name,               /**< name of file to be created */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata,        /**< data about optimization problem on GNN */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent       /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   );

#ifdef __cplusplus
}
#endif

#endif
