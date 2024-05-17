/**@file   struct_gnn.h
 * @brief  structs for GNNs
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_STRUCT_GNN_H_
#define __SCIP_STRUCT_GNN_H_

#include "scip/scip.h"
#include "type_gnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** information of a sage layer in a GNN */
typedef struct GNN_Layerinfo_Sage
{
   int                   ninputfeatures;     /**< number of input features */
   int                   noutputfeatures;    /**< number of output features */
   SCIP_Real**           nodeweights;        /**< node weights of sage layer */
   SCIP_Real**           edgeweights;        /**< edge weights of sage layer */
   SCIP_Real*            bias;               /**< biases in sage layer */
   GNN_ACTIVATIONTYPE    activation;         /**< activation function of sage layer */
} GNN_LAYERINFO_SAGE;

/** information of a dense layer in a GNN */
typedef struct GNN_Layerinfo_Dense
{
   int                   ninputfeatures;     /**< number of input features */
   int                   noutputfeatures;    /**< number of output features */
   SCIP_Real**           weights;            /**< weights of dense layer */
   SCIP_Real*            bias;               /**< biases of dense layer */
   GNN_ACTIVATIONTYPE    activation;         /**< activation function of dense layer */
} GNN_LAYERINFO_DENSE;

/** information of an pooling layer in a GNN */
typedef struct GNN_Layerinfo_Pool
{
   int                   ninputfeatures;     /**< number of input features */
   int                   noutputfeatures;    /**< number of output features */
   GNN_POOLTYPE          type;               /**< type of pooling layer */
} GNN_LAYERINFO_POOL;

/** information of an input layer in a GNN */
typedef struct GNN_Layerinfo_Input
{
   int                   ninputfeatures;     /**< number of input features */
} GNN_LAYERINFO_INPUT;

/** information about a layer */
typedef union GNN_Layerinfo
{
   GNN_LAYERINFO_INPUT   inputinfo;          /**< information about input layer */
   GNN_LAYERINFO_SAGE    sageinfo;           /**< information about sage layer */
   GNN_LAYERINFO_POOL    poolinfo;           /**< information about pooling layer */
   GNN_LAYERINFO_DENSE   denseinfo;          /**< information about dense layer */
} GNN_LAYERINFO;

/** data to encode a GNN */
typedef struct GNN_Data
{
   int                   nlayers;            /**< number of layers of GNN */
   GNN_LAYERTYPE*        layertypes;         /**< array of layer types */
   GNN_LAYERINFO**       layerinfo;          /**< information about layers */
} GNN_DATA;

#ifdef __cplusplus
}
#endif

#endif
