/**@file   type_gnn.h
 * @brief  type definitions for GNNs
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_TYPE_GNN_H_
#define __SCIP_TYPE_GNN_H_

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** define type of activation functions in GNN */
enum GNN_Activationtype
{
   GNN_ACTIVATIONTYPE_NONE = 0,                /**< no activarion */
   GNN_ACTIVATIONTYPE_RELU = 1,                /**< ReLU activation */
};
typedef enum GNN_Activationtype GNN_ACTIVATIONTYPE;

/** define type of layers in GNN */
enum GNN_Layertype
{
   GNN_LAYERTYPE_INPUT   = 0,                /**< input layer */
   GNN_LAYERTYPE_SAGE    = 1,                /**< sage layer */
   GNN_LAYERTYPE_POOL    = 2,                /**< pooling layer */
   GNN_LAYERTYPE_DENSE   = 3                 /**< dense layer */
};
typedef enum GNN_Layertype GNN_LAYERTYPE;

/** define type of pooling layers*/
enum GNN_Pooltype
{
   GNN_POOLTYPE_ADD      = 0                 /**< add pooling layer */
};
typedef enum GNN_Pooltype GNN_POOLTYPE;

#ifdef __cplusplus
}
#endif

#endif
