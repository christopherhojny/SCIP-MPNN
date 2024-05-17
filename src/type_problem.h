/**@file   type_gnn.h
 * @brief  type definitions for problems on GNNs
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_TYPE_PROBLEM_H_
#define __SCIP_TYPE_PROBLEM_H_

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** define type of problems on GNNs */
enum GNNProb_Type
{
   GNNPROB_TYPE_ROBUSTCLASSIFY = 0,          /**< robust classification */
   GNNPROB_TYPE_NODECLASSIFY   = 1           /**< node classification */
};
typedef enum GNNProb_Type GNNPROB_TYPE;

#ifdef __cplusplus
}
#endif

#endif
