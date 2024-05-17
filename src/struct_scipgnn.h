/**@file   struct_scipgnn.h
 * @brief  structs for links between SCIP and GNN data structures
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_STRUCT_SCIPGNN_H_
#define __SCIP_STRUCT_SCIPGNN_H_

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** information about variables in a GNN model */
typedef struct SCIPGNN_Var
{
   SCIP_Bool             isfixed;            /**< whether the variable has been fixed */
   SCIP_Real             fixedval;           /**< value to which variable is fixed (if it is fixed) */
   SCIP_VAR*             var;                /**< pointer to variable (if it is not fixed) */
} SCIPGNN_VAR;

#ifdef __cplusplus
}
#endif

#endif
