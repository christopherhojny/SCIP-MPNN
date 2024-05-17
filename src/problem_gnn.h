/**@file   problem_gnn.h
 * @brief  Basic setup of problems on GNNs
 * @author Christopher Hojny
 *
 * This file is responsible for building the problems on GNNs.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __PROBLEM_GNN_H__
#define __PROBLEM_GNN_H__

#include "scip/scip.h"
#include "struct_gnn.h"
#include "struct_problem.h"
#include "type_gnn.h"
#include "type_problem.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates initial model for a problem on a GNN */
extern
SCIP_RETCODE SCIPcreateModel(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata         /**< data about optimization problem on GNN */
   );

#ifdef __cplusplus
}
#endif

#endif
