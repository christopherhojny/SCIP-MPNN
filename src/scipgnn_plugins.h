/**@file   scipgnn.h
 * @brief  load plugins for SCIP-GNN
 * @author Christopher Hojny
 */

#ifndef __SCIPGNNPLUGINS_H__
#define __SCIPGNNPLUGINS_H__

// SCIP include
#include <scip/scip.h>
#include "type_problem.h"


#ifdef __cplusplus
extern "C" {
#endif

/** include basic plugins needed for SCIP-GNN */
extern
SCIP_RETCODE includeSCIPGNNPlugins(
   SCIP*                 scip,               /**< SCIP data structure */
   GNNPROB_TYPE          gnnprobtype         /**< type of GNN problem to be solved */
   );

#ifdef __cplusplus
}
#endif

#endif
