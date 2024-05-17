/**@file   scipparams.h
 * @brief  create params for SCIP-GNN
 * @author Christopher Hojny
 */

#ifndef __SCIPGNNPARAMS_H__
#define __SCIPGNNPARAMS_H__

// SCIP include
#include <scip/scip.h>


#ifdef __cplusplus
extern "C" {
#endif

/** create parameters for SCIP-GNN */
extern
SCIP_RETCODE createSCIPGNNParams(
   SCIP*                 scip                /**< SCIP data structure */
   );

#ifdef __cplusplus
}
#endif

#endif
