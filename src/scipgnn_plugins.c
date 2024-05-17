/**@file   scipgnn.c
 * @brief  load plugins for SCIP-GNN
 * @author Christopher Hojny
 */

#include "scipgnn_plugins.h"

#include "scip/scipdefplugins.h"
#include "sepa_relu_nodeclassify.h"
#include "sepa_relu_robustclassify.h"


/** include basic plugins needed for SCIP-GNN */
SCIP_RETCODE includeSCIPGNNPlugins(
   SCIP*                 scip,               /**< SCIP data structure */
   GNNPROB_TYPE          gnnprobtype         /**< type of GNN problem to be solved */
   )
{
   assert( scip != NULL );

   SCIP_CALL( SCIPincludeDefaultPlugins(scip) );
   if( gnnprobtype == GNNPROB_TYPE_ROBUSTCLASSIFY )
   {
      SCIP_CALL( SCIPincludeSepaReLURobustClassify(scip) );
   }
   else if( gnnprobtype == GNNPROB_TYPE_NODECLASSIFY )
   {
      SCIP_CALL( SCIPincludeSepaReLUNodeClassify(scip) );
   }

   return SCIP_OKAY;
}
