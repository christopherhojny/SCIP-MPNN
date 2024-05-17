/**@file   scipparams.c
 * @brief  create params for SCIP-GNN
 * @author Christopher Hojny
 */

#include "scipgnn_params.h"

#define DEFAULT_ROBUSTCLASSIFY_USEENHANCEDBOUNDS   TRUE
#define DEFAULT_ROBUSTCLASSIFY_USEINPUTBASEDBOUNDS TRUE
#define DEFAULT_NODECLASSIFY_USEENHANCEDBOUNDS     TRUE
#define DEFAULT_NODECLASSIFY_USEINPUTBASEDBOUNDS   TRUE
#define DEFAULT_ONLYWRITEMODEL                       ""
#define DEFAULT_ONLYCHECKSOL                         ""

/** create parameters for SCIP-GNN */
SCIP_RETCODE createSCIPGNNParams(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);

   SCIP_CALL( SCIPaddBoolParam(scip, "gnn/robustclassify/useenhancedbounds",
         "whether enhanced bounds adapted to the robust classification problem shall be used",
         NULL, FALSE, DEFAULT_ROBUSTCLASSIFY_USEENHANCEDBOUNDS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "gnn/robustclassify/useinputbasedbounds",
         "whether bounds for all variables shall take bounds on input into account",
         NULL, FALSE, DEFAULT_ROBUSTCLASSIFY_USEINPUTBASEDBOUNDS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "gnn/nodeclassify/useenhancedbounds",
         "whether enhanced bounds adapted to the node classification problem shall be used",
         NULL, FALSE, DEFAULT_NODECLASSIFY_USEENHANCEDBOUNDS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "gnn/nodeclassify/useinputbasedbounds",
         "whether bounds for all variables shall take bounds on input into account",
         NULL, FALSE, DEFAULT_NODECLASSIFY_USEINPUTBASEDBOUNDS, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip, "gnn/onlywritemodel",
         "path to which  model shall be written (if non-empty)",
         NULL, FALSE, DEFAULT_ONLYWRITEMODEL, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip, "gnn/onlychecksol",
         "path of solution that needs to be checked for feasibility (if non-empty)",
         NULL, FALSE, DEFAULT_ONLYCHECKSOL, NULL, NULL) );

   return SCIP_OKAY;
}
