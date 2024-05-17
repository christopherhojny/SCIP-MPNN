/**@file   sepa_relu_robustclassify.h
 * @brief  separator for strengthened ReLU linearization cuts for robust classification problems
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_SEPA_RELU_ROBUSTCLASSIFY_H__
#define __SCIP_SEPA_RELU_ROBUSTCLASSIFY_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the ReLU linearization separator and includes it in SCIP */
SCIP_RETCODE SCIPincludeSepaReLURobustClassify(
   SCIP*                 scip                /**< SCIP data structure */
   );

#ifdef __cplusplus
}
#endif

#endif
