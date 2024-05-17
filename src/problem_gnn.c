/**@file   problem_gnn.c
 * @brief  Basic setup of problems on GNNs
 * @author Christopher Hojny
 *
 * This file is responsible for building the problems on GNNs.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scip/scip.h"
#include "gnn.h"
#include "gnn_bounds.h"
#include "gnn_bounds_nodeclassify.h"
#include "gnn_bounds_robustclassify.h"
#include "problem_gnn.h"
#include "probdata_nodeclassify.h"
#include "probdata_robustclassify.h"

/* #define WRITE_PYTHON_BOUNDS */

/** creates initial model for a robust classification problem */
static
SCIP_RETCODE createModelRobustClassify(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata         /**< data about optimization problem on GNN */
   )
{
   GNNPROB_ROBUSTCLASSIFY* probinfo;
   SCIP_Bool useenhancedbounds;
   SCIP_Bool** adjacencymatrix;
   SCIP_Real** lbgnnoutputvars;
   SCIP_Real** lbauxvars;
   SCIP_Real** lbnodecontent;
   SCIP_Real** ubgnnoutputvars;
   SCIP_Real** ubauxvars;
   SCIP_Real** ubnodecontent;
   int* localbudget;
   int globalbudget;
   int nnodes;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(gnnprobdata != NULL);
   assert(SCIPgetGNNProbType(gnnprobdata) == GNNPROB_TYPE_ROBUSTCLASSIFY);

   probinfo = SCIPgetGNNProbDataRobustClassify(gnnprobdata);
   assert(probinfo != NULL);

   adjacencymatrix = SCIPgetAdjacencyMatrixRobustClassify(probinfo);
   localbudget = SCIPgetLocalBudgetRobustClassify(probinfo);
   globalbudget = SCIPgetGlobalBudgetRobustClassify(probinfo);
   nnodes = SCIPgetNNodesRobustClassify(probinfo);
   SCIP_CALL( SCIPgetBoolParam(scip, "gnn/robustclassify/useenhancedbounds", &useenhancedbounds) );

   /* compute bounds on variables */
   if( useenhancedbounds )
   {
      SCIP_CALL( SCIPcomputeBoundsGNNRobustClassify(scip, gnndata, adjacencymatrix, nnodes, globalbudget, localbudget,
            NULL, NULL, probinfo->featurelb, probinfo->featureub, &lbgnnoutputvars, &lbauxvars, &lbnodecontent,
            &ubgnnoutputvars, &ubauxvars, &ubnodecontent) );
   }
   else
   {
      SCIP_Real** lbs = NULL;
      SCIP_Real** ubs = NULL;

      SCIP_CALL( SCIPgetBoolParam(scip, "gnn/robustclassify/useinputbasedbounds", &useenhancedbounds) );

      if( useenhancedbounds )
      {
         lbs = probinfo->featurelb;
         ubs = probinfo->featureub;
      }

      SCIP_CALL( SCIPcomputeBoundsGNN(scip, FALSE, gnndata, nnodes, lbs, ubs, &lbgnnoutputvars, &lbauxvars,
            &lbnodecontent, &ubgnnoutputvars, &ubauxvars, &ubnodecontent) );
   }

   /*
    * set problem specific parameters
    */

   /* terminate early if the instance is proven to be (non-) robust */
   SCIP_CALL( SCIPsetRealParam(scip, "limits/gap", SCIPinfinity(scip)) );

#ifdef WRITE_PYTHON_BOUNDS
   SCIP_CALL( SCIPwriteBoundsPython("debug_bounds.py", gnndata, gnnprobdata,
         lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent) );
#endif

   SCIP_CALL( SCIPprobdataCreateRobustClassify(scip, gnndata, gnnprobdata,
         lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent) );

   /* free bounds (not needed anymore) */
   SCIP_CALL( SCIPfreeBoundsGNN(scip, gnndata, nnodes,
         &lbgnnoutputvars, &lbauxvars, &lbnodecontent, &ubgnnoutputvars, &ubauxvars, &ubnodecontent) );

   return SCIP_OKAY;
}


/** creates initial model for a node classification problem */
static
SCIP_RETCODE createModelNodeClassify(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata         /**< data about optimization problem on GNN */
   )
{
   GNNPROB_NODECLASSIFY* probinfo;
   SCIP_Bool useenhancedbounds;
   SCIP_Bool** adjacencymatrix;
   SCIP_Real** lbgnnoutputvars;
   SCIP_Real** lbauxvars;
   SCIP_Real** lbnodecontent;
   SCIP_Real** ubgnnoutputvars;
   SCIP_Real** ubauxvars;
   SCIP_Real** ubnodecontent;
   int localbudget;
   int globalbudget;
   int nnodes;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(gnnprobdata != NULL);
   assert(SCIPgetGNNProbType(gnnprobdata) == GNNPROB_TYPE_NODECLASSIFY);

   probinfo = SCIPgetGNNProbDataNodeClassify(gnnprobdata);
   assert(probinfo != NULL);

   adjacencymatrix = SCIPgetAdjacencyMatrixNodeClassify(probinfo);
   localbudget = SCIPgetLocalBudgetNodeClassify(probinfo);
   globalbudget = SCIPgetGlobalBudgetNodeClassify(probinfo);
   nnodes = SCIPgetNNodesNodeClassify(probinfo);
   SCIP_CALL( SCIPgetBoolParam(scip, "gnn/nodeclassify/useenhancedbounds", &useenhancedbounds) );

   /* compute bounds on variables */
   if( useenhancedbounds )
   {
      SCIP_CALL( SCIPcomputeBoundsGNNNodeClassify(scip, gnndata, adjacencymatrix, nnodes, globalbudget, localbudget,
            NULL, NULL, probinfo->featurelb, probinfo->featureub, &lbgnnoutputvars, &lbauxvars, &lbnodecontent,
            &ubgnnoutputvars, &ubauxvars, &ubnodecontent) );
   }
   else
   {
      SCIP_Real** lbs = NULL;
      SCIP_Real** ubs = NULL;

      SCIP_CALL( SCIPgetBoolParam(scip, "gnn/nodeclassify/useinputbasedbounds", &useenhancedbounds) );

      if( useenhancedbounds )
      {
         lbs = probinfo->featurelb;
         ubs = probinfo->featureub;
      }

      SCIP_CALL( SCIPcomputeBoundsGNN(scip, TRUE, gnndata, nnodes, lbs, ubs, &lbgnnoutputvars, &lbauxvars, &lbnodecontent,
            &ubgnnoutputvars, &ubauxvars, &ubnodecontent) );
   }

   /*
    * set problem specific parameters
    */

   /* terminate early if the instance is proven to be (non-) robust */
   SCIP_CALL( SCIPsetRealParam(scip, "limits/gap", SCIPinfinity(scip)) );

#ifdef WRITE_PYTHON_BOUNDS
   SCIP_CALL( SCIPwriteBoundsPython("debug_bounds.py", gnndata, gnnprobdata,
         lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent) );
#endif

   SCIP_CALL( SCIPprobdataCreateNodeClassify(scip, gnndata, gnnprobdata,
         lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent) );

   /* free bounds (not needed anymore) */
   SCIP_CALL( SCIPfreeBoundsGNN(scip, gnndata, nnodes,
         &lbgnnoutputvars, &lbauxvars, &lbnodecontent, &ubgnnoutputvars, &ubauxvars, &ubnodecontent) );

   return SCIP_OKAY;
}

/** creates initial model for a problem on a GNN */
SCIP_RETCODE SCIPcreateModel(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata         /**< data about optimization problem on GNN */
   )
{
   GNNPROB_TYPE probtype;

   assert(gnnprobdata != NULL);

   probtype = SCIPgetGNNProbType(gnnprobdata);

   switch( probtype )
   {
   case GNNPROB_TYPE_ROBUSTCLASSIFY:
      SCIP_CALL( createModelRobustClassify(scip, gnndata, gnnprobdata) );
      break;
   case GNNPROB_TYPE_NODECLASSIFY:
      SCIP_CALL( createModelNodeClassify(scip, gnndata, gnnprobdata) );
      break;
   default:
      assert(FALSE);
   }

   return SCIP_OKAY;
}
