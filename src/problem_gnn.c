/**@file   problem_gnn.c
 * @brief  Basic setup of problems on GNNs
 * @author Christopher Hojny
 *
 * This file is responsible for building the problems on GNNs.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scipgnn_plugins.h"
#include "gnn.h"
#include "gnn_bounds.h"
#include "gnn_bounds_nodeclassify.h"
#include "gnn_bounds_robustclassify.h"
#include "problem_gnn.h"
#include "probdata_nodeclassify.h"
#include "probdata_robustclassify.h"
#include <time.h>

/* #define WRITE_PYTHON_BOUNDS */


/** gets information about layer from GNN data */
static
SCIP_RETCODE getDataLayer(
   GNN_DATA*             gnndata,            /**< data of GNN */
   int                   nnodes,             /**< number of nodes in underlying graph */
   int                   layeridx,           /**< index of layer for which data shall be extracted */
   GNN_LAYERTYPE*        layertype,          /**< pointer to store type of layer */
   GNN_ACTIVATIONTYPE*   activation,         /**< pointer to store type of activation */
   int*                  noutputfeatures,    /**< pointer to store number of output features */
   int*                  ngnnoutputvars,     /**< pointer to store number of gnnoutput variables */
   int*                  nauxvars            /**< pointer to store number of auxiliary variables */
   )
{
   assert(gnndata != NULL);
   assert(nnodes > 0);
   assert(0 <= layeridx && layeridx < SCIPgetGNNNLayers(gnndata));
   assert(layertype != NULL);
   assert(noutputfeatures != NULL);
   assert(ngnnoutputvars != NULL);
   assert(nauxvars != NULL);
   assert(0 <= layeridx && layeridx < SCIPgetGNNNLayers(gnndata));

   *layertype = SCIPgetGNNLayerType(gnndata, layeridx);
   switch( *layertype )
   {
   case GNN_LAYERTYPE_INPUT:
      *noutputfeatures = SCIPgetNInputFeaturesInputLayer(SCIPgetGNNLayerinfoInput(gnndata, layeridx));
      *ngnnoutputvars = nnodes * (*noutputfeatures);
      *nauxvars = nnodes * nnodes * (*noutputfeatures);
      *activation = GNN_ACTIVATIONTYPE_NONE;
      break;
   case GNN_LAYERTYPE_SAGE:
      *noutputfeatures = SCIPgetNOutputFeaturesSageLayer(SCIPgetGNNLayerinfoSage(gnndata, layeridx));
      *ngnnoutputvars = nnodes * (*noutputfeatures);
      if( layeridx < SCIPgetGNNNLayers(gnndata) - 1 && SCIPgetGNNLayerType(gnndata, layeridx+1) == GNN_LAYERTYPE_SAGE )
         *nauxvars = nnodes * nnodes * (*noutputfeatures);
      else
         *nauxvars = 0;
      *activation = SCIPgetSageLayerActivationType(SCIPgetGNNLayerinfoSage(gnndata, layeridx));
      break;
   case GNN_LAYERTYPE_POOL:
      *noutputfeatures = SCIPgetNOutputFeaturesPoolLayer(SCIPgetGNNLayerinfoPool(gnndata, layeridx));
      *ngnnoutputvars = *noutputfeatures;
      *nauxvars = 0;
      *activation = GNN_ACTIVATIONTYPE_NONE;
      break;
   case GNN_LAYERTYPE_DENSE:
      *noutputfeatures = SCIPgetNOutputFeaturesDenseLayer(SCIPgetGNNLayerinfoDense(gnndata, layeridx));
      *ngnnoutputvars = *noutputfeatures;
      *nauxvars = 0;
      *activation = SCIPgetDenseLayerActivationType(SCIPgetGNNLayerinfoDense(gnndata, layeridx));
      break;
   default:
      assert(FALSE);
   }

   return SCIP_OKAY;
}


/** creates GNN data for OBBT up to a specific layer */
static
SCIP_RETCODE createGNNDataOBBT(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN top be copied */
   GNN_DATA**            gnndataobbt,        /**< pointer to store GNN data for OBBT */
   int                   lastlayer           /**< last layer that should be copied */
   )
{
   int nlayers;
   int i;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(gnndataobbt != NULL);
   assert(0 <= lastlayer && lastlayer < gnndata->nlayers);

   nlayers = gnndata->nlayers;

   SCIP_CALL( SCIPallocBlockMemory(scip, gnndataobbt) );
   (*gnndataobbt)->nlayers = nlayers;

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(*gnndataobbt)->layertypes, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(*gnndataobbt)->layerinfo, nlayers) );

   for( i = 0; i < nlayers; ++i )
      (*gnndataobbt)->layertypes[i] = gnndata->layertypes[i];

   for( i = 0; i < nlayers; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemory(scip, &((*gnndataobbt)->layerinfo[i])) );

      if( gnndata->layertypes[i] == GNN_LAYERTYPE_INPUT )
      {
         (*gnndataobbt)->layerinfo[i]->inputinfo.ninputfeatures = gnndata->layerinfo[i]->inputinfo.ninputfeatures;
      }
      else if( gnndata->layertypes[i] == GNN_LAYERTYPE_SAGE )
      {
         GNN_LAYERINFO_SAGE* sageinfo;
         int ninput;
         int noutput;
         int j;
         int k;

         sageinfo = &(*gnndataobbt)->layerinfo[i]->sageinfo;
         ninput = gnndata->layerinfo[i]->sageinfo.ninputfeatures;
         noutput = gnndata->layerinfo[i]->sageinfo.noutputfeatures;
         sageinfo->ninputfeatures = ninput;
         sageinfo->noutputfeatures = noutput;
         sageinfo->activation = gnndata->layerinfo[i]->sageinfo.activation;

         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->nodeweights, noutput) );
         for( j = 0; j < noutput; ++j )
         {
            SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->nodeweights[j], ninput) );
            for( k = 0; k < ninput; ++k )
               sageinfo->nodeweights[j][k] = gnndata->layerinfo[i]->sageinfo.nodeweights[j][k];
         }

         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->edgeweights, noutput) );
         for( j = 0; j < noutput; ++j )
         {
            SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->edgeweights[j], ninput) );
            for( k = 0; k < ninput; ++k )
               sageinfo->edgeweights[j][k] = gnndata->layerinfo[i]->sageinfo.edgeweights[j][k];
         }

         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->bias, noutput) );
         for( j = 0; j < noutput; ++j )
            sageinfo->bias[j] = gnndata->layerinfo[i]->sageinfo.bias[j];
      }
      else if( gnndata->layertypes[i] == GNN_LAYERTYPE_POOL )
      {
         (*gnndataobbt)->layerinfo[i]->poolinfo.ninputfeatures = gnndata->layerinfo[i]->poolinfo.ninputfeatures;
         (*gnndataobbt)->layerinfo[i]->poolinfo.noutputfeatures = gnndata->layerinfo[i]->poolinfo.noutputfeatures;
         (*gnndataobbt)->layerinfo[i]->poolinfo.type = gnndata->layerinfo[i]->poolinfo.type;
      }
      else
      {
         GNN_LAYERINFO_DENSE* denseinfo;
         int ninput;
         int noutput;
         int j;
         int k;

         denseinfo = &(*gnndataobbt)->layerinfo[i]->denseinfo;
         ninput = gnndata->layerinfo[i]->denseinfo.ninputfeatures;
         noutput = gnndata->layerinfo[i]->denseinfo.noutputfeatures;
         denseinfo->ninputfeatures = ninput;
         denseinfo->noutputfeatures = noutput;
         denseinfo->activation = gnndata->layerinfo[i]->denseinfo.activation;

         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &denseinfo->weights, noutput) );
         for( j = 0; j < noutput; ++j )
         {
            SCIP_CALL( SCIPallocBlockMemoryArray(scip, &denseinfo->weights[j], ninput) );
            for( k = 0; k < ninput; ++k )
               denseinfo->weights[j][k] = gnndata->layerinfo[i]->denseinfo.weights[j][k];
         }

         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &denseinfo->bias, noutput) );
         for( j = 0; j < noutput; ++j )
            denseinfo->bias[j] = gnndata->layerinfo[i]->denseinfo.bias[j];
      }
   }

   return SCIP_OKAY;
}

/** frees data of a GNN sage layer */
static
SCIP_RETCODE freeGNNLayerinfoSage(
   SCIP*                 scip,               //!< SCIP pointer
   GNN_LAYERINFO**       layerinfo           //!< pointer to information about a single layer
   )
{
   int ninput;
   int noutput;

   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(*layerinfo != NULL);

   ninput = (*layerinfo)->sageinfo.ninputfeatures;
   noutput = (*layerinfo)->sageinfo.noutputfeatures;

   for( int i = 0; i < noutput; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &(*layerinfo)->sageinfo.nodeweights[i], ninput);
   }
   SCIPfreeBlockMemoryArray(scip, &(*layerinfo)->sageinfo.nodeweights, noutput);
   for( int i = 0; i < noutput; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &(*layerinfo)->sageinfo.edgeweights[i], ninput);
   }
   SCIPfreeBlockMemoryArray(scip, &(*layerinfo)->sageinfo.edgeweights, noutput);
   SCIPfreeBlockMemoryArray(scip, &(*layerinfo)->sageinfo.bias, noutput);

   return SCIP_OKAY;
}

/** frees data of a GNN dense layer */
static
SCIP_RETCODE freeGNNLayerinfoDense(
   SCIP*                 scip,               //!< SCIP pointer
   GNN_LAYERINFO**       layerinfo           //!< pointer to information about a single layer
   )
{
   int ninput;
   int noutput;

   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(*layerinfo != NULL);

   ninput = (*layerinfo)->denseinfo.ninputfeatures;
   noutput = (*layerinfo)->denseinfo.noutputfeatures;

   for( int i = 0; i < noutput; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &(*layerinfo)->denseinfo.weights[i], ninput);
   }
   SCIPfreeBlockMemoryArray(scip, &(*layerinfo)->denseinfo.weights, noutput);
   SCIPfreeBlockMemoryArray(scip, &(*layerinfo)->denseinfo.bias, noutput);

   return SCIP_OKAY;
}

/** frees data of a GNN layer */
static
SCIP_RETCODE freeGNNLayerinfo(
   SCIP*                 scip,               //!< SCIP pointer
   GNN_LAYERTYPE         layertype,          //!< type of layer to be freed
   GNN_LAYERINFO**       layerinfo           //!< pointer to information about a single layer
   )
{
   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(*layerinfo != NULL);

   switch( layertype )
   {
   case GNN_LAYERTYPE_SAGE:
      SCIP_CALL( freeGNNLayerinfoSage(scip, layerinfo) );
      break;
   case GNN_LAYERTYPE_DENSE:
      SCIP_CALL( freeGNNLayerinfoDense(scip, layerinfo) );
      break;
   default:
      // no separate data to be freed here
      assert(layertype == GNN_LAYERTYPE_INPUT || layertype == GNN_LAYERTYPE_POOL);
   }

   SCIPfreeBlockMemory(scip, layerinfo);

   return SCIP_OKAY;
}

/** frees GNN data */
static
SCIP_RETCODE freeGNNDataOBBT(
   SCIP*                 scip,               //!< SCIP pointer
   GNN_DATA*             gnndata             //!< pointer to data of GNN
   )
{
   assert(scip != NULL);
   assert(gnndata != NULL);

   // free the different layers
   for( int l = 0; l < gnndata->nlayers; ++l )
   {
      SCIP_CALL( freeGNNLayerinfo(scip, gnndata->layertypes[l], &gnndata->layerinfo[l]) );
   }

   SCIPfreeBlockMemoryArray(scip, &gnndata->layertypes, gnndata->nlayers);
   SCIPfreeBlockMemoryArray(scip, &gnndata->layerinfo, gnndata->nlayers);
   SCIPfreeBlockMemory(scip, &gnndata);

   return SCIP_OKAY;
}

/** creates initial model for a robust classification problem */
static
SCIP_RETCODE createModelRobustClassify(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata,        /**< data about optimization problem on GNN */
   char*                 problemname         /**< name of the problem to be solved */
   )
{
   FILE* fp;
   GNNPROB_ROBUSTCLASSIFY* probinfo;
   clock_t starttime;
   clock_t endtime;
   SCIP_Real totaltime;
   SCIP_Bool useobbt;
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
   SCIP_Bool writebounds;

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
   SCIP_CALL( SCIPgetBoolParam(scip, "gnn/robustclassify/useobbt", &useobbt) );
   SCIP_CALL( SCIPgetBoolParam(scip, "gnn/robustclassify/writebounds", &writebounds) );

   /* compute bounds on variables */
   if( useobbt )
   {
      int nobbt = 0;
      int l;

      SCIP_CALL( SCIPcomputeBoundsGNNRobustClassify(scip, gnndata, adjacencymatrix, nnodes, globalbudget, localbudget,
            NULL, NULL, probinfo->featurelb, probinfo->featureub, &lbgnnoutputvars, &lbauxvars, &lbnodecontent,
            &ubgnnoutputvars, &ubauxvars, &ubnodecontent) );

      /*
       * iterate through all layers (except for layer 0) and maximize/minimize bounds of each neuron
       */
      starttime = clock();

      if( writebounds )
      {
         char filename[SCIP_MAXSTRLEN];

         strcpy(filename, problemname);
         strcat(filename, ".csv");
         fp = fopen(filename, "w");
         fprintf(fp, "method,layeridx,nodeidx,featureidx,lb,ub\n");
      }

      /* iterate through the layers (except for the input layer) */
      for( l = 1; l < SCIPgetGNNNLayers(gnndata); ++l )
      {
         SCIP* subscip;
         GNN_DATA* gnndataobbt;
         GNN_LAYERTYPE type;
         GNN_ACTIVATIONTYPE activation;
         int nfeatures;
         int ngnnoutputvars;
         int nauxvars;
         int nodebound;
         int v;
         int f;

         SCIP_CALL( getDataLayer(gnndata, nnodes, l, &type, &activation, &nfeatures, &ngnnoutputvars, &nauxvars) );

         if( type == GNN_LAYERTYPE_SAGE )
            nodebound = nnodes;
         else if( type == GNN_LAYERTYPE_DENSE )
            nodebound = 1;
         else
            continue;

         SCIP_CALL( SCIPcreate(&subscip) );
         SCIP_CALL( includeSCIPGNNPlugins(subscip, SCIPgetGNNProbType(gnnprobdata)) );

         SCIPsetMessagehdlrQuiet(subscip, TRUE);

         /* set up model with previously computed bounds */
         SCIP_CALL( createGNNDataOBBT(scip, gnndata, &gnndataobbt, l) );

         SCIP_CALL( SCIPprobdataCreateRobustClassify(subscip, gnndataobbt, gnnprobdata,
               lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent, TRUE) );

         /* iterate through each neuron in the layer */
         for( v = 0; v < nodebound; ++v )
         {
            for( f = 0; f < nfeatures; ++f )
            {
               SCIP* subsubscip;
               SCIP_Real objval;
               char name[SCIP_MAXSTRLEN];

               if( writebounds )
                  fprintf(fp, "SBT,%d,%d,%d,%f,%f\n", l, v, f,
                     lbgnnoutputvars[l][v*nfeatures + f], ubgnnoutputvars[l][v*nfeatures + f]);

               /* skip bound improvements if bounds are already tight */
               if( SCIPisEQ(scip, ubgnnoutputvars[l][v*nfeatures + f], lbgnnoutputvars[l][v*nfeatures + f]) )
               {
                  if( writebounds )
                     fprintf(fp, "OBBT,%d,%d,%d,%f,%f\n", l, v, f,
                        lbgnnoutputvars[l][v*nfeatures + f], ubgnnoutputvars[l][v*nfeatures + f]);
                  continue;
               }

               /* maximize the value at the current neuron and update bound */
               SCIP_CALL( SCIPsetOBBTobjective(subscip, nfeatures, l, v, f, TRUE) );

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "tmp_%s.mps", problemname);
               SCIP_CALL( SCIPwriteOrigProblem(subscip, name, "mps", FALSE) );
               SCIP_CALL( SCIPcreate(&subsubscip) );
               SCIP_CALL( SCIPincludeDefaultPlugins(subsubscip) );
               SCIP_CALL( SCIPreadProb(subsubscip, name, "mps") );
               SCIP_CALL( SCIPsolve(subsubscip) );
               objval = SCIPgetSolOrigObj(subsubscip, SCIPgetBestSol(subsubscip));
               SCIP_CALL( SCIPfreeTransform(subsubscip) );
               SCIP_CALL( SCIPfree(&subsubscip) );

               /* extract objective value */
               ubgnnoutputvars[l][v*nfeatures + f] = MIN(ubgnnoutputvars[l][v*nfeatures + f], objval + 0.01);

               /* build model anew and minimize value at current neuron */
               SCIP_CALL( SCIPsetOBBTobjective(subscip, nfeatures, l, v, f, FALSE) );

               SCIP_CALL( SCIPwriteOrigProblem(subscip, name, "mps", FALSE) );
               SCIP_CALL( SCIPcreate(&subsubscip) );
               SCIP_CALL( SCIPincludeDefaultPlugins(subsubscip) );
               SCIP_CALL( SCIPreadProb(subsubscip, name, "mps") );
               SCIP_CALL( SCIPsolve(subsubscip) );
               objval = SCIPgetSolOrigObj(subsubscip, SCIPgetBestSol(subsubscip));
               SCIP_CALL( SCIPfreeTransform(subsubscip) );
               SCIP_CALL( SCIPfree(&subsubscip) );

               /* SCIP_CALL( SCIPsolve(subscip) ); */

               /* update bound */
               lbgnnoutputvars[l][v*nfeatures + f] = MAX(lbgnnoutputvars[l][v*nfeatures + f], objval - 0.01);

               SCIP_CALL( SCIPresetOBBTobjective(subscip, nfeatures, l, v, f) );

               ++nobbt;

               if( writebounds )
               {
                  fprintf(fp, "OBBT,%d,%d,%d,%f,%f\n", l, v, f,
                     lbgnnoutputvars[l][v*nfeatures + f], ubgnnoutputvars[l][v*nfeatures + f]);
               }
            }
         }

         SCIP_CALL( freeGNNDataOBBT(scip, gnndataobbt) );
      }

      endtime = clock();

      totaltime = ((double) (endtime - starttime)) / CLOCKS_PER_SEC;
      printf("time OBBT: %f (%f)\n", totaltime, totaltime / nobbt);

      if( writebounds )
         fclose(fp);
   }
   else if( useenhancedbounds )
   {
      starttime = clock();
      SCIP_CALL( SCIPcomputeBoundsGNNRobustClassify(scip, gnndata, adjacencymatrix, nnodes, globalbudget, localbudget,
            NULL, NULL, probinfo->featurelb, probinfo->featureub, &lbgnnoutputvars, &lbauxvars, &lbnodecontent,
            &ubgnnoutputvars, &ubauxvars, &ubnodecontent) );
      endtime = clock();
      totaltime = ((double) (endtime - starttime)) / CLOCKS_PER_SEC;
      printf("time SBT: %f\n", totaltime);

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

      starttime = clock();
      SCIP_CALL( SCIPcomputeBoundsGNN(scip, FALSE, gnndata, nnodes, lbs, ubs, &lbgnnoutputvars, &lbauxvars,
            &lbnodecontent, &ubgnnoutputvars, &ubauxvars, &ubnodecontent) );
      endtime = clock();
      totaltime = ((double) (endtime - starttime)) / CLOCKS_PER_SEC;
      printf("time naive bounds: %f\n", totaltime);
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

   starttime = clock();
   SCIP_CALL( SCIPprobdataCreateRobustClassify(scip, gnndata, gnnprobdata,
         lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent, FALSE) );
   endtime = clock();
   totaltime = ((double) (endtime - starttime)) / CLOCKS_PER_SEC;
   printf("time model creation: %f\n", totaltime);

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
   GNNPROB_DATA*         gnnprobdata,        /**< data about optimization problem on GNN */
   char*                 problemname         /**< name of the problem to be solved */
   )
{
   GNNPROB_TYPE probtype;

   assert(gnnprobdata != NULL);

   probtype = SCIPgetGNNProbType(gnnprobdata);

   switch( probtype )
   {
   case GNNPROB_TYPE_ROBUSTCLASSIFY:
      SCIP_CALL( createModelRobustClassify(scip, gnndata, gnnprobdata, problemname) );
      break;
   case GNNPROB_TYPE_NODECLASSIFY:
      SCIP_CALL( createModelNodeClassify(scip, gnndata, gnnprobdata) );
      break;
   default:
      assert(FALSE);
   }

   return SCIP_OKAY;
}
