/**@file   read_gnn.cpp
 * @brief  functions to read GNN problems
 * @author Christopher Hojny
 */

#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include "struct_gnn.h"
#include "struct_problem.h"
#include "type_gnn.h"
#include "type_problem.h"
#include "read_gnn.h"

using namespace std;

/** reads information about an input layer and stores its data */
static
SCIP_RETCODE readInfoInputlayer(
   ifstream&             inputstream,        //!< input stream for getting problem data
   GNN_LAYERINFO*        layerinfo           //!< pointer to information about single layer
   )
{
   int ninput;

   assert(layerinfo != NULL);

   inputstream >> ninput;
   assert(ninput > 0);

   layerinfo->inputinfo.ninputfeatures = ninput;

   return SCIP_OKAY;
}

/** reads information about a sage layer and stores its data */
static
SCIP_RETCODE readInfoSagelayer(
   SCIP*                 scip,               //!< SCIP data structure
   ifstream&             inputstream,        //!< input stream for getting problem data
   GNN_LAYERINFO*        layerinfo           //!< pointer to information about single layer
   )
{
   GNN_LAYERINFO_SAGE* sageinfo;
   string info;
   SCIP_Real val;
   int ninput;
   int noutput;

   assert(scip != NULL);
   assert(layerinfo != NULL);

   inputstream >> ninput;
   inputstream >> noutput;

   assert(ninput > 0);
   assert(noutput > 0);

   inputstream >> info;
   assert(info == "none" || info == "relu");

   sageinfo = &layerinfo->sageinfo;
   sageinfo->ninputfeatures = ninput;
   sageinfo->noutputfeatures = noutput;
   if( info == "none" )
      sageinfo->activation = GNN_ACTIVATIONTYPE_NONE;
   else
      sageinfo->activation = GNN_ACTIVATIONTYPE_RELU;

   inputstream >> info;
   assert(info == "nodeweights");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->nodeweights, noutput) );
   for( int i = 0; i < noutput; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->nodeweights[i], ninput) );
      for( int j = 0; j < ninput; ++j )
      {
         inputstream >> val;
         sageinfo->nodeweights[i][j] = val;
      }
   }

   inputstream >> info;
   assert(info == "edgeweights");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->edgeweights, noutput) );
   for( int i = 0; i < noutput; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->edgeweights[i], ninput) );
      for( int j = 0; j < ninput; ++j )
      {
         inputstream >> val;
         sageinfo->edgeweights[i][j] = val;
      }
   }

   inputstream >> info;
   assert(info == "bias");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &sageinfo->bias, noutput) );
   for( int i = 0; i < noutput; ++i )
   {
      inputstream >> val;
      sageinfo->bias[i] = val;
   }

   return SCIP_OKAY;
}

/** reads information about a pooling layer and stores its data */
static
SCIP_RETCODE readInfoPoollayer(
   ifstream&             inputstream,        //!< input stream for getting problem data
   GNN_LAYERINFO*        layerinfo,          //!< pointer to information about single layer
   GNN_POOLTYPE          pooltype,           //!< type of pooling layer
   int                   nfeatures           //!< number of features of pooling layer
   )
{
   assert(layerinfo != NULL);

   layerinfo->poolinfo.ninputfeatures = nfeatures;
   layerinfo->poolinfo.noutputfeatures = nfeatures;
   layerinfo->poolinfo.type = pooltype;

   return SCIP_OKAY;
}

/** reads information about a dense layer and stores its data */
static
SCIP_RETCODE readInfoDenselayer(
   SCIP*                 scip,               //!< SCIP data structure
   ifstream&             inputstream,        //!< input stream for getting problem data
   GNN_LAYERINFO*        layerinfo           //!< pointer to information about single layer
   )
{
   GNN_LAYERINFO_DENSE* denseinfo;
   string info;
   SCIP_Real val;
   int ninput;
   int noutput;

   assert(scip != NULL);
   assert(layerinfo != NULL);

   inputstream >> ninput;
   inputstream >> noutput;

   assert(ninput > 0);
   assert(noutput > 0);

   inputstream >> info;
   assert(info == "none" || info == "relu");

   denseinfo = &layerinfo->denseinfo;
   denseinfo->ninputfeatures = ninput;
   denseinfo->noutputfeatures = noutput;
   if( info == "none" )
      denseinfo->activation = GNN_ACTIVATIONTYPE_NONE;
   else
      denseinfo->activation = GNN_ACTIVATIONTYPE_RELU;

   inputstream >> info;
   assert(info == "denseweights");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &denseinfo->weights, noutput) );
   for( int i = 0; i < noutput; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &denseinfo->weights[i], ninput) );
      for( int j = 0; j < ninput; ++j )
      {
         inputstream >> val;
         denseinfo->weights[i][j] = val;
      }
   }

   inputstream >> info;
   assert(info == "bias");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &denseinfo->bias, noutput) );
   for( int i = 0; i < noutput; ++i )
   {
      inputstream >> val;
      denseinfo->bias[i] = val;
   }

   return SCIP_OKAY;
}

/** reads a GNN from a file and stores its data */
SCIP_RETCODE readGNN(
   SCIP*                 scip,               //!< SCIP data structure
   std::string           filename,           //!< name of file encoding GNN
   GNN_DATA**            gnndata,            //!< pointer to GNN data
   SCIP_Bool*            success             //!< pointer to store whether GNN could be read
   )
{
   string layertype;
   string activation;
   int nlayers;
#ifndef NDEBUG
   int ninput;
   int noutput;
   SCIP_Bool hasinput = FALSE;
   SCIP_Bool hassage = FALSE;
   SCIP_Bool haspool = FALSE;
   SCIP_Bool hasdense = FALSE;
#endif

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(success != NULL);

   *success = FALSE;

   ifstream inputstream(filename);
   if (!inputstream)
   {
      cout << "GNN file not found." << endl;
      return SCIP_OKAY;
   }

   // read number of data points and dimension of data points
   inputstream >> nlayers;
   assert(nlayers > 0);

   SCIP_CALL( SCIPallocBlockMemory(scip, gnndata) );

   (*gnndata)->nlayers = nlayers;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(*gnndata)->layertypes, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(*gnndata)->layerinfo, nlayers) );

   // read remaining part of file and store information about layers
   for( int l = 0; l < nlayers; ++l )
   {
      inputstream >> layertype;

      SCIP_CALL( SCIPallocBlockMemory(scip, &((*gnndata)->layerinfo[l])) );

      if( layertype == "input" )
      {
#ifndef NDEBUG
         assert(!hasinput);
         assert(!hassage);
         assert(!haspool);
         assert(!hasdense);
         hasinput = TRUE;
#endif

         (*gnndata)->layertypes[l] = GNN_LAYERTYPE_INPUT;
         SCIP_CALL( readInfoInputlayer(inputstream, (*gnndata)->layerinfo[l]) );

#ifndef NDEBUG
         // set information for consistency check in next layer
         noutput = (*gnndata)->layerinfo[l]->inputinfo.ninputfeatures;
#endif
      }
      else if( layertype == "sage" )
      {
#ifndef NDEBUG
         assert(hasinput);
         assert(!haspool);
         assert(!hasdense);
         hassage = TRUE;
#endif

         (*gnndata)->layertypes[l] = GNN_LAYERTYPE_SAGE;
         SCIP_CALL( readInfoSagelayer(scip, inputstream, (*gnndata)->layerinfo[l]) );

#ifndef NDEBUG
         // consistency checks
         ninput = (*gnndata)->layerinfo[l]->sageinfo.ninputfeatures;
         assert(ninput == noutput);
         noutput = (*gnndata)->layerinfo[l]->sageinfo.noutputfeatures;
#endif
      }
      else if( layertype == "addpool" )
      {
         int noutputfeatures;

#ifndef NDEBUG
         assert(hasinput);
         assert(hassage);
         assert(!haspool);
         assert(!hasdense);
         haspool = TRUE;
#endif

         (*gnndata)->layertypes[l] = GNN_LAYERTYPE_POOL;

         assert(l > 0);
         assert((*gnndata)->layertypes[l-1] == GNN_LAYERTYPE_SAGE);
         noutputfeatures = (*gnndata)->layerinfo[l-1]->sageinfo.noutputfeatures;

         SCIP_CALL( readInfoPoollayer(inputstream, (*gnndata)->layerinfo[l], GNN_POOLTYPE_ADD, noutputfeatures) );

#ifndef NDEBUG
         // set data for consistency checks
         noutput = (*gnndata)->layerinfo[l]->poolinfo.noutputfeatures;
#endif
      }
      else
      {
#ifndef NDEBUG
         assert(layertype == "dense");
         assert(hasinput);
         assert(haspool || !hassage);
         hasdense = TRUE;
#endif

         (*gnndata)->layertypes[l] = GNN_LAYERTYPE_DENSE;
         SCIP_CALL( readInfoDenselayer(scip, inputstream, (*gnndata)->layerinfo[l]) );

#ifndef NDEBUG
         // consistency checks
         ninput = (*gnndata)->layerinfo[l]->denseinfo.ninputfeatures;
         assert(ninput == noutput);
         noutput = (*gnndata)->layerinfo[l]->denseinfo.noutputfeatures;
#endif
      }
   }

   *success = TRUE;

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
SCIP_RETCODE freeGNNData(
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

/** prints information about a GNN to screen */
SCIP_RETCODE printGNN(
   SCIP*                 scip,               //!< SCIP pointer
   GNN_DATA*             gnndata             //!< data of GNN
   )
{
   assert(gnndata != NULL);

   SCIPinfoMessage(scip, NULL, "%d\n", gnndata->nlayers);
   for( int l = 0; l < gnndata->nlayers; ++l )
   {
      if( gnndata->layertypes[l] == GNN_LAYERTYPE_INPUT )
      {
         SCIPinfoMessage(scip, NULL, "input\n");
         SCIPinfoMessage(scip, NULL, "%d\n", gnndata->layerinfo[l]->inputinfo.ninputfeatures);
      }
      else if( gnndata->layertypes[l] == GNN_LAYERTYPE_SAGE )
      {
         SCIPinfoMessage(scip, NULL, "sage\n");
         SCIPinfoMessage(scip, NULL, "%d %d ", gnndata->layerinfo[l]->sageinfo.ninputfeatures,
            gnndata->layerinfo[l]->sageinfo.noutputfeatures);
         if( gnndata->layerinfo[l]->sageinfo.activation == GNN_ACTIVATIONTYPE_RELU )
            SCIPinfoMessage(scip, NULL, "relu\n");
         else
            SCIPinfoMessage(scip, NULL, "none\n");
         SCIPinfoMessage(scip, NULL, "nodeweights\n");
         for( int i = 0; i < gnndata->layerinfo[l]->sageinfo.noutputfeatures; ++i)
         {
            for( int j = 0; j < gnndata->layerinfo[l]->sageinfo.ninputfeatures; ++j)
            {
               SCIPinfoMessage(scip, NULL, "%20.17f", gnndata->layerinfo[l]->sageinfo.nodeweights[i][j]);
               if( j < gnndata->layerinfo[l]->sageinfo.ninputfeatures - 1 )
                  SCIPinfoMessage(scip, NULL, " ");
            }
            SCIPinfoMessage(scip, NULL, "\n");
         }
         SCIPinfoMessage(scip, NULL, "edgeweights\n");
         for( int i = 0; i < gnndata->layerinfo[l]->sageinfo.noutputfeatures; ++i)
         {
            for( int j = 0; j < gnndata->layerinfo[l]->sageinfo.ninputfeatures; ++j)
            {
               SCIPinfoMessage(scip, NULL, "%20.17f", gnndata->layerinfo[l]->sageinfo.edgeweights[i][j]);
               if( j < gnndata->layerinfo[l]->sageinfo.ninputfeatures - 1 )
                  SCIPinfoMessage(scip, NULL, " ");
            }
            SCIPinfoMessage(scip, NULL, "\n");
         }
         SCIPinfoMessage(scip, NULL, "bias\n");
         for( int i = 0; i < gnndata->layerinfo[l]->sageinfo.noutputfeatures; ++i)
         {
            SCIPinfoMessage(scip, NULL, "%20.17f\n", gnndata->layerinfo[l]->sageinfo.bias[i]);
         }
      }
      else if( gnndata->layertypes[l] == GNN_LAYERTYPE_POOL )
      {
         SCIPinfoMessage(scip, NULL, "addpool\n");
      }
      else
      {
         SCIPinfoMessage(scip, NULL, "dense\n");
         SCIPinfoMessage(scip, NULL, "%d %d ", gnndata->layerinfo[l]->denseinfo.ninputfeatures,
            gnndata->layerinfo[l]->denseinfo.noutputfeatures);
         if( gnndata->layerinfo[l]->denseinfo.activation == GNN_ACTIVATIONTYPE_RELU )
            SCIPinfoMessage(scip, NULL, "relu\n");
         else
            SCIPinfoMessage(scip, NULL, "none\n");
         SCIPinfoMessage(scip, NULL, "denseweights\n");
         for( int i = 0; i < gnndata->layerinfo[l]->denseinfo.noutputfeatures; ++i)
         {
            for( int j = 0; j < gnndata->layerinfo[l]->denseinfo.ninputfeatures; ++j)
            {
               SCIPinfoMessage(scip, NULL, "%20.17f", gnndata->layerinfo[l]->denseinfo.weights[i][j]);
               if( j < gnndata->layerinfo[l]->denseinfo.ninputfeatures - 1 )
                  SCIPinfoMessage(scip, NULL, " ");
            }
            SCIPinfoMessage(scip, NULL, "\n");
         }
         SCIPinfoMessage(scip, NULL, "bias\n");
         for( int i = 0; i < gnndata->layerinfo[l]->denseinfo.noutputfeatures; ++i)
         {
            SCIPinfoMessage(scip, NULL, "%20.17f\n", gnndata->layerinfo[l]->denseinfo.bias[i]);
         }
      }
   }

   return SCIP_OKAY;
}

/** reads a robust classification problem from a file */
static
SCIP_RETCODE readRobustClassificationProblem(
   SCIP*                 scip,               //!< SCIP data structure
   ifstream&             inputstream,        //!< input stream for getting problem data */
   GNNPROB_DATA**        gnnprobdata         //!< pointer to GNN problem data
   )
{
   GNNPROB_ROBUSTCLASSIFY* probinfo;
   string info;
   SCIP_Real rval;
   int nfeatures;
   int nnodes;
   int val;

   assert(scip != NULL);
   assert(gnnprobdata != NULL);

   SCIP_CALL( SCIPallocBlockMemory(scip, gnnprobdata) );

   (*gnnprobdata)->probtype = GNNPROB_TYPE_ROBUSTCLASSIFY;
   SCIP_CALL( SCIPallocBlockMemory(scip, &(*gnnprobdata)->probtypeinfo) );
   probinfo = &(*gnnprobdata)->probtypeinfo->robustclassifyinfo;

   // read number of nodes
   inputstream >> info;
   assert(info == "nodes");

   inputstream >> nnodes;
   assert(nnodes > 0);

   probinfo->nnodes = nnodes;

   // read adjacency matrix
   inputstream >> info;
   assert(info == "adjacencies");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->adjacencymatrix, nnodes) );
   for( int i = 0; i < nnodes; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->adjacencymatrix[i], nnodes) );
      for( int j = 0; j < nnodes; ++j )
      {
         inputstream >> val;
         probinfo->adjacencymatrix[i][j] = val;
      }
   }

   // read number of features
   inputstream >> info;
   assert(info == "features");

   inputstream >> nfeatures;
   assert(nfeatures > 0);

   probinfo->nfeatures = nfeatures;

   // read lower and upper bounds on feature assignments
   inputstream >> info;
   assert(info == "featureLB");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->featurelb, nnodes) );
   for( int i = 0; i < nnodes; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->featurelb[i], nfeatures) );
      for( int j = 0; j < nfeatures; ++j )
      {
         inputstream >> rval;
         probinfo->featurelb[i][j] = rval;
      }
   }

   inputstream >> info;
   assert(info == "featureUB");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->featureub, nnodes) );
   for( int i = 0; i < nnodes; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->featureub[i], nfeatures) );
      for( int j = 0; j < nfeatures; ++j )
      {
         inputstream >> rval;
         probinfo->featureub[i][j] = rval;
      }
   }

   // read global attack budget
   inputstream >> info;
   assert(info == "global");

   inputstream >> val;
   assert(val >= 0);

   probinfo->globalbudget = val;

   // read local attack budget
   inputstream >> info;
   assert(info == "local");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->localbudget, nnodes) );
   for( int i = 0; i < nnodes; ++i )
   {
      inputstream >> val;
      probinfo->localbudget[i] = val;
   }

   // read original and target feature
   inputstream >> info;
   assert(info == "originalclassification");

   inputstream >> val;
   assert(val >= 0);

   probinfo->graphclassification = val;

   inputstream >> info;
   assert(info == "modifiedclassification");

   inputstream >> val;
   assert(val >= 0);

   probinfo->targetclassification = val;

   return SCIP_OKAY;
}

/** reads a node classification problem from a file */
static
SCIP_RETCODE readNodeClassificationProblem(
   SCIP*                 scip,               //!< SCIP data structure
   ifstream&             inputstream,        //!< input stream for getting problem data */
   GNNPROB_DATA**        gnnprobdata         //!< pointer to GNN problem data
   )
{
   GNNPROB_NODECLASSIFY* probinfo;
   string info;
   SCIP_Real rval;
   int nfeatures;
   int nnodes;
   int val;

   assert(scip != NULL);
   assert(gnnprobdata != NULL);

   SCIP_CALL( SCIPallocBlockMemory(scip, gnnprobdata) );

   (*gnnprobdata)->probtype = GNNPROB_TYPE_NODECLASSIFY;
   SCIP_CALL( SCIPallocBlockMemory(scip, &(*gnnprobdata)->probtypeinfo) );
   probinfo = &(*gnnprobdata)->probtypeinfo->nodeclassifyinfo;

   // read number of nodes
   inputstream >> info;
   assert(info == "nodes");

   inputstream >> nnodes;
   assert(nnodes > 0);

   probinfo->nnodes = nnodes;

   // read adjacency matrix
   inputstream >> info;
   assert(info == "adjacencies");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->adjacencymatrix, nnodes) );
   for( int i = 0; i < nnodes; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->adjacencymatrix[i], nnodes) );
      for( int j = 0; j < nnodes; ++j )
      {
         inputstream >> val;
         probinfo->adjacencymatrix[i][j] = val;
      }
   }

   // read number of features
   inputstream >> info;
   assert(info == "features");

   inputstream >> nfeatures;
   assert(nfeatures > 0);

   probinfo->nfeatures = nfeatures;

   // read lower and upper bounds on feature assignments
   inputstream >> info;
   assert(info == "featureLB");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->featurelb, nnodes) );
   for( int i = 0; i < nnodes; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->featurelb[i], nfeatures) );
      for( int j = 0; j < nfeatures; ++j )
      {
         inputstream >> rval;
         probinfo->featurelb[i][j] = rval;
      }
   }

   inputstream >> info;
   assert(info == "featureUB");

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->featureub, nnodes) );
   for( int i = 0; i < nnodes; ++i )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &probinfo->featureub[i], nfeatures) );
      for( int j = 0; j < nfeatures; ++j )
      {
         inputstream >> rval;
         probinfo->featureub[i][j] = rval;
      }
   }

   // read global attack budget
   inputstream >> info;
   assert(info == "global");

   inputstream >> val;
   assert(val >= 0);

   probinfo->globalbudget = val;

   // read local attack budget
   inputstream >> info;
   assert(info == "local");

   inputstream >> val;
   probinfo->localbudget = val;

   // read node for which classification is computed
   inputstream >> info;
   assert(info == "nodeclassification");

   inputstream >> val;
   assert(val >= 0);

   probinfo->nodeclassification = val;

   // read original and target feature
   inputstream >> info;
   assert(info == "originalclassification");

   inputstream >> val;
   assert(val >= 0);

   probinfo->graphclassification = val;

   inputstream >> info;
   assert(info == "modifiedclassification");

   inputstream >> val;
   assert(val >= 0);

   probinfo->targetclassification = val;

   return SCIP_OKAY;
}

/** reads a GNN problem from a file and stores its data */
extern
SCIP_RETCODE readGNNProb(
   SCIP*                 scip,               //!< SCIP data structure
   std::string           filename,           //!< name of file encoding GNN
   GNNPROB_DATA**        gnnprobdata,        //!< pointer to GNN problem data
   SCIP_Bool*            success             //!< pointer to store whether GNN could be read
   )
{
   string info;

   assert(scip != NULL);
   assert(gnnprobdata != NULL);
   assert(success != NULL);

   *success = FALSE;

   ifstream inputstream(filename);
   if (!inputstream)
   {
      cout << "GNN problem file not found." << endl;
      return SCIP_OKAY;
   }

   // read problem type
   inputstream >> info;
   if( info == "robustclassification" )
   {
      SCIP_CALL( readRobustClassificationProblem(scip, inputstream, gnnprobdata) );
   }
   else if( info == "nodeclassification" )
   {
      SCIP_CALL( readNodeClassificationProblem(scip, inputstream, gnnprobdata) );
   }
   else
   {
      SCIPerrorMessage("stop reading file, detected unknown problem type\n");
      return SCIP_INVALIDDATA;
   }

   *success = TRUE;

   return SCIP_OKAY;
}

/** frees data of a robust classification problem */
static
SCIP_RETCODE freeRobustClassificationProblem(
   SCIP*                 scip,               //!< SCIP pointer
   GNNPROB_TYPEINFO*     probtypeinfo        //!< information about problem
   )
{
   GNNPROB_ROBUSTCLASSIFY* probinfo;
   int nnodes;
   int nfeatures;

   assert(scip != NULL);
   assert(probtypeinfo != NULL);

   probinfo = &probtypeinfo->robustclassifyinfo;
   nnodes = probinfo->nnodes;
   nfeatures = probinfo->nfeatures;

   // free adjacency matrix
   for( int i = 0; i < nnodes; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &probinfo->adjacencymatrix[i], nnodes);
   }
   SCIPfreeBlockMemoryArray(scip, &probinfo->adjacencymatrix, nnodes);

   // free bounds on feature assignments
   for( int i = 0; i < nnodes; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &probinfo->featurelb[i], nfeatures);
   }
   SCIPfreeBlockMemoryArray(scip, &probinfo->featurelb, nnodes);
   for( int i = 0; i < nnodes; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &probinfo->featureub[i], nfeatures);
   }
   SCIPfreeBlockMemoryArray(scip, &probinfo->featureub, nnodes);

   // free local attack budget
   SCIPfreeBlockMemoryArray(scip, &probinfo->localbudget, nnodes);

   return SCIP_OKAY;
}

/** frees data of a node classification problem */
static
SCIP_RETCODE freeNodeClassificationProblem(
   SCIP*                 scip,               //!< SCIP pointer
   GNNPROB_TYPEINFO*     probtypeinfo        //!< information about problem
   )
{
   GNNPROB_NODECLASSIFY* probinfo;
   int nnodes;
   int nfeatures;

   assert(scip != NULL);
   assert(probtypeinfo != NULL);

   probinfo = &probtypeinfo->nodeclassifyinfo;
   nnodes = probinfo->nnodes;
   nfeatures = probinfo->nfeatures;

   // free adjacency matrix
   for( int i = 0; i < nnodes; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &probinfo->adjacencymatrix[i], nnodes);
   }
   SCIPfreeBlockMemoryArray(scip, &probinfo->adjacencymatrix, nnodes);

   // free bounds on feature assignments
   for( int i = 0; i < nnodes; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &probinfo->featurelb[i], nfeatures);
   }
   SCIPfreeBlockMemoryArray(scip, &probinfo->featurelb, nnodes);
   for( int i = 0; i < nnodes; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &probinfo->featureub[i], nfeatures);
   }
   SCIPfreeBlockMemoryArray(scip, &probinfo->featureub, nnodes);

   return SCIP_OKAY;
}

/** frees GNN problem data */
extern
SCIP_RETCODE freeGNNProbData(
   SCIP*                 scip,               //!< SCIP pointer
   GNNPROB_DATA*         gnnprobdata         //!< pointer to GNN problem data
   )
{
   assert(scip != NULL);
   assert(gnnprobdata != NULL);

   switch( gnnprobdata->probtype )
   {
   case GNNPROB_TYPE_ROBUSTCLASSIFY:
      SCIP_CALL( freeRobustClassificationProblem(scip, gnnprobdata->probtypeinfo) );
      break;
   case GNNPROB_TYPE_NODECLASSIFY:
      SCIP_CALL( freeNodeClassificationProblem(scip, gnnprobdata->probtypeinfo) );
      break;
   default:
      assert(FALSE);
   }

   SCIPfreeBlockMemory(scip, &gnnprobdata->probtypeinfo);
   SCIPfreeBlockMemory(scip, &gnnprobdata);

   return SCIP_OKAY;
}
