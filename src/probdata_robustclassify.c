/**@file   probdata_robustclassify.c
 * @brief  Problem data for robust classification problems on GNNs
 * @author Christopher Hojny
 *
 * This file handles the main problem data used in robust classification problems on GNNs.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scip/scip.h"
#include "scip/cons_linear.h"

#include "probdata_robustclassify.h"
#include "gnn.h"
#include "struct_gnn.h"
#include "struct_problem.h"
#include "type_gnn.h"
#include "type_problem.h"

/** @brief Problem data which is accessible in all places
 *
 * This problem data is used to store the input of the problem, all variables which are created, and all
 * constraints.
 */
struct SCIP_ProbData
{
   /* global information about GNN and robust classification problem */
   GNN_DATA*             gnndata;            /**< data about GNN */
   int                   nnodes;             /**< number of nodes of underlying graph */
   SCIP_Bool**           adjacencymatrix;    /**< adjacency matrix of underlying graph */
   int                   ngraphfeatures;     /**< number of node features of graph */
   int                   globalbudget;       /**< global attack budget on graph */
   int*                  localbudget;        /**< array assigning each node an attack budget */
   int                   graphclassification; /**< index of feature the graph is classified as */
   int                   targetclassification; /**< index of feature a modified graph should be classified as */
   SCIP_Bool             uselprelax;         /**< whether just the LP relaxation shall be solved */

   /* variables and constraints per layer of GNN */
   SCIP_VAR***           gnnoutputvars;      /**< variables modeling output values at nodes of GNN per layer*/
   SCIP_VAR***           auxvars;            /**< variables to linearize products of variables per layer */
   SCIP_VAR***           isactivevars;       /**< variables modeling whether ReLU is active per layer */
   SCIP_CONS***          linkauxconss;       /**< conss modeling linking of auxvars with remaining vars per layer */
   SCIP_CONS***          layerlinkingconss;  /**< conss linking gnnoutputvars of consecutive layers per layer */

   /* layer-independent variables and constraints of problem */
   SCIP_VAR**            adjacencyvars;      /**< variables modeling adjacency of nodes in modified graph */
   SCIP_CONS*            globalattackcons;   /**< constraint bounding number of global attacks */
   SCIP_CONS**           localattackconss;   /**< constraints bounding number of local attacks */

   /* information for freeing data (gnndata is no longer available by then) */
   int                   nlayers;            /**< number of layers of GNN */
   GNN_LAYERTYPE*        layertypes;         /**< array of layer types */
   GNN_ACTIVATIONTYPE*   activation;         /**< activation function of layer */
   int*                  noutputfeatures;    /**< number of output features per layer */
};

/** returns index of edge in list of edges */
static
int getEdgeIdx(
   int                   u,                  /**< first node in edge */
   int                   v,                  /**< second node in edge */
   int                   nnodes              /**< number of nodes in complete graph */
   )
{
   assert(u != v);

   if( u < v )
      return v - u + nnodes * u - (u * u + u)/2 - 1;

   return u - v + nnodes * v - (v * v + v)/2 - 1;
}

/** creates adjacency variable */
static
SCIP_RETCODE createAdjacencyVariable(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            var,                /**< pointer to store variable */
   int                   u,                  /**< index of first node */
   int                   v,                  /**< index of second node */
   SCIP_Bool             iscontinuous        /**< whether variables are forced to be continuous */
   )
{
   char name[SCIP_MAXSTRLEN];

   assert(scip != NULL);

   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "adj_%d_%d", u, v);

   SCIP_CALL( SCIPcreateVar(scip, var, name, 0.0, 1.0, 0.0,
         iscontinuous ? SCIP_VARTYPE_CONTINUOUS : SCIP_VARTYPE_BINARY,
         TRUE, FALSE, NULL, NULL, NULL, NULL, NULL) );
   SCIP_CALL( SCIPaddVar(scip, *var) );

   return SCIP_OKAY;
}

/** creates global attack budget constraint */
static
SCIP_RETCODE createGlobalAttackBudgetCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to store constraint */
   int                   nnodes,             /**< number of nodes of underlying graph */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   SCIP_VAR**            adjacencyvars,      /**< adjacency variables */
   SCIP_Real*            coefs,              /**< allocated array to store coefficients */
   int                   globalbudget        /**< global attack budget */
   )
{
   char name[SCIP_MAXSTRLEN];
   SCIP_Real rhs;
   int nadjvars;
   int cnt;
   int i;
   int j;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(nnodes > 0);
   assert(adjacencymatrix != NULL);
   assert(adjacencyvars != NULL);
   assert(globalbudget > 0);

   nadjvars = nnodes * (nnodes - 1) / 2;
   rhs = (SCIP_Real) globalbudget;

   for( i = 0, cnt = 0; i < nnodes; ++i )
   {
      for( j = i + 1; j < nnodes; ++j )
      {
         if( adjacencymatrix[i][j] )
         {
            coefs[cnt++] = -1.0;
            rhs -= 1.0;
         }
         else
            coefs[cnt++] = 1.0;
      }
   }
   assert(cnt == nadjvars);

   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "globalbudget");
   SCIP_CALL( SCIPcreateConsLinear(scip, cons, name, nadjvars, adjacencyvars, coefs, -SCIPinfinity(scip), rhs,
         TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
   SCIP_CALL( SCIPaddCons(scip, *cons) );

   return SCIP_OKAY;
}

/** creates local attack budget constraint */
static
SCIP_RETCODE createLocalAttackBudgetCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to store constraint */
   int                   nnodes,             /**< number of nodes of underlying graph */
   int                   nodeidx,            /**< index of node for which constraint is created */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   SCIP_VAR**            adjacencyvars,      /**< adjacency variables */
   SCIP_VAR**            vars,               /**< allocated array to store variables */
   SCIP_Real*            coefs,              /**< allocated array to store coefficients */
   int                   localbudget         /**< local attack budget */
   )
{
   char name[SCIP_MAXSTRLEN];
   SCIP_Real rhs;
   int cnt;
   int i;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(nnodes > 0);
   assert(0 <= nodeidx && nodeidx < nnodes);
   assert(adjacencymatrix != NULL);
   assert(adjacencyvars != NULL);
   assert(localbudget >= 0);

   rhs = (SCIP_Real) localbudget;

   for( i = 0, cnt = 0; i < nnodes; ++i )
   {
      /* skip the edge to the node itself */
      if( i == nodeidx )
         continue;

      vars[cnt] = adjacencyvars[getEdgeIdx(i, nodeidx, nnodes)];

      if( adjacencymatrix[i][nodeidx] )
      {
         coefs[cnt++] = -1.0;
         rhs -= 1.0;
      }
      else
         coefs[cnt++] = 1.0;
   }
   assert(cnt == nnodes - 1);

   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "localbudget%d", nodeidx);
   SCIP_CALL( SCIPcreateConsLinear(scip, cons, name, nnodes - 1, vars, coefs, -SCIPinfinity(scip), rhs,
         TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
   SCIP_CALL( SCIPaddCons(scip, *cons) );

   return SCIP_OKAY;
}

/** creates problem data and assigns basic information */
static
SCIP_RETCODE probdataCreateBasic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROBDATA**       targetprobdata,     /**< pointer to problem data to be created */
   GNN_DATA*             gnndata,            /**< data about GNN */
   int                   nnodes,             /**< number of nodes of underlying graph */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   int                   globalbudget,       /**< global attack budget on graph */
   int*                  localbudget,        /**< array assigning each node an attack budget */
   int                   graphclassification, /**< index of feature the graph is classified as */
   int                   targetclassification,/**< index of feature a modified graph should be classified as */
   SCIP_Bool             uselprelax          /**< whether just the LP relaxation shall be solved */
   )
{
   int nlayers;
   int l;
   int i;

   assert(scip != NULL);
   assert(targetprobdata != NULL);
   assert(gnndata != NULL);
   assert(nnodes > 0);
   assert(adjacencymatrix != NULL);
   assert(globalbudget > 0);
   assert(localbudget != NULL);
   assert(graphclassification >= 0);
   assert(targetclassification >= 0);

   /* allocate memory */
   SCIP_CALL( SCIPallocBlockMemory(scip, targetprobdata) );

   (*targetprobdata)->gnndata = gnndata;
   (*targetprobdata)->nnodes = nnodes;
   (*targetprobdata)->uselprelax = uselprelax;
   SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(*targetprobdata)->adjacencymatrix, adjacencymatrix, nnodes) );
   for( i = 0; i < nnodes; ++i )
   {
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(*targetprobdata)->adjacencymatrix[i],
            adjacencymatrix[i], nnodes) );
   }
   (*targetprobdata)->globalbudget = globalbudget;
   SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(*targetprobdata)->localbudget, localbudget, nnodes) );
   (*targetprobdata)->graphclassification = graphclassification;
   (*targetprobdata)->targetclassification = targetclassification;

   nlayers = SCIPgetGNNNLayers(gnndata);
   (*targetprobdata)->nlayers = nlayers;
   SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &(*targetprobdata)->layertypes, gnndata->layertypes, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(*targetprobdata)->activation, nlayers) );
   for( l = 0; l < nlayers; ++l )
   {
      switch( (*targetprobdata)->layertypes[l] )
      {
      case GNN_LAYERTYPE_INPUT:
      case GNN_LAYERTYPE_POOL:
         (*targetprobdata)->activation[l] = GNN_ACTIVATIONTYPE_NONE;
         break;
      case GNN_LAYERTYPE_SAGE:
         (*targetprobdata)->activation[l] = SCIPgetSageLayerActivationType(SCIPgetGNNLayerinfoSage(gnndata, l));
         break;
      case GNN_LAYERTYPE_DENSE:
         (*targetprobdata)->activation[l] = SCIPgetDenseLayerActivationType(SCIPgetGNNLayerinfoDense(gnndata, l));
         break;
      default:
         assert(FALSE);
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(*targetprobdata)->noutputfeatures, nlayers) );
   for( l = 0; l < nlayers; ++l )
   {
      switch( (*targetprobdata)->layertypes[l] )
      {
      case GNN_LAYERTYPE_INPUT:
         (*targetprobdata)->noutputfeatures[l] = SCIPgetNInputFeaturesInputLayer(SCIPgetGNNLayerinfoInput(gnndata, l));
         break;
      case GNN_LAYERTYPE_SAGE:
         (*targetprobdata)->noutputfeatures[l] = SCIPgetNOutputFeaturesSageLayer(SCIPgetGNNLayerinfoSage(gnndata, l));
         break;
      case GNN_LAYERTYPE_POOL:
         (*targetprobdata)->noutputfeatures[l] = SCIPgetNOutputFeaturesPoolLayer(SCIPgetGNNLayerinfoPool(gnndata, l));
         break;
      case GNN_LAYERTYPE_DENSE:
         (*targetprobdata)->noutputfeatures[l] = SCIPgetNOutputFeaturesDenseLayer(SCIPgetGNNLayerinfoDense(gnndata, l));
         break;
      default:
         assert(FALSE);
      }
   }

   return SCIP_OKAY;
}

/** creates global variables and constraints */
static
SCIP_RETCODE probdataCreateGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROBDATA*        sourceprobdata,     /**< problem data which shall be copied (or NULL) */
   SCIP_PROBDATA*        targetprobdata,     /**< problem data to be created (basic information already assigned) */
   SCIP_Bool             uselprelax          /**< whether just the LP relaxation shall be solved */
   )
{
   SCIP_Bool** adjmat;
   int nadjvars;
   int nnodes;
   int i;
   int j;

   assert(scip != NULL);
   assert(targetprobdata != NULL);

   nnodes = targetprobdata->nnodes;
   adjmat = targetprobdata->adjacencymatrix;
   nadjvars = nnodes * (nnodes - 1)/2;

   /* either create variables and constraints from scratch or copy them */
   if( sourceprobdata != NULL )
   {
      /* copy adjacency variables */
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->adjacencyvars,
            sourceprobdata->adjacencyvars, nadjvars) );

      /* copy global attack constraint */
      targetprobdata->globalattackcons = sourceprobdata->globalattackcons;

      /* copy local attack constraints */
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->localattackconss,
            sourceprobdata->localattackconss, nnodes) );
   }
   else
   {
      SCIP_VAR** vars;
      SCIP_Real* coefs;
      int cnt;

      /* create temporary memory to avoid re-allocation in functions to be called */
      SCIP_CALL( SCIPallocBufferArray(scip, &vars, nnodes - 1) );
      SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nadjvars) );

      /* create adjacency variables */
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &targetprobdata->adjacencyvars, nadjvars) );
      for( i = 0, cnt = 0; i < nnodes; ++i )
      {
         for( j = i + 1; j < nnodes; ++j )
         {
            SCIP_CALL( createAdjacencyVariable(scip, &targetprobdata->adjacencyvars[cnt++], i, j, uselprelax) );
         }
      }

      /* create global attack constraint */
      SCIP_CALL( createGlobalAttackBudgetCons(scip, &targetprobdata->globalattackcons, nnodes,
            adjmat, targetprobdata->adjacencyvars, coefs, targetprobdata->globalbudget) );

      /* create local attack constraints */
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &targetprobdata->localattackconss, nnodes) );
      for( i = 0; i < nnodes; ++i )
      {
         SCIP_CALL( createLocalAttackBudgetCons(scip, &targetprobdata->localattackconss[i], nnodes, i,
               adjmat, targetprobdata->adjacencyvars, vars, coefs, targetprobdata->localbudget[i]) );
      }

      SCIPfreeBufferArray(scip, &coefs);
      SCIPfreeBufferArray(scip, &vars);
   }

   return SCIP_OKAY;
}

/** gets information about layer */
static
SCIP_RETCODE getDataLayerProbdata(
   int                   lidx,               /**< index of layer for which information is extracted */
   int                   nnodes,             /**< number of nodes in underlying graph */
   int                   nlayers,            /**< number of layers in GNN */
   GNN_LAYERTYPE*        layertypes,         /**< array of types of different layers */
   int*                  noutputfeatures,    /**< array of number of output features per layer */
   int*                  ngnnoutputvars,     /**< pointer to store number of gnnoutput variables */
   int*                  nauxvars            /**< pointer to store number of auxiliary variables */
   )
{
   assert(0 <= lidx && lidx < nlayers);
   assert(nnodes > 0);
   assert(layertypes != NULL);
   assert(noutputfeatures != NULL);
   assert(ngnnoutputvars != NULL);
   assert(nauxvars != NULL);

   switch( layertypes[lidx] )
   {
   case GNN_LAYERTYPE_INPUT:
      *ngnnoutputvars = nnodes * noutputfeatures[lidx];
      *nauxvars = nnodes * nnodes * noutputfeatures[lidx];
      break;
   case GNN_LAYERTYPE_SAGE:
      *ngnnoutputvars = nnodes * noutputfeatures[lidx];
      if( lidx < nlayers - 1 && layertypes[lidx + 1] == GNN_LAYERTYPE_SAGE )
         *nauxvars = nnodes * nnodes * noutputfeatures[lidx];
      else
         *nauxvars = 0;
      break;
   case GNN_LAYERTYPE_POOL:
      *ngnnoutputvars = noutputfeatures[lidx];
      *nauxvars = 0;
      break;
   case GNN_LAYERTYPE_DENSE:
      *ngnnoutputvars = noutputfeatures[lidx];
      *nauxvars = 0;
      break;
   default:
      assert(FALSE);
   }

   return SCIP_OKAY;
}

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

/** creates gnnoutput variables for a layer */
static
SCIP_RETCODE createGNNOutputVarsLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           gnnoutputvars,      /**< pointer to array for storing variables */
   GNN_LAYERTYPE         layertype,          /**< type of layer */
   int                   nnodes,             /**< number of nodes in underlying graph */
   int                   nfeatures,          /**< number of features of layer */
   int                   layeridx,           /**< index of layer */
   SCIP_Real**           featurelb,          /**< (nnodes x nfeatures)-matrix of lower bounds on feature assignments */
   SCIP_Real**           featureub,          /**< (nnodes x nfeatures)-matrix of upper bounds on feature assignments */
   SCIP_Real*            lbs,                /**< array containing lower bounds on gnnoutputvars */
   SCIP_Real*            ubs                /**< array containing lower bounds on gnnoutputvars */
   )
{
   char name[SCIP_MAXSTRLEN];
   SCIP_Real lb;
   SCIP_Real ub;
   int nodebound;
   int nvars;
   int cnt;
   int v;
   int f;

   assert(scip != NULL);
   assert(gnnoutputvars != NULL);
   assert(nnodes > 0);
   assert(nfeatures > 0);
   assert(layeridx >= 0);
   assert(featurelb != NULL);
   assert(featureub != NULL);
   assert(lbs != NULL);
   assert(ubs != NULL);

   /* input and sage layers have a node for every (node,feature) tuple */
   nvars = nfeatures;
   nodebound = 1;
   if( layertype == GNN_LAYERTYPE_INPUT || layertype == GNN_LAYERTYPE_SAGE )
   {
      nvars *= nnodes;
      nodebound = nnodes;
   }

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, gnnoutputvars, nvars) );
   for( v = 0, cnt = 0; v < nodebound; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "gnnoutputvar%d#%d#%d", layeridx, v, f);

         lb = lbs[cnt];
         ub = ubs[cnt];
         if( layertype == GNN_LAYERTYPE_INPUT )
         {
            lb = MAX(lb, featurelb[v][f]);
            ub = MIN(ub, featureub[v][f]);
         }

         SCIP_CALL( SCIPcreateVar(scip, &(*gnnoutputvars)[cnt], name, lb, ub, 0.0,
               SCIP_VARTYPE_CONTINUOUS, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );
         SCIP_CALL( SCIPaddVar(scip, (*gnnoutputvars)[cnt++]) );
      }
   }

   return SCIP_OKAY;
}

/** creates auxiliary variables for a layer */
static
SCIP_RETCODE createAuxVarsLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           auxvars,            /**< pointer to array for storing variables */
   GNN_LAYERTYPE         layertype,          /**< type of layer */
   int                   nnodes,             /**< number of nodes in underlying graph */
   int                   nfeatures,          /**< number of features of layer */
   int                   layeridx,           /**< index of layer */
   SCIP_Real*            lbs,                /**< array containing lower bounds on auxvars */
   SCIP_Real*            ubs                 /**< array containing lower bounds on auxvars */
   )
{
   char name[SCIP_MAXSTRLEN];
   int nvars;
   int cnt;
   int v;
   int w;
   int f;

   assert(scip != NULL);
   assert(auxvars != NULL || layertype == GNN_LAYERTYPE_POOL || layertype == GNN_LAYERTYPE_DENSE);
   assert(nnodes > 0);
   assert(nfeatures > 0);
   assert(layeridx >= 0);
   assert(lbs != NULL || layertype == GNN_LAYERTYPE_POOL || layertype == GNN_LAYERTYPE_DENSE);
   assert(ubs != NULL || layertype == GNN_LAYERTYPE_POOL || layertype == GNN_LAYERTYPE_DENSE);

   /* there are no auxiliary variables in pooling and dense layers */
   if( layertype == GNN_LAYERTYPE_POOL || layertype == GNN_LAYERTYPE_DENSE )
   {
      *auxvars = NULL;
      return SCIP_OKAY;
   }

   nvars = nfeatures * nnodes * nnodes;

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, auxvars, nvars) );
   for( v = 0, cnt = 0; v < nnodes; ++v )
   {
      for( w = 0; w < nnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "auxvar%d#%d#%d#%d", layeridx, v, w, f);

            SCIP_CALL( SCIPcreateVar(scip, &(*auxvars)[cnt], name, lbs[cnt], ubs[cnt], 0.0,
                  SCIP_VARTYPE_CONTINUOUS, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );
            SCIP_CALL( SCIPaddVar(scip, (*auxvars)[cnt++]) );
         }
      }
   }

   return SCIP_OKAY;
}

/** creates isactive variables for a layer */
static
SCIP_RETCODE createIsactiveVarsLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           isactivevars,       /**< pointer to array for storing variables */
   GNN_LAYERTYPE         layertype,          /**< type of layer */
   GNN_ACTIVATIONTYPE    activation,         /**< type of activation at layer */
   int                   nnodes,             /**< number of nodes in underlying graph */
   int                   nfeatures,          /**< number of features of layer */
   int                   layeridx,           /**< index of layer */
   SCIP_Real*            lbs,                /**< array containing lower bounds on gnnoutputvars */
   SCIP_Real*            ubs,                /**< array containing lower bounds on gnnoutputvars */
   SCIP_Bool             iscontinuous        /**< whether variables are forced to be continuous */
   )
{
   char name[SCIP_MAXSTRLEN];
   SCIP_Real ub;
   SCIP_Real lb;
   int nodebound;
   int nvars;
   int cnt;
   int v;
   int f;

   assert(scip != NULL);
   assert(isactivevars != NULL || layertype == GNN_LAYERTYPE_INPUT || layertype == GNN_LAYERTYPE_SAGE);
   assert(nnodes > 0);
   assert(nfeatures > 0);
   assert(layeridx >= 0);
   assert(lbs != NULL);
   assert(ubs != NULL);

   if( activation == GNN_ACTIVATIONTYPE_NONE )
   {
      *isactivevars = NULL;
      return SCIP_OKAY;
   }

   /* input and sage layers have a node for every (node,feature) tuple */
   nvars = nfeatures;
   nodebound = 1;
   if( layertype == GNN_LAYERTYPE_INPUT || layertype == GNN_LAYERTYPE_SAGE )
   {
      nvars *= nnodes;
      nodebound = nnodes;
   }

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, isactivevars, nvars) );
   for( v = 0, cnt = 0; v < nodebound; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "isactivevar%d#%d#%d", layeridx, v, f);

         if( SCIPisPositive(scip, lbs[cnt]) )
            lb = 1.0;
         else
            lb = 0.0;
         if( SCIPisLE(scip, ubs[cnt], 0.0) )
            ub = 0.0;
         else
            ub = 1.0;

         SCIP_CALL( SCIPcreateVar(scip, &(*isactivevars)[cnt], name, lb, ub, 0.0,
               iscontinuous ? SCIP_VARTYPE_CONTINUOUS : SCIP_VARTYPE_BINARY,
               TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );
         SCIP_CALL( SCIPaddVar(scip, (*isactivevars)[cnt++]) );
      }
   }

   return SCIP_OKAY;
}

/** creates constraints linking auxiliary variables for a layer */
static
SCIP_RETCODE createLinkAuxConssLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS***          linkingconss,       /**< pointer to array for storing constraints */
   GNN_LAYERTYPE         layertype,          /**< type of layer */
   GNN_ACTIVATIONTYPE    activation,         /**< type of activation at layer */
   int                   nnodes,             /**< number of nodes in underlying graph */
   int                   nfeatures,          /**< number of features of layer */
   int                   layeridx,           /**< index of layer */
   SCIP_VAR**            adjacencyvars,      /**< adjacency variables of problem */
   SCIP_VAR**            gnnoutputvars,      /**< GNN output variables of layer */
   SCIP_VAR**            auxvars,            /**< auxiliary variables of layer */
   SCIP_Real*            lbgnnoutputvars,    /**< array containing lower bounds on gnnoutputvars */
   SCIP_Real*            lbauxvars,          /**< array containing lower bounds on auxvars */
   SCIP_Real*            ubgnnoutputvars,    /**< array containing upper bounds on gnnoutputvars */
   SCIP_Real*            ubauxvars           /**< array containing upper bounds on auxvars */
   )
{
   char name[SCIP_MAXSTRLEN];
   SCIP_VAR* vars[3];
   SCIP_Real coefs[3];
   int auxvaridx;
   int gnnoutidx;
   int edgeidx;
   int nconss;
   int cnt;
   int v;
   int w;
   int f;

   assert(scip != NULL);
   assert(linkingconss != NULL || layertype == GNN_LAYERTYPE_POOL || layertype == GNN_LAYERTYPE_DENSE);
   assert(nnodes > 0);
   assert(nfeatures > 0);
   assert(layeridx >= 0);
   assert(adjacencyvars != NULL);
   assert(gnnoutputvars != NULL);
   assert(auxvars != NULL || layertype == GNN_LAYERTYPE_POOL || layertype == GNN_LAYERTYPE_DENSE);
   assert(lbgnnoutputvars != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(lbauxvars != NULL || layertype == GNN_LAYERTYPE_POOL || layertype == GNN_LAYERTYPE_DENSE);
   assert(ubauxvars != NULL || layertype == GNN_LAYERTYPE_POOL || layertype == GNN_LAYERTYPE_DENSE);

   /* there are no auxiliary variables in pooling and dense layers */
   if( layertype == GNN_LAYERTYPE_POOL || layertype == GNN_LAYERTYPE_DENSE )
      return SCIP_OKAY;

   nconss = 4 * nfeatures * nnodes * nnodes;

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, linkingconss, nconss) );
   for( v = 0, cnt = 0; v < nnodes; ++v )
   {
      for( w = 0; w < nnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            auxvaridx = SCIPgetAuxvarIdxLayer(nnodes, nfeatures, v, w, f);
            gnnoutidx = SCIPgetGNNNodevarIdxLayer(nnodes, nfeatures, w, f);

            vars[0] = auxvars[auxvaridx];
            coefs[0] = 1.0;

            /* handle self-edges differently as there is no variable for them */
            if( v == w )
            {
               vars[1] = gnnoutputvars[gnnoutidx];

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "linkingA%d#%d#%d#%d", layeridx, v, w, f);
               coefs[1] = -1.0;

               SCIP_CALL( SCIPcreateConsLinear(scip, &(*linkingconss)[cnt], name, 2, vars, coefs,
                     -SCIPinfinity(scip), 0.0,
                     TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
               SCIP_CALL( SCIPaddCons(scip, (*linkingconss)[cnt++]) );

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "linkingB%d#%d#%d#%d", layeridx, v, w, f);

               SCIP_CALL( SCIPcreateConsLinear(scip, &(*linkingconss)[cnt], name, 2, vars, coefs,
                     0.0, SCIPinfinity(scip),
                     TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
               SCIP_CALL( SCIPaddCons(scip, (*linkingconss)[cnt++]) );

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "linkingC%d#%d#%d#%d", layeridx, v, w, f);

               SCIP_CALL( SCIPcreateConsLinear(scip, &(*linkingconss)[cnt], name, 1, vars, coefs,
                     -SCIPinfinity(scip), ubauxvars[auxvaridx],
                     TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
               SCIP_CALL( SCIPaddCons(scip, (*linkingconss)[cnt++]) );

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "linkingD%d#%d#%d#%d", layeridx, v, w, f);

               SCIP_CALL( SCIPcreateConsLinear(scip, &(*linkingconss)[cnt], name, 1, vars, coefs,
                     lbauxvars[auxvaridx], SCIPinfinity(scip),
                     TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
               SCIP_CALL( SCIPaddCons(scip, (*linkingconss)[cnt++]) );
            }
            else
            {
               edgeidx = getEdgeIdx(v, w, nnodes);

               vars[1] = adjacencyvars[edgeidx];
               vars[2] = gnnoutputvars[gnnoutidx];

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "linkingA%d#%d#%d#%d", layeridx, v, w, f);
               coefs[1] = -lbgnnoutputvars[gnnoutidx];
               coefs[2] = -1.0;

               SCIP_CALL( SCIPcreateConsLinear(scip, &(*linkingconss)[cnt], name, 3, vars, coefs,
                     -SCIPinfinity(scip), -lbgnnoutputvars[gnnoutidx],
                     TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
               SCIP_CALL( SCIPaddCons(scip, (*linkingconss)[cnt++]) );

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "linkingB%d#%d#%d#%d", layeridx, v, w, f);
               coefs[1] = -ubgnnoutputvars[gnnoutidx];

               SCIP_CALL( SCIPcreateConsLinear(scip, &(*linkingconss)[cnt], name, 3, vars, coefs,
                     -ubgnnoutputvars[gnnoutidx], SCIPinfinity(scip),
                     TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
               SCIP_CALL( SCIPaddCons(scip, (*linkingconss)[cnt++]) );

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "linkingC%d#%d#%d#%d", layeridx, v, w, f);
               coefs[1] = -ubauxvars[auxvaridx];

               SCIP_CALL( SCIPcreateConsLinear(scip, &(*linkingconss)[cnt], name, 2, vars, coefs,
                     -SCIPinfinity(scip), 0.0,
                     TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
               SCIP_CALL( SCIPaddCons(scip, (*linkingconss)[cnt++]) );

               (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "linkingD%d#%d#%d#%d", layeridx, v, w, f);
               coefs[1] = -lbauxvars[auxvaridx];

               SCIP_CALL( SCIPcreateConsLinear(scip, &(*linkingconss)[cnt], name, 2, vars, coefs,
                     0.0, SCIPinfinity(scip),
                     TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
               SCIP_CALL( SCIPaddCons(scip, (*linkingconss)[cnt++]) );
            }
         }
      }
   }

   return SCIP_OKAY;
}

/** creates constraints modeling ReLU activation for a layer */
static
SCIP_RETCODE createReluConssLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS***          reluconss,          /**< pointer to store ReLU conss */
   int                   nnodes,             /**< number of nodes of underlying graph */
   GNN_LAYERTYPE         type,               /**< type of layer for which conss are created */
   GNN_ACTIVATIONTYPE    activation,         /**< activation function at layer for which conss are created */
   GNN_LAYERTYPE         typeprev,           /**< type of previous layer */
   GNN_ACTIVATIONTYPE    activationprev,     /**< activation function at previous layer */
   int                   nfeatures,          /**< number of features at current layer */
   int                   nfeaturesprev,      /**< number of features at previous layer */
   int                   lidx,               /**< index of layer for which conss are created */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency */
   SCIP_VAR**            gnnoutputvars,      /**< output variabes for each node of current layer */
   SCIP_VAR**            isactivevars,       /**< variabes modeling whether ReLU is active for current layer */
   SCIP_VAR**            auxvarsprev,        /**< auxiliary variables of previous layer (or NULL) */
   SCIP_VAR**            gnnoutputvarsprev,  /**< output variables of previous layer */
   SCIP_Real*            lbnodecontent,      /**< array of lower bounds of input for ReLU function */
   SCIP_Real*            ubnodecontent,      /**< array of upper bounds of input for ReLU function */
   GNN_DATA*             gnndata             /**< data of underlying GNN */
   )
{
   char name[SCIP_MAXSTRLEN];
   GNN_LAYERINFO_SAGE* sageinfo;
   GNN_LAYERINFO_DENSE* denseinfo;
   SCIP_VAR** vars;
   SCIP_Real* coefs;
   SCIP_Real bias;
   int cntconss;
   int storagelen;
   int nodebound;
   int cnt;
   int idx;
   int v;
   int w;
   int f;
   int f2;

   assert(scip != NULL);
   assert(reluconss != NULL);
   assert(nnodes > 0);
   assert(nfeatures > 0);
   assert(nfeaturesprev > 0);
   assert(lidx > 0);
   assert(gnnoutputvars != NULL);
   assert(isactivevars != NULL);
   assert(activation == GNN_ACTIVATIONTYPE_RELU);
   assert(auxvarsprev != NULL || typeprev == GNN_LAYERTYPE_POOL || typeprev == GNN_LAYERTYPE_DENSE
      || lidx == 1);
   assert(gnnoutputvarsprev != NULL);
   assert(lbnodecontent != NULL || activation != GNN_ACTIVATIONTYPE_RELU);
   assert(ubnodecontent != NULL || activation != GNN_ACTIVATIONTYPE_RELU);

   /* possibly ignore nodes if we are in a dense layer and get information about layer */
   if( type == GNN_LAYERTYPE_DENSE )
   {
      nodebound = 1;
      denseinfo = SCIPgetGNNLayerinfoDense(gnndata, lidx);
   }
   else
   {
      nodebound = nnodes;
      sageinfo = SCIPgetGNNLayerinfoSage(gnndata, lidx);
   }

   /* allocate temporary memory for variables and coefficients */
   storagelen = nodebound * nfeaturesprev + 2;
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, storagelen) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, storagelen) );

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, reluconss, 4 * nodebound * nfeatures) );

   /* iterate through nodes of GNN layer */
   cntconss = 0;
   for( v = 0; v < nodebound; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         /* set variables and coefficients as the gnnoutputvar of the current layer's node
          * and the input expression of the ReLU activation function
          */
         idx = SCIPgetGNNNodevarIdxLayer(nodebound, nfeatures, v, f);
         vars[0] = gnnoutputvars[idx];
         coefs[0] = 1.0;

         if( type == GNN_LAYERTYPE_DENSE )
            bias = SCIPgetDenseLayerFeatureBias(denseinfo, f);
         else
            bias = SCIPgetSageLayerFeatureBias(sageinfo, f);

         cnt = 1;
         if( type == GNN_LAYERTYPE_SAGE )
         {
            assert(typeprev == GNN_LAYERTYPE_INPUT || typeprev == GNN_LAYERTYPE_SAGE);
            for( w = 0; w < nnodes; ++w )
            {
               for( f2 = 0; f2 < nfeaturesprev; ++f2 )
               {
                  /* in the input layer, there are no auxiliary variables due to fixed features */
                  if( lidx == 1 )
                  {
                     idx = SCIPgetGNNNodevarIdxLayer(nnodes, nfeaturesprev, w, f2);
                     assert(SCIPisEQ(scip, SCIPvarGetLbLocal(gnnoutputvarsprev[idx]),
                           SCIPvarGetUbLocal(gnnoutputvarsprev[idx])));
                     if( v != w )
                     {
                        SCIP_Real c;
                        vars[cnt] = adjacencyvars[getEdgeIdx(v, w, nnodes)];
                        c = -SCIPgetSageLayerFeatureEdgeweight(sageinfo, f2, f);
                        c = c * SCIPvarGetLbLocal(gnnoutputvarsprev[idx]);
                        coefs[cnt++] = c;
                     }
                     else
                     {
                        /* the auxiliary variable coincides with the (fixed) output variable (in our applications, input is fixed) */
                        bias += SCIPgetSageLayerFeatureNodeweight(sageinfo, f2, f) * SCIPvarGetLbLocal(gnnoutputvarsprev[idx]);
                     }
                  }
                  else
                  {
                     idx = SCIPgetAuxvarIdxLayer(nnodes, nfeaturesprev, v, w, f2);
                     vars[cnt] = auxvarsprev[idx];

                     if( v == w )
                        coefs[cnt++] = -SCIPgetSageLayerFeatureNodeweight(sageinfo, f2, f);
                     else
                        coefs[cnt++] = -SCIPgetSageLayerFeatureEdgeweight(sageinfo, f2, f);
                  }
               }
            }
         }
         else
         {
            assert(typeprev == GNN_LAYERTYPE_POOL || typeprev == GNN_LAYERTYPE_DENSE);
            for( f2 = 0; f2 < nfeaturesprev; ++f2 )
            {
               idx = SCIPgetGNNNodevarIdxLayer(1, nfeaturesprev, v, f2);
               vars[cnt] = gnnoutputvarsprev[idx];

               coefs[cnt++] = -SCIPgetDenseLayerFeatureWeight(denseinfo, f2, f);
            }
         }

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "reluA%d#%d#%d", lidx, v, f);
         SCIP_CALL( SCIPcreateConsLinear(scip, &(*reluconss)[cntconss], name, cnt, vars, coefs,
               bias, SCIPinfinity(scip),
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
         SCIP_CALL( SCIPaddCons(scip, (*reluconss)[cntconss++]) );

         /* second type of ReLU constraint */
         idx = SCIPgetGNNNodevarIdxLayer(nodebound, nfeatures, v, f);
         vars[cnt] = isactivevars[idx];
         coefs[cnt++] = -lbnodecontent[idx];

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "reluB%d#%d#%d", lidx, v, f);
         SCIP_CALL( SCIPcreateConsLinear(scip, &(*reluconss)[cntconss], name, cnt, vars, coefs,
               -SCIPinfinity(scip), bias - lbnodecontent[idx],
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
         SCIP_CALL( SCIPaddCons(scip, (*reluconss)[cntconss++]) );

         /* third type of ReLU constraint */
         vars[1] = isactivevars[idx];
         coefs[1] = -ubnodecontent[idx];

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "reluC%d#%d#%d", lidx, v, f);
         SCIP_CALL( SCIPcreateConsLinear(scip, &(*reluconss)[cntconss], name, 2, vars, coefs,
               -SCIPinfinity(scip), 0.0,
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
         SCIP_CALL( SCIPaddCons(scip, (*reluconss)[cntconss++]) );

         /* fourth type of ReLU constraint */
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "reluD%d#%d#%d", lidx, v, f);
         SCIP_CALL( SCIPcreateConsLinear(scip, &(*reluconss)[cntconss], name, 1, vars, coefs,
               0.0, SCIPinfinity(scip),
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
         SCIP_CALL( SCIPaddCons(scip, (*reluconss)[cntconss++]) );
      }
   }

   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}

/** creates linking constraints of consecutive layers without activation function */
static
SCIP_RETCODE createPlainLinkingLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS***          layerlinkingconss,  /**< pointer to store layer linking conss */
   int                   nnodes,             /**< number of nodes of underlying graph */
   GNN_LAYERTYPE         targettype,         /**< type of target layer */
   GNN_ACTIVATIONTYPE    targetactivation,   /**< activation function at target layer */
   GNN_LAYERTYPE         sourcetype,         /**< type of source layer */
   GNN_ACTIVATIONTYPE    sourceactivation,   /**< activation function at source layer */
   int                   ntargetfeatures,    /**< number of features in target layer */
   int                   nsourcefeatures,    /**< number of features in source layer */
   int                   targetlayeridx,     /**< index of target layer */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency */
   SCIP_VAR**            targetgnnoutputvars, /**< array of output variables at nodes of target layer (or NULL) */
   SCIP_VAR**            sourceauxvars,      /**< array of auxiliary variables from source layer (or NULL) */
   SCIP_VAR**            sourcegnnoutputvars, /**< array of output variables at nodes of source layer */
   GNN_DATA*             gnndata             /**< data of GNN */
   )
{
   char name[SCIP_MAXSTRLEN];
   GNN_LAYERINFO_SAGE* sageinfo;
   GNN_LAYERINFO_POOL* poolinfo;
   GNN_LAYERINFO_DENSE* denseinfo;
   SCIP_VAR** vars;
   SCIP_Real* coefs;
   SCIP_Real bias;
   SCIP_Real scale = 1.0;
   int cntconss;
   int storagelen;
   int nodebound;
   int cnt;
   int idx;
   int v;
   int w;
   int f;
   int f2;

   assert(scip != NULL);
   assert(layerlinkingconss != NULL);
   assert(nnodes > 0);
   assert(ntargetfeatures > 0);
   assert(nsourcefeatures > 0);
   assert(targetlayeridx > 0);
   assert(targetgnnoutputvars != NULL);
   assert(sourceauxvars != NULL || sourcetype == GNN_LAYERTYPE_POOL || sourcetype == GNN_LAYERTYPE_DENSE
      || targettype != GNN_LAYERTYPE_SAGE);
   assert(sourcegnnoutputvars != NULL);
   assert(targetactivation == GNN_ACTIVATIONTYPE_NONE);

   /* possibly ignore nodes if we are in a dense layer and get information about layer */
   if( targettype == GNN_LAYERTYPE_DENSE )
   {
      nodebound = 1;
      denseinfo = SCIPgetGNNLayerinfoDense(gnndata, targetlayeridx);
   }
   else if( targettype == GNN_LAYERTYPE_POOL )
   {
      nodebound = 1;
      poolinfo = SCIPgetGNNLayerinfoPool(gnndata, targetlayeridx);
   }
   else
   {
      assert(targettype == GNN_LAYERTYPE_SAGE);
      nodebound = nnodes;
      sageinfo = SCIPgetGNNLayerinfoSage(gnndata, targetlayeridx);
   }

   /* allocate temporary memory for variables and coefficients */
   storagelen = nodebound * nsourcefeatures + 1;
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, storagelen) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, storagelen) );

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, layerlinkingconss, nodebound * nsourcefeatures) );

   /* iterate through nodes of GNN layer */
   cntconss = 0;
   for( v = 0; v < nodebound; ++v )
   {
      for( f = 0; f < ntargetfeatures; ++f )
      {
         /* set variables and coefficients as the gnnoutputvar of the current layer's node
          * and the input expression of the ReLU activation function
          */
         idx = SCIPgetGNNNodevarIdxLayer(nodebound, ntargetfeatures, v, f);
         vars[0] = targetgnnoutputvars[idx];
         coefs[0] = 1.0;

         if( targettype == GNN_LAYERTYPE_DENSE )
            bias = SCIPgetDenseLayerFeatureBias(denseinfo, f);
         else if( targettype == GNN_LAYERTYPE_POOL )
            bias = 0.0;
         else
            bias = SCIPgetSageLayerFeatureBias(sageinfo, f);

         cnt = 1;
         if( targettype == GNN_LAYERTYPE_SAGE )
         {
            assert(sourcetype == GNN_LAYERTYPE_INPUT || sourcetype == GNN_LAYERTYPE_SAGE);
            for( w = 0; w < nnodes; ++w )
            {
               for( f2 = 0; f2 < nsourcefeatures; ++f2 )
               {
                  /* in the input layer, there are no auxiliary variables due to fixed features */
                  if( targetlayeridx == 1 )
                  {
                     idx = SCIPgetGNNNodevarIdxLayer(nnodes, nsourcefeatures, w, f2);
                     assert(SCIPisEQ(scip, SCIPvarGetLbLocal(sourcegnnoutputvars[idx]),
                           SCIPvarGetUbLocal(sourcegnnoutputvars[idx])));
                     if( v != w )
                     {
                        SCIP_Real c;
                        vars[cnt] = adjacencyvars[getEdgeIdx(v, w, nnodes)];
                        c = -SCIPgetSageLayerFeatureEdgeweight(sageinfo, f2, f);
                        c = c * SCIPvarGetLbLocal(sourcegnnoutputvars[idx]);
                        coefs[cnt++] = c;
                     }
                     else
                     {
                        /* the auxiliary variable coincides with the (fixed) output variable (in our applications, input is fixed) */
                        bias += SCIPgetSageLayerFeatureNodeweight(sageinfo, f2, f) * SCIPvarGetLbLocal(sourcegnnoutputvars[idx]);
                     }
                  }
                  else
                  {
                     idx = SCIPgetAuxvarIdxLayer(nnodes, nsourcefeatures, v, w, f2);
                     vars[cnt] = sourceauxvars[idx];

                     if( v == w )
                        coefs[cnt++] = -SCIPgetSageLayerFeatureNodeweight(sageinfo, f2, f);
                     else
                        coefs[cnt++] = -SCIPgetSageLayerFeatureEdgeweight(sageinfo, f2, f);
                  }
               }
            }
         }
         else if( targettype == GNN_LAYERTYPE_POOL )
         {
            assert(sourcetype == GNN_LAYERTYPE_SAGE);
            assert(nsourcefeatures == ntargetfeatures);

            if( SCIPgetTypePoolLayer(poolinfo) == GNN_POOLTYPE_ADD )
               scale = 1.0;
            else
               assert(FALSE);

            for( w = 0; w < nnodes; ++w )
            {
               idx = SCIPgetGNNNodevarIdxLayer(nnodes, nsourcefeatures, w, f);
               vars[cnt] = sourcegnnoutputvars[idx];
               coefs[cnt++] = -scale;
            }
         }
         else
         {
            assert(sourcetype == GNN_LAYERTYPE_POOL || sourcetype == GNN_LAYERTYPE_DENSE);
            for( f2 = 0; f2 < nsourcefeatures; ++f2 )
            {
               idx = SCIPgetGNNNodevarIdxLayer(1, nsourcefeatures, v, f2);
               vars[cnt] = sourcegnnoutputvars[idx];

               coefs[cnt++] = -SCIPgetDenseLayerFeatureWeight(denseinfo, f2, f);
            }
         }

         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "linklayers%d#%d#%d", targetlayeridx, v, f);
         SCIP_CALL( SCIPcreateConsLinear(scip, &(*layerlinkingconss)[cntconss], name, cnt, vars, coefs, bias, bias,
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );
         SCIP_CALL( SCIPaddCons(scip, (*layerlinkingconss)[cntconss++]) );
      }
   }

   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}

/** creates constraints to link consecutive layers */
static
SCIP_RETCODE linkConsecutiveLayers(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS***          layerlinkingconss,  /**< pointer to store layer linking conss */
   int                   nnodes,             /**< number of nodes of underlying graph */
   GNN_LAYERTYPE         targettype,         /**< type of target layer */
   GNN_ACTIVATIONTYPE    targetactivation,   /**< activation function at target layer */
   GNN_LAYERTYPE         sourcetype,         /**< type of source layer */
   GNN_ACTIVATIONTYPE    sourceactivation,   /**< activation function at source layer */
   int                   ntargetfeatures,    /**< number of features in target layer */
   int                   nsourcefeatures,    /**< number of features in source layer */
   int                   targetlayeridx,     /**< index of target layer */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency */
   SCIP_VAR**            targetgnnoutputvars, /**< array of output variables at nodes of target layer (or NULL) */
   SCIP_VAR**            targetisactivevars, /**< array of variables indicating activation at target node */
   SCIP_VAR**            sourceauxvars,      /**< array of auxiliary variables from source layer (or NULL) */
   SCIP_VAR**            sourcegnnoutputvars, /**< array of output variables at nodes of source layer */
   SCIP_Real*            lbinputactivation,  /**< array of lower bounds on the input of activation functions */
   SCIP_Real*            ubinputactivation,  /**< array of upper bounds on the input of activation functions */
   GNN_DATA*             gnndata             /**< data of GNN */
   )
{
   assert(scip != NULL);
   assert(layerlinkingconss != NULL);
   assert(nnodes > 0);
   assert(ntargetfeatures > 0);
   assert(nsourcefeatures > 0);
   assert(targetlayeridx > 0);
   assert(targetgnnoutputvars != NULL);
   assert(targetisactivevars != NULL || targetactivation != GNN_ACTIVATIONTYPE_RELU);
   assert(sourceauxvars != NULL || sourcetype == GNN_LAYERTYPE_POOL || sourcetype == GNN_LAYERTYPE_DENSE
      || targettype != GNN_LAYERTYPE_SAGE || targetlayeridx == 1);
   assert(sourcegnnoutputvars != NULL);
   assert(lbinputactivation != NULL || targetactivation != GNN_ACTIVATIONTYPE_RELU);
   assert(ubinputactivation != NULL || targetactivation != GNN_ACTIVATIONTYPE_RELU);

   if( targetactivation == GNN_ACTIVATIONTYPE_RELU )
   {
      SCIP_CALL( createReluConssLayer(scip, layerlinkingconss, nnodes, targettype, targetactivation,
            sourcetype, sourceactivation, ntargetfeatures, nsourcefeatures, targetlayeridx,
            adjacencyvars, targetgnnoutputvars, targetisactivevars, sourceauxvars, sourcegnnoutputvars,
            lbinputactivation, ubinputactivation, gnndata) );
   }
   else
   {
      assert(targetactivation == GNN_ACTIVATIONTYPE_NONE);

      SCIP_CALL( createPlainLinkingLayer(scip, layerlinkingconss, nnodes, targettype, targetactivation,
            sourcetype, sourceactivation, ntargetfeatures, nsourcefeatures, targetlayeridx, adjacencyvars,
            targetgnnoutputvars, sourceauxvars, sourcegnnoutputvars, gnndata) );
   }

   return SCIP_OKAY;
}

/** creates layer-based variables and constraints */
static
SCIP_RETCODE probdataCreateLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROBDATA*        sourceprobdata,     /**< problem data which shall be copied (or NULL) */
   SCIP_PROBDATA*        targetprobdata,     /**< problem data to be created (basic information already assigned) */
   SCIP_Real**           featurelb,          /**< (nnodes x nfeatures)-matrix of lower bounds on feature assignments */
   SCIP_Real**           featureub,          /**< (nnodes x nfeatures)-matrix of upper bounds on feature assignments */
   SCIP_Real**           lbgnnoutputvars,    /**< array of lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< array of lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< array of upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< array of upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Bool             uselprelax          /**< whether just the LP relaxation shall be solved */
   )
{
   GNN_DATA* gnndata;
   GNN_LAYERTYPE type;
   GNN_LAYERTYPE typeprev;
   GNN_ACTIVATIONTYPE activation;
   GNN_ACTIVATIONTYPE activationprev;
   SCIP_VAR** auxvarsprev;
   SCIP_VAR** gnnoutputvarsprev;
   int nlayers;
   int nfeatures;
   int nfeaturesprev;
   int nnodes;
   int ngnnoutputvars;
   int nauxvars;
   int l;

   assert(scip != NULL);
   assert(targetprobdata != NULL);
   assert(targetprobdata->gnndata != NULL);

   gnndata = targetprobdata->gnndata;
   nlayers = SCIPgetGNNNLayers(gnndata);
   nnodes = targetprobdata->nnodes;

   /* either copy or create arrays */
   if( sourceprobdata != NULL )
   {
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->gnnoutputvars,
            sourceprobdata->gnnoutputvars, nlayers) );
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->auxvars,
            sourceprobdata->auxvars, nlayers) );
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->isactivevars,
            sourceprobdata->isactivevars, nlayers) );
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->linkauxconss,
            sourceprobdata->linkauxconss, nlayers) );
      SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->layerlinkingconss,
            sourceprobdata->layerlinkingconss, nlayers) );
   }
   else
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &targetprobdata->gnnoutputvars, nlayers) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &targetprobdata->auxvars, nlayers) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &targetprobdata->isactivevars, nlayers) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &targetprobdata->linkauxconss, nlayers) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &targetprobdata->layerlinkingconss, nlayers) );
   }

   for( l = 0; l < nlayers; ++l )
   {
      SCIP_CALL( getDataLayer(gnndata, nnodes, l, &type, &activation, &nfeatures, &ngnnoutputvars, &nauxvars) );

      /* either create variables and constraints from scratch or copy them */
      if( sourceprobdata != NULL )
      {
         SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->gnnoutputvars[l],
               sourceprobdata->gnnoutputvars[l], ngnnoutputvars) );
         if( nauxvars > 0 && l > 0 )
         {
            SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->auxvars[l],
                  sourceprobdata->auxvars[l], nauxvars) );
         }
         else
            targetprobdata->auxvars[l] = NULL;
         if( activation == GNN_ACTIVATIONTYPE_RELU )
         {
            SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->isactivevars[l],
                  sourceprobdata->isactivevars[l], ngnnoutputvars) );
         }
         else
         {
            assert(activation == GNN_ACTIVATIONTYPE_NONE);
            targetprobdata->isactivevars[l] = NULL;
         }
         if( nauxvars > 0 && l > 0 )
         {
            SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->linkauxconss[l],
                  sourceprobdata->linkauxconss[l], 4 * nauxvars) );
         }
         else
            targetprobdata->linkauxconss[l] = NULL;
         if( activation == GNN_ACTIVATIONTYPE_RELU )
         {
            SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->layerlinkingconss[l],
                  sourceprobdata->layerlinkingconss[l], 4 * ngnnoutputvars) );
         }
         else
         {
            assert(activation == GNN_ACTIVATIONTYPE_NONE);

            /* there is no activation in the first layer */
            if( l > 0 )
            {
               SCIP_CALL( SCIPduplicateBlockMemoryArray(scip, &targetprobdata->layerlinkingconss[l],
                     sourceprobdata->layerlinkingconss[l], ngnnoutputvars) );
            }
            else
               sourceprobdata->layerlinkingconss[l] = NULL;
         }
      }
      else
      {
         assert(featurelb != NULL);
         assert(featureub != NULL);

         SCIP_CALL( createGNNOutputVarsLayer(scip, &targetprobdata->gnnoutputvars[l], type,
               nnodes, nfeatures, l, featurelb, featureub, lbgnnoutputvars[l], ubgnnoutputvars[l]) );

         /* only create auxiliary variables if the next layer makes use of them
          * (we assume that all input features are fixed, so no auxvars needed for input layer)
          */
         if( nauxvars > 0 && l > 0 )
         {
            SCIP_CALL( createAuxVarsLayer(scip, &targetprobdata->auxvars[l], type,
                  nnodes, nfeatures, l, lbauxvars[l], ubauxvars[l]) );
         }
         else
            targetprobdata->auxvars[l] = NULL;

         SCIP_CALL( createIsactiveVarsLayer(scip, &targetprobdata->isactivevars[l], type, activation,
               nnodes, nfeatures, l, lbgnnoutputvars[l], ubgnnoutputvars[l], uselprelax) );
         if( nauxvars > 0 && l > 0 )
         {
            SCIP_CALL( createLinkAuxConssLayer(scip, &targetprobdata->linkauxconss[l], type, activation,
                  nnodes, nfeatures, l, targetprobdata->adjacencyvars, targetprobdata->gnnoutputvars[l],
                  targetprobdata->auxvars[l], lbgnnoutputvars[l], lbauxvars[l], ubgnnoutputvars[l], ubauxvars[l]) );
         }
         else
            targetprobdata->linkauxconss[l] = NULL;

         /* there is no link to previous layers in input layers*/
         if( l > 0 )
         {
            SCIP_CALL( linkConsecutiveLayers(scip, &targetprobdata->layerlinkingconss[l], nnodes, type, activation,
                  typeprev, activationprev, nfeatures, nfeaturesprev, l, targetprobdata->adjacencyvars,
                  targetprobdata->gnnoutputvars[l], targetprobdata->isactivevars[l], auxvarsprev, gnnoutputvarsprev,
                  lbnodecontent[l], ubnodecontent[l], gnndata) );
         }
         else
         {
            assert(activation == GNN_ACTIVATIONTYPE_NONE);
            targetprobdata->layerlinkingconss[0] = NULL;
         }

         /* store characteristics of this layer for next layer */
         typeprev = type;
         activationprev = activation;
         nfeaturesprev = nfeatures;
         gnnoutputvarsprev = targetprobdata->gnnoutputvars[l];
         auxvarsprev = targetprobdata->auxvars[l];
         assert(type == GNN_LAYERTYPE_INPUT || type == GNN_LAYERTYPE_SAGE || auxvarsprev == NULL);
      }
   }

   return SCIP_OKAY;
}

/** creates problem data */
static
SCIP_RETCODE createProbdata(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROBDATA*        sourcedata,         /**< problem data that shall be copied (or NULL) */
   SCIP_PROBDATA**       targetdata,         /**< pointer to problem data that shall be created */
   GNN_DATA*             gnndata,            /**< data of underlying GNN */
   int                   nnodes,             /**< number of nodes in underlying graph */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   int                   globalbudget,       /**< global attack budget */
   int*                  localbudget,        /**< local attack budget for every node */
   int                   origclass,          /**< original classification of underlying graph */
   int                   targetclass,        /**< target classification of underlying graph */
   SCIP_Real**           featurelb,          /**< lower bounds on feature assignment per layer */
   SCIP_Real**           featureub,          /**< upper bounds on feature assignment per layer */
   SCIP_Real**           lbgnnoutputvars,    /**< lower bounds on output variables (or NULL) */
   SCIP_Real**           lbauxvars,          /**< lower bounds on auxiliary variables (or NULL) */
   SCIP_Real**           lbnodecontent,      /**< lower bounds on inputs for activatioin function (or NULL) */
   SCIP_Real**           ubgnnoutputvars,    /**< upper bounds on output variables (or NULL) */
   SCIP_Real**           ubauxvars,          /**< upper bounds on auxiliary variables (or NULL) */
   SCIP_Real**           ubnodecontent,      /**< upper bounds on inputs for activatioin function (or NULL) */
   SCIP_Bool             uselprelax          /**< whether just the LP relaxation shall be solved */
   )
{
   SCIP_CALL( probdataCreateBasic(scip, targetdata, gnndata, nnodes, adjacencymatrix,
         globalbudget, localbudget, origclass, targetclass, uselprelax) );
   SCIP_CALL( probdataCreateGlobal(scip, sourcedata, *targetdata, uselprelax) );
   SCIP_CALL( probdataCreateLayer(scip, sourcedata, *targetdata, featurelb, featureub,
         lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent, uselprelax) );

   return SCIP_OKAY;
}

/** sets objective of robust classification problem */
static
SCIP_RETCODE setObjective(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data of underlying GNN */
   SCIP_VAR***           gnnoutputvars,      /**< output variables of the problem */
   int                   origclass,          /**< original classification of underlying graph */
   int                   targetclass         /**< target classification of underlying graph */
   )
{
   int nlayers;
#ifndef NDEBUG
   GNN_LAYERTYPE type;
   int nfeatures;
#endif
   int lidx;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(gnnoutputvars != NULL);

   nlayers = SCIPgetGNNNLayers(gnndata);
   lidx = nlayers - 1;

#ifndef NDEBUG
   type = SCIPgetGNNLayerType(gnndata, lidx);
   assert(type == GNN_LAYERTYPE_POOL || type == GNN_LAYERTYPE_DENSE);

   if( type == GNN_LAYERTYPE_POOL )
   {
      GNN_LAYERINFO_POOL* poolinfo;

      poolinfo = SCIPgetGNNLayerinfoPool(gnndata, lidx);
      nfeatures = SCIPgetNOutputFeaturesPoolLayer(poolinfo);
   }
   else
   {
      GNN_LAYERINFO_DENSE* denseinfo;

      denseinfo = SCIPgetGNNLayerinfoDense(gnndata, lidx);
      nfeatures = SCIPgetNOutputFeaturesDenseLayer(denseinfo);
   }
   assert(0 <= origclass && origclass < nfeatures);
   assert(0 <= targetclass && targetclass < nfeatures);
#endif
   assert(gnnoutputvars[lidx] != NULL);
   assert(gnnoutputvars[lidx][origclass] != NULL);
   assert(gnnoutputvars[lidx][targetclass] != NULL);

   SCIP_CALL( SCIPchgVarObj(scip, gnnoutputvars[lidx][origclass], 1.0) );
   SCIP_CALL( SCIPchgVarObj(scip, gnnoutputvars[lidx][targetclass], -1.0) );

   /* set objective sense */
   SCIP_CALL( SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE) );

   return SCIP_OKAY;
}

/** fres problem data */
static
SCIP_RETCODE probdataFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROBDATA**       probdata            /**< pointer to problem data */
   )
{
   int nlayers;
   int nadjvars;
   int nnodes;
   int l;
   int i;

   assert(scip != NULL);
   assert(probdata != NULL);

   nnodes = (*probdata)->nnodes;
   nadjvars = nnodes * (nnodes - 1) / 2;
   nlayers = (*probdata)->nlayers;

   /* free basic structures */
   for( i = 0; i < nnodes; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &(*probdata)->adjacencymatrix[i], nnodes);
   }
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->adjacencymatrix, nnodes);
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->localbudget, nnodes);

   /* free global variables and constraints */
   for( i = 0; i < nadjvars; ++i )
   {
      SCIP_CALL( SCIPreleaseVar(scip, &(*probdata)->adjacencyvars[i]) );
   }
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->adjacencyvars, nadjvars);

   SCIP_CALL( SCIPreleaseCons(scip, &(*probdata)->globalattackcons) );

   for( i = 0; i < nnodes; ++i )
   {
      SCIP_CALL( SCIPreleaseCons(scip, &(*probdata)->localattackconss[i]) );
   }
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->localattackconss, nnodes);

   /* free layer-based structures */
   for( l = 0; l < nlayers; ++l )
   {
      int nfeatures;
      int ngnnoutputvars;
      int nauxvars;
      int nodebound;
      int len;

      nfeatures = (*probdata)->noutputfeatures[l];
      SCIP_CALL( getDataLayerProbdata(l, nnodes, nlayers, (*probdata)->layertypes,
            (*probdata)->noutputfeatures, &ngnnoutputvars, &nauxvars) );
      if( (*probdata)->layertypes[l] == GNN_LAYERTYPE_SAGE )
         nodebound = nnodes;
      else
         nodebound = 1;

      for( i = 0; i < ngnnoutputvars; ++i )
      {
         SCIP_CALL( SCIPreleaseVar(scip, &(*probdata)->gnnoutputvars[l][i]) );
      }
      SCIPfreeBlockMemoryArray(scip, &(*probdata)->gnnoutputvars[l], ngnnoutputvars);
      if( l > 0 )
      {
         for( i = 0; i < nauxvars; ++i )
         {
            SCIP_CALL( SCIPreleaseVar(scip, &(*probdata)->auxvars[l][i]) );
         }
         SCIPfreeBlockMemoryArrayNull(scip, &(*probdata)->auxvars[l], nauxvars);
      }
      if( (*probdata)->isactivevars[l] )
      {
         for( i = 0; i < ngnnoutputvars; ++i )
         {
            SCIP_CALL( SCIPreleaseVar(scip, &(*probdata)->isactivevars[l][i]) );
         }
         SCIPfreeBlockMemoryArray(scip, &(*probdata)->isactivevars[l], ngnnoutputvars);
      }
      if( nauxvars > 0 && l > 0 )
      {
         for( i = 0; i < 4 * nauxvars; ++i )
         {
            SCIP_CALL( SCIPreleaseCons(scip, &(*probdata)->linkauxconss[l][i]) );
         }
         SCIPfreeBlockMemoryArray(scip, &(*probdata)->linkauxconss[l], 4 * nauxvars);
      }

      len = nodebound * nfeatures;
      if( (*probdata)->activation[l] == GNN_ACTIVATIONTYPE_RELU )
         len *= 4;

      /* there are no linking constraints in the first layer */
      if( l > 0 )
      {
         for( i = 0; i < len; ++i )
         {
            SCIP_CALL( SCIPreleaseCons(scip, &(*probdata)->layerlinkingconss[l][i]) );
         }
         SCIPfreeBlockMemoryArray(scip, &(*probdata)->layerlinkingconss[l], len);
      }
      else
         assert((*probdata)->layerlinkingconss[l] == NULL);
   }
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->gnnoutputvars, nlayers);
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->auxvars, nlayers);
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->isactivevars, nlayers);
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->linkauxconss, nlayers);
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->layerlinkingconss, nlayers);
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->layertypes, nlayers);
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->activation, nlayers);
   SCIPfreeBlockMemoryArray(scip, &(*probdata)->noutputfeatures, nlayers);

   /* free probdata itself */
   SCIPfreeBlockMemory(scip, probdata);

   return SCIP_OKAY;
}

/** transforms problem data */
static
SCIP_RETCODE transformData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROBDATA*        probdata            /**< problem data to be transformed */
   )
{
   GNN_LAYERTYPE type;
   GNN_ACTIVATIONTYPE activation;
   int nfeatures;
   int nlayers;
   int nnodes;
   int l;

   assert(scip != NULL);
   assert(probdata != NULL);

   nlayers = SCIPgetGNNNLayers(probdata->gnndata);
   nnodes = probdata->nnodes;

   /* transform constraints */
   SCIP_CALL( SCIPtransformCons(scip, probdata->globalattackcons, &probdata->globalattackcons) );
   SCIP_CALL( SCIPtransformConss(scip, nnodes, probdata->localattackconss, probdata->localattackconss) );
   for( l = 0; l < nlayers; ++l )
   {
      type = probdata->layertypes[l];
      activation = probdata->activation[l];

      if( probdata->linkauxconss[l] != NULL )
      {
         assert(type == GNN_LAYERTYPE_INPUT || type == GNN_LAYERTYPE_SAGE);

         nfeatures = probdata->noutputfeatures[l] * nnodes * nnodes;
         SCIP_CALL( SCIPtransformConss(scip, 4 * nfeatures, probdata->linkauxconss[l], probdata->linkauxconss[l]) );
      }

      assert(probdata->layerlinkingconss[l] != NULL || l == 0);
      if( probdata->layerlinkingconss[l] != NULL )
      {
         if( type == GNN_LAYERTYPE_SAGE )
            nfeatures = probdata->noutputfeatures[l] * nnodes;
         else
            nfeatures = probdata->noutputfeatures[l];

         if( activation == GNN_ACTIVATIONTYPE_RELU )
         {
            SCIP_CALL( SCIPtransformConss(scip, 4 * nfeatures, probdata->layerlinkingconss[l],
                  probdata->layerlinkingconss[l]) );
         }
         else
         {
            assert(activation == GNN_ACTIVATIONTYPE_NONE);
            SCIP_CALL( SCIPtransformConss(scip, nfeatures, probdata->layerlinkingconss[l],
                  probdata->layerlinkingconss[l]) );
         }
      }
   }

   /* transform variables */
   SCIP_CALL( SCIPtransformVars(scip, nnodes * (nnodes - 1) / 2, probdata->adjacencyvars, probdata->adjacencyvars) );
   for( l = 0; l < nlayers; ++l )
   {
      type = probdata->layertypes[l];
      activation = probdata->activation[l];

      switch( type )
      {
      case GNN_LAYERTYPE_INPUT:
      case GNN_LAYERTYPE_SAGE:
         nfeatures = nnodes * probdata->noutputfeatures[l];
         SCIP_CALL( SCIPtransformVars(scip, nfeatures, probdata->gnnoutputvars[l], probdata->gnnoutputvars[l]) );
         break;
      default:
         nfeatures = probdata->noutputfeatures[l];
         SCIP_CALL( SCIPtransformVars(scip, nfeatures, probdata->gnnoutputvars[l], probdata->gnnoutputvars[l]) );
      }

      switch( type )
      {
      case GNN_LAYERTYPE_INPUT:
      case GNN_LAYERTYPE_SAGE:
         nfeatures = nnodes * nnodes * probdata->noutputfeatures[l];
         if( probdata->auxvars[l] != NULL )
         {
            SCIP_CALL( SCIPtransformVars(scip, nfeatures, probdata->auxvars[l], probdata->auxvars[l]) );
         }
         break;
      default:
         assert(probdata->auxvars[l] == NULL);
      }

      switch( activation )
      {
      case GNN_ACTIVATIONTYPE_RELU:
         assert(type == GNN_LAYERTYPE_SAGE || type == GNN_LAYERTYPE_DENSE );

         if( type == GNN_LAYERTYPE_SAGE )
            nfeatures = nnodes * probdata->noutputfeatures[l];
         else
            nfeatures = probdata->noutputfeatures[l];

         SCIP_CALL( SCIPtransformVars(scip, nfeatures, probdata->isactivevars[l], probdata->isactivevars[l]) );
         break;
      default:
         assert(probdata->isactivevars[l] == NULL);
      }
   }

   return SCIP_OKAY;
}

/**@name Callback methods of problem data
 *
 * @{
 */

/** frees user data of original problem (called when the original problem is freed) */
static
SCIP_DECL_PROBDELORIG(probdelorigRobustClassify)
{
   SCIPdebugMsg(scip, "free original problem data\n");

   SCIP_CALL( probdataFree(scip, probdata) );

   return SCIP_OKAY;
}

/** frees user data of transformed problem (called when the transformed problem is freed) */
static
SCIP_DECL_PROBDELTRANS(probdeltransRobustClassify)
{
   SCIPdebugMsg(scip, "free transformed problem data\n");

   SCIP_CALL( probdataFree(scip, probdata) );

   return SCIP_OKAY;
}

/** creates user data of transformed problem by transforming the original user problem data
 *  (called after problem was transformed) */
static
SCIP_DECL_PROBTRANS(probtransRobustClassify)
{
   SCIP_CALL( createProbdata(scip, sourcedata, targetdata, sourcedata->gnndata, sourcedata->nnodes,
      sourcedata->adjacencymatrix, sourcedata->globalbudget, sourcedata->localbudget,
         sourcedata->graphclassification, sourcedata->targetclassification,
         NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, sourcedata->uselprelax) );

   SCIP_CALL( transformData(scip, *targetdata) );

   return SCIP_OKAY;
}

/**@} */

/** sets up the problem data */
SCIP_RETCODE SCIPprobdataCreateRobustClassify(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata,        /**< data about optimization problem on GNN */
   SCIP_Real**           lbgnnoutputvars,    /**< array of lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< array of lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< array of upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< array of upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Bool             uselprelax          /**< whether we just want to solve the LP relaxation */
   )
{
   GNNPROB_ROBUSTCLASSIFY* rcprobdata;
   SCIP_PROBDATA* probdata;
   SCIP_Bool** adjacencymatrix;
   SCIP_Real** featurelb;
   SCIP_Real** featureub;
   int* localbudget;
   int graphclassification;
   int targetclassification;
   int globalbudget;
   int nnodes;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(gnnprobdata != NULL);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);

   rcprobdata = SCIPgetGNNProbDataRobustClassify(gnnprobdata);
   assert(rcprobdata != NULL);

   /* create problem in SCIP and add non-NULL callbacks via setter functions */
   SCIP_CALL( SCIPcreateProbBasic(scip, "") );

   SCIP_CALL( SCIPsetProbDelorig(scip, probdelorigRobustClassify) );
   SCIP_CALL( SCIPsetProbTrans(scip, probtransRobustClassify) );
   SCIP_CALL( SCIPsetProbDeltrans(scip, probdeltransRobustClassify) );

   /* create basic problem data */
   nnodes = SCIPgetNNodesRobustClassify(rcprobdata);
   adjacencymatrix = SCIPgetAdjacencyMatrixRobustClassify(rcprobdata);
   globalbudget = SCIPgetGlobalBudgetRobustClassify(rcprobdata);
   localbudget = SCIPgetLocalBudgetRobustClassify(rcprobdata);
   graphclassification = SCIPgetGraphClassificationRobustClassify(rcprobdata);
   targetclassification = SCIPgetTargetClassificationRobustClassify(rcprobdata);

   featurelb = SCIPgetFeatureLbRobustClassify(rcprobdata);
   featureub = SCIPgetFeatureUbRobustClassify(rcprobdata);

   SCIP_CALL( createProbdata(scip, NULL, &probdata, gnndata, nnodes, adjacencymatrix,
         globalbudget, localbudget, graphclassification, targetclassification, featurelb, featureub,
         lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent, uselprelax) );

   if( ! uselprelax )
   {
      SCIP_CALL( setObjective(scip, gnndata, probdata->gnnoutputvars, graphclassification, targetclassification) );
   }

   /* set user problem data */
   SCIP_CALL( SCIPsetProbData(scip, probdata) );

   return SCIP_OKAY;
}

/** returns number of layers in GNN */
GNN_DATA* SCIPgetProbdataRobustClassifyGNNNData(
   SCIP_PROBDATA*        probdata            /**< problem data */
   )
{
   assert(probdata!= NULL);
   return probdata->gnndata;
}

/** returns adjacency matrix of graph of in robust classification problem */
SCIP_Bool** SCIPgetProbdataRobustClassifyAdjacencyMatrix(
   SCIP_PROBDATA*        probdata            /**< problem data */
   )
{
   assert(probdata!= NULL);
   return probdata->adjacencymatrix;
}

/** returns number of nodes of graph in robust classification problem */
int SCIPgetProbdataRobustClassifyNNodes(
   SCIP_PROBDATA*        probdata            /**< problem data */
   )
{
   assert(probdata!= NULL);
   return probdata->nnodes;
}

/** returns global attack budget in robust classification problem */
int SCIPgetProbdataRobustClassifyGlobalBudget(
   SCIP_PROBDATA*        probdata            /**< problem data */
   )
{
   assert(probdata!= NULL);
   return probdata->globalbudget;
}

/** returns local attack budget in robust classification problem */
int* SCIPgetProbdataRobustClassifyLocalBudget(
   SCIP_PROBDATA*        probdata            /**< problem data */
   )
{
   assert(probdata!= NULL);
   return probdata->localbudget;
}

/** returns GNN output variables */
SCIP_VAR*** SCIPgetProbdataRobustClassifyGNNOutputVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   )
{
   assert(probdata!= NULL);
   return probdata->gnnoutputvars;
}

/** returns auxiliary variables */
SCIP_VAR*** SCIPgetProbdataRobustClassifyAuxVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   )
{
   assert(probdata!= NULL);
   return probdata->auxvars;
}

/** returns variables modeling activity of activation function */
SCIP_VAR*** SCIPgetProbdataRobustClassifyIsActiveVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   )
{
   assert(probdata!= NULL);
   return probdata->isactivevars;
}

/** returns variables modeling adjacency */
SCIP_VAR** SCIPgetProbdataRobustClassifyAdjacencyVars(
   SCIP_PROBDATA*        probdata            /**< problem data */
   )
{
   assert(probdata!= NULL);
   return probdata->adjacencyvars;
}

/** sets objective for a neuron for OBBT */
SCIP_RETCODE SCIPsetOBBTobjective(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nfeatures,          /**< number of features */
   int                   layeridx,           /**< index of layer */
   int                   nodeidx,            /**< index of node of underlying graph */
   int                   featureidx,         /**< index of feature */
   SCIP_Bool             maximize            /**< whether objective sense is maximization */
   )
{
   SCIP_PROBDATA* probdata;

   assert(scip != NULL);
   assert(nodeidx >= 0);
   assert(featureidx >= 0);

   probdata = SCIPgetProbData(scip);

   SCIP_CALL( SCIPchgVarObj(scip, probdata->gnnoutputvars[layeridx][nodeidx*nfeatures + featureidx], 1.0) );

   if( maximize )
   {
      SCIP_CALL( SCIPsetObjsense(scip, SCIP_OBJSENSE_MAXIMIZE) );
   }
   else
   {
      SCIP_CALL( SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE) );
   }

   return SCIP_OKAY;
}

/** resets objective for a neuron for OBBT */
SCIP_RETCODE SCIPresetOBBTobjective(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nfeatures,          /**< number of features */
   int                   layeridx,           /**< index of layer */
   int                   nodeidx,            /**< index of node of underlying graph */
   int                   featureidx          /**< index of feature */
   )
{
   SCIP_PROBDATA* probdata;

   assert(scip != NULL);
   assert(nodeidx >= 0);
   assert(featureidx >= 0);

   probdata = SCIPgetProbData(scip);

   SCIP_CALL( SCIPchgVarObj(scip, probdata->gnnoutputvars[layeridx][nodeidx*nfeatures + featureidx], 0.0) );

   return SCIP_OKAY;
}
