/**@file   gnn_bounds_robustclassify.c
 * @brief  functions to compute bounds on variables and expressions in GNNs for robust classification problems
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "struct_gnn.h"
#include "type_gnn.h"
#include "gnn.h"
#include "gnn_bounds.h"
#include "gnn_bounds_robustclassify.h"

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

/** computes bounds for variables and expressions in an input layer */
static
SCIP_RETCODE computeBoundsGNNInputLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_INPUT*  layerinfo,          /**< information about input layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   int                   globalbudget,       /**< global attack budget */
   int*                  localbudget,        /**< local attack budget per node */
   SCIP_VAR**            gnnoutputvars,      /**< output variables of layer (or NULL) */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency in problem (or NULL) */
   SCIP_Real**           lbinput,            /**< lower bounds on input for GNN nodes (or NULL) */
   SCIP_Real**           ubinput,            /**< upper bounds on input for GNN nodes (or NULL) */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent       /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   )
{
   int nentries;
   int nfeatures;
   int i;
   int f;
   int v;

   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(ngraphnodes > 0);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);

   /* compute the number of nodes in input-sage layer */
   nfeatures = SCIPgetNInputFeaturesInputLayer(layerinfo);
   nentries = nfeatures * ngraphnodes;

   /*
    * allocate and populate bound arrays
    */

   /* input of input layers is scaled to be between 0 and 1 */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbgnnoutputvars, nentries) );
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         (*lbgnnoutputvars)[i] = 0.0;

         /* possibly improve the bound */
         if( lbinput != NULL )
            (*lbgnnoutputvars)[i] = MAX((*lbgnnoutputvars)[i], lbinput[v][f]);
         if( gnnoutputvars != NULL && SCIPisGT(scip, SCIPvarGetLbLocal(gnnoutputvars[i]), 0.0) )
            (*lbgnnoutputvars)[i] = SCIPvarGetLbLocal(gnnoutputvars[i]);
         ++i;
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubgnnoutputvars, nentries) );
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         (*ubgnnoutputvars)[i] = 1.0;

         /* possibly improve the bound */
         if( ubinput != NULL )
            (*ubgnnoutputvars)[i] = MIN((*ubgnnoutputvars)[i], ubinput[v][f]);
         if( gnnoutputvars != NULL && SCIPisLT(scip, SCIPvarGetUbLocal(gnnoutputvars[i]), 1.0) )
            (*ubgnnoutputvars)[i] = SCIPvarGetUbLocal(gnnoutputvars[i]);
         ++i;
      }
   }

   /* set data not needed (due to fixed input in our applications, we don't need auxiliary variables) */
   *lbnodecontent = NULL;
   *ubnodecontent = NULL;
   *lbauxvars = NULL;
   *ubauxvars = NULL;

   return SCIP_OKAY;
}

/** computes a lower or upper bound on the input for the activation function at a node of a sage layer */
static
SCIP_Real computeBoundGNNSageLayerNode(
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< information about sage layer */
   SCIP_Bool             computelb,          /**< whether a lower bound shall be computed */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency in problem (or NULL) */
   int                   budget,             /**< attack budget on adjacent edges */
   int                   sagegraphnodeidx,   /**< node index of sage layer for which bound shall be computed */
   int                   sagefeatureidx,     /**< feature index of sage layer for which bound shall be computed */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output variables from previous layer */
   SCIP_Real*            ubgnnoutputvarsprev, /**< upper bounds on output variables from previous layer */
   SCIP_Real*            weightbuffer        /**< allocated memory of length "ngraphnodes - 1" */
   )
{
   SCIP_Real bound;
   SCIP_Real ub;
   SCIP_Real lb;
   SCIP_Real weight;
   SCIP_Real mult;
#ifndef NDEBUG
   int noutputfeatures;
#endif
   SCIP_Bool hasforcedattack;
   int nforcedattacks;
   int ninputfeatures;
   int nsteps;
   int varidx;
   int edgeidx;
   int cnt;
   int f;
   int v;

   assert(layerinfo != NULL);
   assert(adjacencymatrix != NULL);
   assert(ngraphnodes > 0);
   assert(lbgnnoutputvarsprev != NULL);
   assert(ubgnnoutputvarsprev != NULL);
   assert(weightbuffer != NULL);

   ninputfeatures = SCIPgetNInputFeaturesSageLayer(layerinfo);
#ifndef NDEBUG
   noutputfeatures = SCIPgetNOutputFeaturesSageLayer(layerinfo);
   assert(0 <= sagefeatureidx && sagefeatureidx < noutputfeatures);
#endif
   assert(0 <= sagegraphnodeidx && sagegraphnodeidx < ngraphnodes);

   /* compute the bound as if there were no attacks */
   bound = SCIPgetSageLayerFeatureBias(layerinfo, sagefeatureidx);

   for( f = 0; f < ninputfeatures; ++f )
   {
      /*
       * bound contribution of node weight
       */
      weight = SCIPgetSageLayerFeatureNodeweight(layerinfo, f, sagefeatureidx);

      /* get the index of the output variable for feature f from PREVIOUS layer */
      varidx = sagegraphnodeidx * ninputfeatures + f;
      if( computelb )
      {
         if( weight >= 0 )
            bound += weight * lbgnnoutputvarsprev[varidx];
         else
            bound += weight * ubgnnoutputvarsprev[varidx];
      }
      else
      {
         if( weight >= 0 )
            bound += weight * ubgnnoutputvarsprev[varidx];
         else
            bound += weight * lbgnnoutputvarsprev[varidx];
      }

      /*
       * bound contribution of edge weights
       */
      for( v = 0; v < ngraphnodes; ++v )
      {
         /* skip edges to node itself or non-existing edges */
         if( v == sagegraphnodeidx || !adjacencymatrix[v][sagegraphnodeidx] )
            continue;

         weight = SCIPgetSageLayerFeatureEdgeweight(layerinfo, f, sagefeatureidx);

         /* get the index of the auxiliary variable for feature f from PREVIOUS layer */
         varidx = v * ninputfeatures + f;
         if( computelb )
         {
            if( weight >= 0 )
               bound += weight * lbgnnoutputvarsprev[varidx];
            else
               bound += weight * ubgnnoutputvarsprev[varidx];
         }
         else
         {
            if( weight >= 0 )
               bound += weight * ubgnnoutputvarsprev[varidx];
            else
               bound += weight * lbgnnoutputvarsprev[varidx];
         }
      }
   }

   /* weaken bound by largest weight changes caused by attacks within the budget */
   nforcedattacks = 0;
   for( v = 0, cnt = 0; v < ngraphnodes; ++v )
   {
      /* skip edge to node itself */
      if( v == sagegraphnodeidx )
         continue;

      edgeidx = getEdgeIdx(v, sagegraphnodeidx, ngraphnodes);
      lb = adjacencyvars != NULL ? SCIPvarGetLbLocal(adjacencyvars[edgeidx]) : 0.0;
      ub = adjacencyvars != NULL ? SCIPvarGetUbLocal(adjacencyvars[edgeidx]) : 1.0;

      /* we only proceed if (non-) existing edge is potentially attacked */
      if( adjacencymatrix[v][sagegraphnodeidx] && lb < 0.5 )
         mult = -1.0;
      else if( !adjacencymatrix[v][sagegraphnodeidx]  && ub > 0.5 )
         mult = 1.0;
      else
         continue;

      weightbuffer[cnt] = 0.0;
      hasforcedattack = FALSE;
      for( f = 0; f < ninputfeatures; ++f )
      {
         weight = SCIPgetSageLayerFeatureEdgeweight(layerinfo, f, sagefeatureidx);
         varidx = v * ninputfeatures + f;

         if( (adjacencymatrix[v][sagegraphnodeidx] && ub < 0.5)
            || (!adjacencymatrix[v][sagegraphnodeidx] && lb > 0.5) )
         {
            /* incorporate forced attacks */
            hasforcedattack = TRUE;
            if( computelb )
            {
               if( weight >= 0 )
                  bound += mult * weight * lbgnnoutputvarsprev[varidx];
               else
                  bound += mult * weight * ubgnnoutputvarsprev[varidx];
            }
            else
            {
               if( weight >= 0 )
                  bound += mult * weight * ubgnnoutputvarsprev[varidx];
               else
                  bound += mult * weight * lbgnnoutputvarsprev[varidx];
            }
         }
         else
         {
            /* store bound change if attack occurs */
            if( computelb )
            {
               if( weight >= 0 )
                  weightbuffer[cnt] += mult * weight * lbgnnoutputvarsprev[varidx];
               else
                  weightbuffer[cnt] += mult * weight * ubgnnoutputvarsprev[varidx];
            }
            else
            {
               if( weight >= 0 )
                  weightbuffer[cnt] += mult * weight * ubgnnoutputvarsprev[varidx];
               else
                  weightbuffer[cnt] += mult * weight * lbgnnoutputvarsprev[varidx];
            }
         }
      }
      if( hasforcedattack )
         ++nforcedattacks;
      ++cnt;
   }
   assert(nforcedattacks <= budget);

   /* terminate if the attack budget is saturated */
   if( nforcedattacks >= budget || cnt == 0 )
      return bound;

   /* find largest attacks within budget */
   SCIPsortReal(weightbuffer, cnt);

   nsteps = MIN(cnt, budget - nforcedattacks);
   if( computelb )
   {
      for( v = 0; v < nsteps; ++v )
      {
         if( weightbuffer[v] >= 0 )
            break;
         bound += weightbuffer[v];
      }
   }
   else
   {
      for( v = cnt - 1; v >= cnt - nsteps; --v )
      {
         if( weightbuffer[v] <= 0 )
            break;
         bound += weightbuffer[v];
      }
   }

   return bound;
}

/** computes bounds for variables and expressions in a sage layer */
static
SCIP_RETCODE computeBoundsGNNSageLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< information about sage layer */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   int                   globalbudget,       /**< global attack budget */
   int*                  localbudget,        /**< local attack budget per node */
   SCIP_VAR**            gnnoutputvars,      /**< output variables of layer (or NULL) */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency in problem (or NULL) */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output variables from previous layer */
   SCIP_Real*            ubgnnoutputvarsprev, /**< upper bounds on output variables from previous layer */
   SCIP_Bool             allocatearrays      /**< whether arrays for bounds need to be allocated */
   )
{
   GNN_ACTIVATIONTYPE activationtype;
   SCIP_Real* weightbuffer;
   SCIP_Real bound;
   int edgeidx;
   int nentries;
   int nfeatures;
   int i;
   int f;
   int v;
   int w;

   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(ngraphnodes > 0);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);
   assert(lbgnnoutputvarsprev != NULL);
   assert(ubgnnoutputvarsprev != NULL);

   /* compute the number of nodes in sage layer */
   nfeatures = SCIPgetNOutputFeaturesSageLayer(layerinfo);
   nentries = nfeatures * ngraphnodes;
   activationtype = SCIPgetSageLayerActivationType(layerinfo);

   /*
    * allocate and populate bound arrays
    */

   /* input of activation functions */
   if( allocatearrays )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbnodecontent, nentries) );
   }
   SCIP_CALL( SCIPallocBufferArray(scip, &weightbuffer, ngraphnodes - 1) );
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         bound = computeBoundGNNSageLayerNode(layerinfo, TRUE, adjacencymatrix, ngraphnodes,
            adjacencyvars, MIN(globalbudget, localbudget[v]), v, f,
            lbgnnoutputvarsprev, ubgnnoutputvarsprev, weightbuffer);
         (*lbnodecontent)[i++] = bound;
      }
   }
   if( allocatearrays )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubnodecontent, nentries) );
   }
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         bound = computeBoundGNNSageLayerNode(layerinfo, FALSE, adjacencymatrix, ngraphnodes,
            adjacencyvars, MIN(globalbudget, localbudget[v]), v, f,
            lbgnnoutputvarsprev, ubgnnoutputvarsprev, weightbuffer);
         (*ubnodecontent)[i++] = bound;
      }
   }
   SCIPfreeBufferArray(scip, &weightbuffer);

   /* output of activation functions */
   if( allocatearrays )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbgnnoutputvars, nentries) );
   }
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         bound = (*lbnodecontent)[i];
         switch( activationtype )
         {
         case GNN_ACTIVATIONTYPE_RELU:
            (*lbgnnoutputvars)[i] = MAX(0.0, bound);
            break;
         default:
            assert(activationtype == GNN_ACTIVATIONTYPE_NONE);
            (*lbgnnoutputvars)[i] = bound;
         }
         ++i;
      }
   }
   if( allocatearrays )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubgnnoutputvars, nentries) );
   }
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         bound = (*ubnodecontent)[i];
         switch( activationtype )
         {
         case GNN_ACTIVATIONTYPE_RELU:
            (*ubgnnoutputvars)[i] = MAX(0.0, bound);
            break;
         default:
            assert(activationtype == GNN_ACTIVATIONTYPE_NONE);
            (*ubgnnoutputvars)[i] = bound;
         }
         ++i;
      }
   }

   /* lower and upper bounds on auxiliary variables */
   nentries *= ngraphnodes;
   if( allocatearrays )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbauxvars, nentries) );
   }
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            SCIP_Real lb = 0.0;
            SCIP_Real ub = 1.0;
            int idx;

            idx = SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, w, f);
            edgeidx = -1;
            if( v != w )
               edgeidx = getEdgeIdx(v, w, ngraphnodes);

            if( adjacencyvars != NULL && edgeidx != -1 )
            {
               assert(edgeidx >= 0);
               lb = SCIPvarGetLbLocal(adjacencyvars[edgeidx]);
               ub = SCIPvarGetUbLocal(adjacencyvars[edgeidx]);
            }

            /* the auxiliary variable coincides with the gnnoutput variable if v = w or if the edge exists */
            if( v == w || lb > 0.5 )
               (*lbauxvars)[i] = (*lbgnnoutputvars)[idx];
            else if( ub < 0.5 )
               (*lbauxvars)[i] = 0.0;
            else
               (*lbauxvars)[i] = MIN(0.0, (*lbgnnoutputvars)[idx]);
            ++i;
         }
      }
   }
   if( allocatearrays )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubauxvars, nentries) );
   }
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            int idx;
            idx = SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, w, f);

            /* the auxiliary variable coincides with the gnnoutput variable if v = w or if the edge exists */
            if( v == w || activationtype == GNN_ACTIVATIONTYPE_RELU )
               (*ubauxvars)[i] = (*ubgnnoutputvars)[idx];
            else
            {
               assert(activationtype == GNN_ACTIVATIONTYPE_NONE);
               (*ubauxvars)[i] = MAX(0.0, (*ubgnnoutputvars)[idx]);
            }
            ++i;
         }
      }
   }

   return SCIP_OKAY;
}

/** computes bounds for all variables and expressions in a layer of a GNN for robust classification problems
 *
 *  @pre bounds on previous layers must have been computed and stored in bound arrays
 */
SCIP_RETCODE SCIPcomputeBoundsGNNRobustClassifyLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   int                   globalbudget,       /**< global attack budget */
   int*                  localbudget,        /**< local attack budget per node */
   SCIP_VAR**            gnnoutputvars,      /**< output variables of layer (or NULL) */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency in problem (or NULL) */
   SCIP_Real**           lbinput,            /**< lower bounds on input for GNN nodes (or NULL) */
   SCIP_Real**           ubinput,            /**< upper bounds on input for GNN nodes (or NULL) */
   int                   layeridx,           /**< index of layer for which bounds are computed */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            lbauxvarsprev,      /**< lower bounds on auxiliary variables at previous layer */
   SCIP_Real*            ubgnnoutputvarsprev, /**< upper bounds on output at previous GNN layer */
   SCIP_Real*            ubauxvarsprev,      /**< upper bounds on auxiliary variables at previous layer */
   SCIP_Bool             allocatearrays      /**< whether arrays for bounds need to be allocated */
   )
{
   GNN_LAYERINFO_INPUT* inputinfo;
   GNN_LAYERINFO_SAGE* sageinfo;
   GNN_LAYERINFO_POOL* poolinfo;
   GNN_LAYERINFO_DENSE* denseinfo;
   GNN_LAYERTYPE layertype;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(adjacencymatrix != NULL);
   assert(ngraphnodes > 0);
   assert(0 <= globalbudget && globalbudget <= ngraphnodes);
   assert(localbudget != NULL);
   assert(layeridx >= 0);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);

   layertype = SCIPgetGNNLayerType(gnndata, layeridx);
   assert((layeridx == 0) == (layertype == GNN_LAYERTYPE_INPUT));

   /* compute bounds from bounds on previous layer */
   switch( layertype )
   {
   case GNN_LAYERTYPE_INPUT:
      inputinfo = SCIPgetGNNLayerinfoInput(gnndata, layeridx);
      SCIP_CALL( computeBoundsGNNInputLayer(scip, inputinfo, ngraphnodes, globalbudget, localbudget,
            gnnoutputvars, adjacencyvars, lbinput, ubinput, lbgnnoutputvars, lbauxvars, lbnodecontent,
            ubgnnoutputvars, ubauxvars, ubnodecontent) );
      break;
   case GNN_LAYERTYPE_SAGE:
      sageinfo = SCIPgetGNNLayerinfoSage(gnndata, layeridx);
      SCIP_CALL( computeBoundsGNNSageLayer(scip, sageinfo, adjacencymatrix, ngraphnodes, globalbudget, localbudget,
            gnnoutputvars,adjacencyvars, lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars,
            ubauxvars, ubnodecontent, lbgnnoutputvarsprev, ubgnnoutputvarsprev, allocatearrays) );
      break;
   case GNN_LAYERTYPE_POOL:
      poolinfo = SCIPgetGNNLayerinfoPool(gnndata, layeridx);
      SCIP_CALL( computeBoundsGNNPoolLayer(scip, poolinfo, ngraphnodes,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent,
            lbgnnoutputvarsprev, ubgnnoutputvarsprev, allocatearrays) );
      break;
   case GNN_LAYERTYPE_DENSE:
      denseinfo = SCIPgetGNNLayerinfoDense(gnndata, layeridx);
      SCIP_CALL( computeBoundsGNNDenseLayer(scip, denseinfo,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent,
            lbgnnoutputvarsprev, ubgnnoutputvarsprev, allocatearrays) );
      break;
   default:
      assert(FALSE);
   }

   return SCIP_OKAY;
}

/** computes bounds for all variables and expressions in GNN for robust classification problems */
SCIP_RETCODE SCIPcomputeBoundsGNNRobustClassify(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   int                   globalbudget,       /**< global attack budget */
   int*                  localbudget,        /**< local attack budget per node */
   SCIP_VAR***           gnnoutputvars,      /**< output variables of all layers (or NULL) */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency in problem (or NULL) */
   SCIP_Real**           lbinput,            /**< lower bounds on input for GNN nodes (or NULL) */
   SCIP_Real**           ubinput,            /**< upper bounds on input for GNN nodes (or NULL) */
   SCIP_Real***          lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real***          lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real***          lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real***          ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real***          ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real***          ubnodecontent       /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   )
{
   SCIP_VAR** gnnoutvars;
   int nlayers;
   int l;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(adjacencymatrix != NULL);
   assert(ngraphnodes > 0);
   assert(0 <= globalbudget && globalbudget <= ngraphnodes);
   assert(localbudget != NULL);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);

   nlayers = SCIPgetGNNNLayers(gnndata);

   /* allocate memory for bounds for layers */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbgnnoutputvars, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbauxvars, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbnodecontent, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubgnnoutputvars, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubauxvars, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubnodecontent, nlayers) );

   /* compute bounds per layer */
   if( gnnoutputvars != NULL )
      gnnoutvars = gnnoutputvars[0];
   else
      gnnoutvars = NULL;
   SCIP_CALL( SCIPcomputeBoundsGNNRobustClassifyLayer(scip, gnndata, adjacencymatrix, ngraphnodes,
         globalbudget, localbudget, gnnoutvars, adjacencyvars, lbinput, ubinput, 0,
         &(*lbgnnoutputvars)[0], &(*lbauxvars)[0], &(*lbnodecontent)[0],
         &(*ubgnnoutputvars)[0], &(*ubauxvars)[0], &(*ubnodecontent)[0],
         NULL, NULL, NULL, NULL, TRUE) );
   for( l = 1; l < nlayers; ++l )
   {
      if( gnnoutputvars != NULL )
         gnnoutvars = gnnoutputvars[l];
      else
         gnnoutvars = NULL;
      SCIP_CALL( SCIPcomputeBoundsGNNRobustClassifyLayer(scip, gnndata, adjacencymatrix, ngraphnodes,
            globalbudget, localbudget, gnnoutvars, adjacencyvars, lbinput, ubinput, l,
            &(*lbgnnoutputvars)[l], &(*lbauxvars)[l], &(*lbnodecontent)[l],
            &(*ubgnnoutputvars)[l], &(*ubauxvars)[l], &(*ubnodecontent)[l],
            (*lbgnnoutputvars)[l-1], (*lbauxvars)[l-1], (*ubgnnoutputvars)[l-1], (*ubauxvars)[l-1], TRUE) );
   }

   return SCIP_OKAY;
}
