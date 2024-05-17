/**@file   gnn_bounds.cpp
 * @brief  functions to compute bounds on variables and expressions in GNNs
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "struct_gnn.h"
#include "type_gnn.h"
#include "gnn.h"
#include "gnn_bounds.h"

/** computes bounds for variables and expressions in an input layer for undirected graph */
static
SCIP_RETCODE computeBoundsGNNInputLayerUndirected(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_INPUT*  layerinfo,          /**< information about input layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
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
         if( lbinput != NULL )
            (*lbgnnoutputvars)[i] = MAX(0.0, lbinput[v][f]);
         else
            (*lbgnnoutputvars)[i] = 0.0;
         ++i;
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubgnnoutputvars, nentries) );
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         if( ubinput != NULL )
            (*ubgnnoutputvars)[i] = MAX(0.0, ubinput[v][f]);
         else
            (*ubgnnoutputvars)[i] = 1.0;
         ++i;
      }
   }

   /* lower and upper bounds on auxiliary variables */
   nentries *= ngraphnodes;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbauxvars, nentries) );
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            /* the auxiliary variable coincides with the gnnoutput variable if v = w */
            if( v == w )
            {
               int idx;
               idx = SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, v, f);
               (*lbauxvars)[i++] = (*lbgnnoutputvars)[idx];
            }
            else
               (*lbauxvars)[i++] = 0.0;
         }
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubauxvars, nentries) );
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            /* the auxiliary variable coincides with the gnnoutput variable if v = w */
            if( v == w )
            {
               int idx;
               idx = SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, v, f);
               (*ubauxvars)[i++] = (*ubgnnoutputvars)[idx];
            }
            else
               (*ubauxvars)[i++] = 1.0;
         }
      }
   }

   /* set data not needed */
   *lbnodecontent = NULL;
   *ubnodecontent = NULL;

   return SCIP_OKAY;
}

/** computes bounds for variables and expressions in an input layer for directed graph */
static
SCIP_RETCODE computeBoundsGNNInputLayerDirected(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_INPUT*  layerinfo,          /**< information about input layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
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
         if( lbinput != NULL )
            (*lbgnnoutputvars)[i] = MAX(0.0, lbinput[v][f]);
         else
            (*lbgnnoutputvars)[i] = 0.0;
         ++i;
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubgnnoutputvars, nentries) );
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         if( ubinput != NULL )
            (*ubgnnoutputvars)[i] = MAX(0.0, ubinput[v][f]);
         else
            (*ubgnnoutputvars)[i] = 1.0;
         ++i;
      }
   }

   /* lower and upper bounds on auxiliary variables */
   nentries *= ngraphnodes;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbauxvars, nentries) );
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            /* do not distinguish whether v = w since self-arcs could be vulnerable */
            (*lbauxvars)[i++] = 0.0;
         }
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubauxvars, nentries) );
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            /* do not distinguish whether v = w since self-arcs could be vulnerable */
            (*ubauxvars)[i++] = 1.0;
         }
      }
   }

   /* set data not needed */
   *lbnodecontent = NULL;
   *ubnodecontent = NULL;

   return SCIP_OKAY;
}

/** computes bounds for variables and expressions in an input layer */
static
SCIP_RETCODE computeBoundsGNNInputLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             isdirected,         /**< whether underlying  graph is directed */
   GNN_LAYERINFO_INPUT*  layerinfo,          /**< information about input layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
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

   if( isdirected )
   {
      SCIP_CALL( computeBoundsGNNInputLayerDirected(scip, layerinfo, ngraphnodes, lbinput, ubinput,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent) );
   }
   else
   {
      SCIP_CALL( computeBoundsGNNInputLayerUndirected(scip, layerinfo, ngraphnodes, lbinput, ubinput,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent) );
   }

   return SCIP_OKAY;
}

/** computes lower or upper bound on input for activation function at a node of a sage layer for undirected graph */
static
SCIP_Real computeBoundGNNSageLayerNodeUndirected(
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< information about sage layer */
   SCIP_Bool             computelb,          /**< whether a lower bound shall be computed */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   int                   sagegraphnodeidx,   /**< node index of sage layer for which bound shall be computed */
   int                   sagefeatureidx,     /**< feature index of sage layer for which bound shall be computed */
   SCIP_Real*            lbauxvarsprev,      /**< lower bounds on auxiliary variables from previous layer */
   SCIP_Real*            ubauxvarsprev       /**< upper bounds on auxiliary variables from previous layer */
)
{
   SCIP_Real bound;
   SCIP_Real weight;
#ifndef NDEBUG
   int noutputfeatures;
#endif
   int ninputfeatures;
   int auxvaridx;
   int f;
   int v;

   assert(layerinfo != NULL);
   assert(lbauxvarsprev != NULL);
   assert(ubauxvarsprev != NULL);

   ninputfeatures = SCIPgetNInputFeaturesSageLayer(layerinfo);
#ifndef NDEBUG
   noutputfeatures = SCIPgetNOutputFeaturesSageLayer(layerinfo);
   assert(0 <= sagefeatureidx && sagefeatureidx < noutputfeatures);
#endif
   assert(0 <= sagegraphnodeidx && sagegraphnodeidx < ngraphnodes);

   bound = SCIPgetSageLayerFeatureBias(layerinfo, sagefeatureidx);

   /* bound contribution via node weights */
   for( f = 0; f < ninputfeatures; ++f )
   {
      weight = SCIPgetSageLayerFeatureNodeweight(layerinfo, f, sagefeatureidx);

      /* get the index of the auxiliary variable for feature f from PREVIOUS layer */
      auxvaridx = SCIPgetAuxvarIdxLayer(ngraphnodes, ninputfeatures, sagegraphnodeidx, sagegraphnodeidx, f);
      if( computelb )
      {
         if( weight >= 0 )
            bound += weight * lbauxvarsprev[auxvaridx];
         else
            bound += weight * ubauxvarsprev[auxvaridx];
      }
      else
      {
         if( weight >= 0 )
            bound += weight * ubauxvarsprev[auxvaridx];
         else
            bound += weight * lbauxvarsprev[auxvaridx];
      }
   }

   /* bound contribution via edge weights */
   for( v = 0; v < ngraphnodes; ++v )
   {
      /* skip edges to node itself */
      if( v == sagegraphnodeidx )
         continue;

      for( f = 0; f < ninputfeatures; ++f )
      {
         weight = SCIPgetSageLayerFeatureEdgeweight(layerinfo, f, sagefeatureidx);

         /* get the index of the auxiliary variable for feature f from PREVIOUS layer */
         auxvaridx = SCIPgetAuxvarIdxLayer(ngraphnodes, ninputfeatures, sagegraphnodeidx, v, f);
         if( computelb )
         {
            if( weight >= 0 )
               bound += weight * lbauxvarsprev[auxvaridx];
            else
               bound += weight * ubauxvarsprev[auxvaridx];
         }
         else
         {
            if( weight >= 0 )
               bound += weight * ubauxvarsprev[auxvaridx];
            else
               bound += weight * lbauxvarsprev[auxvaridx];
         }
      }
   }

   return bound;
}

/** computes lower or upper bound on input for activation function at a node of a sage layer for directed graphs */
static
SCIP_Real computeBoundGNNSageLayerNodeDirected(
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< information about sage layer */
   SCIP_Bool             computelb,          /**< whether a lower bound shall be computed */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   int                   sagegraphnodeidx,   /**< node index of sage layer for which bound shall be computed */
   int                   sagefeatureidx,     /**< feature index of sage layer for which bound shall be computed */
   SCIP_Real*            lbauxvarsprev,      /**< lower bounds on auxiliary variables from previous layer */
   SCIP_Real*            ubauxvarsprev,      /**< upper bounds on auxiliary variables from previous layer */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            ubgnnoutputvarsprev /**< upper bounds on output at previous GNN layer */
)
{
   SCIP_Real bound;
   SCIP_Real weight;
#ifndef NDEBUG
   int noutputfeatures;
#endif
   int ninputfeatures;
   int auxvaridx;
   int gnnvaridx;
   int f;
   int v;

   assert(layerinfo != NULL);
   assert(lbauxvarsprev != NULL);
   assert(ubauxvarsprev != NULL);
   assert(lbgnnoutputvarsprev != NULL);
   assert(ubgnnoutputvarsprev != NULL);

   ninputfeatures = SCIPgetNInputFeaturesSageLayer(layerinfo);
#ifndef NDEBUG
   noutputfeatures = SCIPgetNOutputFeaturesSageLayer(layerinfo);
   assert(0 <= sagefeatureidx && sagefeatureidx < noutputfeatures);
#endif
   assert(0 <= sagegraphnodeidx && sagegraphnodeidx < ngraphnodes);

   bound = SCIPgetSageLayerFeatureBias(layerinfo, sagefeatureidx);

   /* bound contribution via node weights */
   for( f = 0; f < ninputfeatures; ++f )
   {
      weight = SCIPgetSageLayerFeatureNodeweight(layerinfo, f, sagefeatureidx);

      /* get the index of the gnnoutput variable for feature f from PREVIOUS layer */
      gnnvaridx = sagegraphnodeidx * ninputfeatures + f;
      if( computelb )
      {
         if( weight >= 0 )
            bound += weight * lbgnnoutputvarsprev[gnnvaridx];
         else
            bound += weight * ubgnnoutputvarsprev[gnnvaridx];
      }
      else
      {
         if( weight >= 0 )
            bound += weight * ubgnnoutputvarsprev[gnnvaridx];
         else
            bound += weight * lbgnnoutputvarsprev[gnnvaridx];
      }
   }

   /* bound contribution via edge weights */
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < ninputfeatures; ++f )
      {
         weight = SCIPgetSageLayerFeatureEdgeweight(layerinfo, f, sagefeatureidx);

         /* get the index of the auxiliary variable for feature f from PREVIOUS layer */
         auxvaridx = SCIPgetAuxvarIdxLayer(ngraphnodes, ninputfeatures, sagegraphnodeidx, v, f);
         if( computelb )
         {
            if( weight >= 0 )
               bound += weight * lbauxvarsprev[auxvaridx];
            else
               bound += weight * ubauxvarsprev[auxvaridx];
         }
         else
         {
            if( weight >= 0 )
               bound += weight * ubauxvarsprev[auxvaridx];
            else
               bound += weight * lbauxvarsprev[auxvaridx];
         }
      }
   }

   return bound;
}

/** computes bounds for variables and expressions in a sage layer for an undirected graph */
static
SCIP_RETCODE computeBoundsGNNSageLayerUndirected(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< information about sage layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbauxvarsprev,      /**< lower bounds on auxiliary variables from previous layer */
   SCIP_Real*            ubauxvarsprev       /**< upper bounds on auxiliary variables from previous layer */
   )
{
   GNN_ACTIVATIONTYPE activationtype;
   SCIP_Real bound;
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
   assert(lbauxvarsprev != NULL);
   assert(ubauxvarsprev != NULL);

   /* compute the number of nodes in sage layer */
   nfeatures = SCIPgetNOutputFeaturesSageLayer(layerinfo);
   nentries = nfeatures * ngraphnodes;
   activationtype = SCIPgetSageLayerActivationType(layerinfo);

   /*
    * allocate and populate bound arrays
    */

   /* input of activation functions */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbnodecontent, nentries) );
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         bound = computeBoundGNNSageLayerNodeUndirected(layerinfo, TRUE, ngraphnodes, v, f,
            lbauxvarsprev, ubauxvarsprev);
         (*lbnodecontent)[i++] = bound;
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubnodecontent, nentries) );
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         bound = computeBoundGNNSageLayerNodeUndirected(layerinfo, FALSE, ngraphnodes, v, f,
            lbauxvarsprev, ubauxvarsprev);
         (*ubnodecontent)[i++] = bound;
      }
   }

   /* output of activation functions */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbgnnoutputvars, nentries) );
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
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubgnnoutputvars, nentries) );
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
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbauxvars, nentries) );
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            int idx;
            idx = SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, w, f);

            /* the auxiliary variable coincides with the gnnoutput variable if v = w */
            if( v == w )
               (*lbauxvars)[i] = (*lbgnnoutputvars)[idx];
            else
               (*lbauxvars)[i] = MIN(0.0, (*lbgnnoutputvars)[idx]);
            ++i;
         }
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubauxvars, nentries) );
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            int idx;
            idx = SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, w, f);

            /* the auxiliary variable coincides with the gnnoutput variable if v = w */
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

/** computes bounds for variables and expressions in a sage layer for an directed graph */
static
SCIP_RETCODE computeBoundsGNNSageLayerDirected(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< information about sage layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbauxvarsprev,      /**< lower bounds on auxiliary variables from previous layer */
   SCIP_Real*            ubauxvarsprev,      /**< upper bounds on auxiliary variables from previous layer */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            ubgnnoutputvarsprev /**< upper bounds on output at previous GNN layer */
   )
{
   GNN_ACTIVATIONTYPE activationtype;
   SCIP_Real bound;
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
   assert(lbauxvarsprev != NULL);
   assert(ubauxvarsprev != NULL);
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
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbnodecontent, nentries) );
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         bound = computeBoundGNNSageLayerNodeDirected(layerinfo, TRUE, ngraphnodes, v, f,
            lbauxvarsprev, ubauxvarsprev, lbgnnoutputvarsprev, ubgnnoutputvarsprev);
         (*lbnodecontent)[i++] = bound;
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubnodecontent, nentries) );
   i = 0;
   for( v = 0; v < ngraphnodes; ++v )
   {
      for( f = 0; f < nfeatures; ++f )
      {
         bound = computeBoundGNNSageLayerNodeDirected(layerinfo, FALSE, ngraphnodes, v, f,
            lbauxvarsprev, ubauxvarsprev, lbgnnoutputvarsprev, ubgnnoutputvarsprev);
         (*ubnodecontent)[i++] = bound;
      }
   }

   /* output of activation functions */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbgnnoutputvars, nentries) );
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
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubgnnoutputvars, nentries) );
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
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbauxvars, nentries) );
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            int idx;
            idx = SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, w, f);

            /* do not distinguish if v = w since self-edges could be vulnerable */
            (*lbauxvars)[i++] = MIN(0.0, (*lbgnnoutputvars)[idx]);
         }
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubauxvars, nentries) );
   for( v = 0, i = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         for( f = 0; f < nfeatures; ++f )
         {
            int idx;
            idx = SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, w, f);

            /* do not distinguish if v = w since self-edges could be vulnerable */
            if( activationtype == GNN_ACTIVATIONTYPE_RELU )
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

/** computes bounds for variables and expressions in a sage layer */
static
SCIP_RETCODE computeBoundsGNNSageLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             isdirected,         /**< whether underlying  graph is directed */
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< information about sage layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbauxvarsprev,      /**< lower bounds on auxiliary variables from previous layer */
   SCIP_Real*            ubauxvarsprev,      /**< upper bounds on auxiliary variables from previous layer */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            ubgnnoutputvarsprev /**< upper bounds on output at previous GNN layer */
   )
{
   if( isdirected )
   {
      SCIP_CALL( computeBoundsGNNSageLayerDirected(scip, layerinfo, ngraphnodes,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent,
            lbauxvarsprev, ubauxvarsprev, lbgnnoutputvarsprev, ubgnnoutputvarsprev) );
   }
   else
   {
      SCIP_CALL( computeBoundsGNNSageLayerUndirected(scip, layerinfo, ngraphnodes,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent,
            lbauxvarsprev, ubauxvarsprev) );
   }

   return SCIP_OKAY;
}

/** computes bounds for variables and expressions in a pooling layer
 *
 *  @pre bounds on previous layers must have been computed and stored in bound arrays
 */
SCIP_RETCODE computeBoundsGNNPoolLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_POOL*   layerinfo,          /**< information about pooling layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            ubgnnoutputvarsprev /**< upper bounds on output at previous GNN layer */
   )
{
   int nfeatures;
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
   assert(lbgnnoutputvarsprev != NULL);
   assert(ubgnnoutputvarsprev != NULL);

   nfeatures = SCIPgetNOutputFeaturesPoolLayer(layerinfo);
   assert(SCIPgetTypePoolLayer(layerinfo) == GNN_POOLTYPE_ADD);
   assert(nfeatures == SCIPgetNInputFeaturesPoolLayer(layerinfo));

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbgnnoutputvars, nfeatures) );
   for( f = 0; f < nfeatures; ++f )
   {
      (*lbgnnoutputvars)[f] = 0.0;
      for( v = 0; v < ngraphnodes; ++v )
         (*lbgnnoutputvars)[f] += lbgnnoutputvarsprev[SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, v, f)];
   }

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubgnnoutputvars, nfeatures) );
   for( f = 0; f < nfeatures; ++f )
   {
      (*ubgnnoutputvars)[f] = 0.0;
      for( v = 0; v < ngraphnodes; ++v )
         (*ubgnnoutputvars)[f] += ubgnnoutputvarsprev[SCIPgetGNNNodevarIdxLayer(ngraphnodes, nfeatures, v, f)];
   }

   /* set data not needed */
   *lbnodecontent = NULL;
   *ubnodecontent = NULL;
   *lbauxvars = NULL;
   *ubauxvars = NULL;

   return SCIP_OKAY;
}

/** computes a lower or upper bound on the input for the activation function at a node of a dense layer */
static
SCIP_Real computeBoundGNNDenseLayerNode(
   GNN_LAYERINFO_DENSE*  layerinfo,          /**< information about dense layer */
   SCIP_Bool             computelb,          /**< whether a lower bound shall be computed */
   int                   featureidx,         /**< index of feature of dense layer for which bounds shall be computed */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            ubgnnoutputvarsprev /**< upper bounds on output at previous GNN layer */
   )
{
   SCIP_Real bound;
   SCIP_Real weight;
#ifndef NDEBUG
   int noutputfeatures;
#endif
   int ninputfeatures;
   int f;

   assert(layerinfo != NULL);
   assert(lbgnnoutputvarsprev != NULL);
   assert(ubgnnoutputvarsprev != NULL);

   ninputfeatures = SCIPgetNInputFeaturesDenseLayer(layerinfo);
#ifndef NDEBUG
   noutputfeatures = SCIPgetNOutputFeaturesDenseLayer(layerinfo);
   assert(0 <= featureidx && featureidx < noutputfeatures);
#endif

   bound = SCIPgetDenseLayerFeatureBias(layerinfo, featureidx);

   /* bound contribution via node weights */
   for( f = 0; f < ninputfeatures; ++f )
   {
      weight = SCIPgetDenseLayerFeatureWeight(layerinfo, f, featureidx);
      if( computelb )
      {
         if( weight >= 0 )
            bound += weight * lbgnnoutputvarsprev[f];
         else
            bound += weight * ubgnnoutputvarsprev[f];
      }
      else
      {
         if( weight >= 0 )
            bound += weight * ubgnnoutputvarsprev[f];
         else
            bound += weight * lbgnnoutputvarsprev[f];
      }
   }

   return bound;
}

/** computes bounds for variables and expressions in a dense layer
 *
 *  @pre bounds on previous layers must have been computed and stored in bound arrays
 */
SCIP_RETCODE computeBoundsGNNDenseLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_DENSE*  layerinfo,          /**< information about dense layer */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent,      /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbgnnoutputvarsprev, /**< lower bounds on output at previous GNN layer */
   SCIP_Real*            ubgnnoutputvarsprev /**< upper bounds on output at previous GNN layer */
   )
{
   GNN_ACTIVATIONTYPE activationtype;
   SCIP_Real bound;
   int nfeatures;
   int f;

   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);
   assert(lbgnnoutputvarsprev != NULL);
   assert(ubgnnoutputvarsprev != NULL);

   /* compute the number of nodes in sage layer */
   nfeatures = SCIPgetNOutputFeaturesDenseLayer(layerinfo);
   activationtype = SCIPgetDenseLayerActivationType(layerinfo);

   /*
    * allocate and populate bound arrays
    */

   /* input of activation functions */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbnodecontent, nfeatures) );
   for( f = 0; f < nfeatures; ++f )
   {
      bound = computeBoundGNNDenseLayerNode(layerinfo, TRUE, f, lbgnnoutputvarsprev, ubgnnoutputvarsprev);
      (*lbnodecontent)[f] = bound;
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubnodecontent, nfeatures) );
   for( f = 0; f < nfeatures; ++f )
   {
      bound = computeBoundGNNDenseLayerNode(layerinfo, FALSE, f, lbgnnoutputvarsprev, ubgnnoutputvarsprev);
      (*ubnodecontent)[f] = bound;
   }

   /* output of activation functions */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, lbgnnoutputvars, nfeatures) );
   for( f = 0; f < nfeatures; ++f )
   {
      bound = (*lbnodecontent)[f];
      switch( activationtype )
      {
      case GNN_ACTIVATIONTYPE_RELU:
         (*lbgnnoutputvars)[f] = MAX(0.0, bound);
         break;
      default:
         assert(activationtype == GNN_ACTIVATIONTYPE_NONE);
         (*lbgnnoutputvars)[f] = bound;
      }
   }
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, ubgnnoutputvars, nfeatures) );
   for( f = 0; f < nfeatures; ++f )
   {
      bound = (*ubnodecontent)[f];
      switch( activationtype )
      {
      case GNN_ACTIVATIONTYPE_RELU:
         (*ubgnnoutputvars)[f] = MAX(0.0, bound);
         break;
      default:
         assert(activationtype == GNN_ACTIVATIONTYPE_NONE);
         (*ubgnnoutputvars)[f] = bound;
      }
   }

   /* set data not needed */
   *lbauxvars = NULL;
   *ubauxvars = NULL;

   return SCIP_OKAY;
}

/** computes bounds for all variables and expressions in a layer of a GNN
 *
 *  @pre bounds on previous layers must have been computed and stored in bound arrays
 */
SCIP_RETCODE SCIPcomputeBoundsGNNLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             isdirected,         /**< whether underlying  graph is directed */
   GNN_DATA*             gnndata,            /**< data about GNN */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
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
   SCIP_Real*            ubauxvarsprev       /**< upper bounds on auxiliary variables at previous layer */
   )
{
   GNN_LAYERINFO_INPUT* inputinfo;
   GNN_LAYERINFO_SAGE* sageinfo;
   GNN_LAYERINFO_POOL* poolinfo;
   GNN_LAYERINFO_DENSE* denseinfo;
   GNN_LAYERTYPE layertype;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(ngraphnodes > 0);
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
      SCIP_CALL( computeBoundsGNNInputLayer(scip, isdirected, inputinfo, ngraphnodes, lbinput, ubinput,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent) );
      break;
   case GNN_LAYERTYPE_SAGE:
      sageinfo = SCIPgetGNNLayerinfoSage(gnndata, layeridx);
      SCIP_CALL( computeBoundsGNNSageLayer(scip, isdirected, sageinfo, ngraphnodes,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent,
            lbauxvarsprev, ubauxvarsprev, lbgnnoutputvarsprev, ubgnnoutputvarsprev) );
      break;
   case GNN_LAYERTYPE_POOL:
      poolinfo = SCIPgetGNNLayerinfoPool(gnndata, layeridx);
      SCIP_CALL( computeBoundsGNNPoolLayer(scip, poolinfo, ngraphnodes,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent,
            lbgnnoutputvarsprev, ubgnnoutputvarsprev) );
      break;
   case GNN_LAYERTYPE_DENSE:
      denseinfo = SCIPgetGNNLayerinfoDense(gnndata, layeridx);
      SCIP_CALL( computeBoundsGNNDenseLayer(scip, denseinfo,
            lbgnnoutputvars, lbauxvars, lbnodecontent, ubgnnoutputvars, ubauxvars, ubnodecontent,
            lbgnnoutputvarsprev, ubgnnoutputvarsprev) );
      break;
   default:
      assert(FALSE);
   }

   return SCIP_OKAY;
}

/** computes bounds for all variables and expressions in GNN */
SCIP_RETCODE SCIPcomputeBoundsGNN(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             isdirected,         /**< whether underlying  graph is directed */
   GNN_DATA*             gnndata,            /**< data about GNN */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
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
   int nlayers;
   int l;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(ngraphnodes > 0);
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
   SCIP_CALL( SCIPcomputeBoundsGNNLayer(scip, isdirected, gnndata, ngraphnodes, lbinput, ubinput, 0,
         &(*lbgnnoutputvars)[0], &(*lbauxvars)[0], &(*lbnodecontent)[0],
         &(*ubgnnoutputvars)[0], &(*ubauxvars)[0], &(*ubnodecontent)[0],
         NULL, NULL, NULL, NULL) );
   for( l = 1; l < nlayers; ++l )
   {
      SCIP_CALL( SCIPcomputeBoundsGNNLayer(scip, isdirected, gnndata, ngraphnodes, lbinput, ubinput, l,
            &(*lbgnnoutputvars)[l], &(*lbauxvars)[l], &(*lbnodecontent)[l],
            &(*ubgnnoutputvars)[l], &(*ubauxvars)[l], &(*ubnodecontent)[l],
            (*lbgnnoutputvars)[l-1], (*lbauxvars)[l-1], (*ubgnnoutputvars)[l-1], (*ubauxvars)[l-1]) );
   }

   return SCIP_OKAY;
}

/** frees bounds for variables and expressions in an input layer */
static
SCIP_RETCODE freeBoundsGNNInputLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_INPUT*  layerinfo,          /**< information about input layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
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
   int nfeatures;
   int nentries;

   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(ngraphnodes > 0);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);

   nfeatures = SCIPgetNInputFeaturesInputLayer(layerinfo);
   assert(nfeatures > 0);

   nentries = nfeatures * ngraphnodes;
   SCIPfreeBlockMemoryArray(scip, lbgnnoutputvars, nentries);
   SCIPfreeBlockMemoryArray(scip, ubgnnoutputvars, nentries);

   nentries *= ngraphnodes;
   if( *lbauxvars != NULL )
   {
      SCIPfreeBlockMemoryArray(scip, lbauxvars, nentries);
   }
   if( *ubauxvars != NULL )
   {
      SCIPfreeBlockMemoryArray(scip, ubauxvars, nentries);
   }

   assert(*lbnodecontent == NULL);
   assert(*ubnodecontent == NULL);

   return SCIP_OKAY;
}

/** frees bounds for variables and expressions in a sage layer */
static
SCIP_RETCODE freeBoundsGNNSageLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< information about sage layer */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
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
   int nfeatures;
   int nentries;

   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(ngraphnodes > 0);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);

   nfeatures = SCIPgetNOutputFeaturesSageLayer(layerinfo);
   assert(nfeatures > 0);

   nentries = nfeatures * ngraphnodes;
   SCIPfreeBlockMemoryArray(scip, lbgnnoutputvars, nentries);
   SCIPfreeBlockMemoryArray(scip, ubgnnoutputvars, nentries);
   SCIPfreeBlockMemoryArray(scip, lbnodecontent, nentries);
   SCIPfreeBlockMemoryArray(scip, ubnodecontent, nentries);

   nentries *= ngraphnodes;
   SCIPfreeBlockMemoryArray(scip, lbauxvars, nentries);
   SCIPfreeBlockMemoryArray(scip, ubauxvars, nentries);

   return SCIP_OKAY;
}

/** frees bounds for variables and expressions in a pooling layer */
static
SCIP_RETCODE freeBoundsGNNPoolLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_POOL*   layerinfo,          /**< information about pooling layer */
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
   int nfeatures;

   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);

   nfeatures = SCIPgetNOutputFeaturesPoolLayer(layerinfo);
   assert(nfeatures > 0);

   SCIPfreeBlockMemoryArray(scip, lbgnnoutputvars, nfeatures);
   SCIPfreeBlockMemoryArray(scip, ubgnnoutputvars, nfeatures);

   assert(*lbauxvars == NULL);
   assert(*ubauxvars == NULL);
   assert(*lbnodecontent == NULL);
   assert(*ubnodecontent == NULL);

   return SCIP_OKAY;
}

/** frees bounds for variables and expressions in a dense layer */
static
SCIP_RETCODE freeBoundsGNNDenseLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_LAYERINFO_DENSE*  layerinfo,          /**< information about dense layer */
   SCIP_Real**           lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real**           lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real**           lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real**           ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real**           ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real**           ubnodecontent       /**< pointer to array storing upperlower bounds on node content before
                                              *   applying an activation function */
   )
{
   int nfeatures;

   assert(scip != NULL);
   assert(layerinfo != NULL);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);

   nfeatures = SCIPgetNOutputFeaturesDenseLayer(layerinfo);
   assert(nfeatures > 0);

   SCIPfreeBlockMemoryArray(scip, lbgnnoutputvars, nfeatures);
   SCIPfreeBlockMemoryArray(scip, ubgnnoutputvars, nfeatures);
   SCIPfreeBlockMemoryArray(scip, lbnodecontent, nfeatures);
   SCIPfreeBlockMemoryArray(scip, ubnodecontent, nfeatures);

   assert(*lbauxvars == NULL);
   assert(*ubauxvars == NULL);

   return SCIP_OKAY;
}

/** frees arrays containing bounds for all variables and expressions in a layer of a GNN */
static
SCIP_RETCODE SCIPfreeBoundsGNNLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   int                   layeridx,           /**< index of layer for which bounds are computed */
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
   GNN_LAYERINFO_INPUT* inputinfo;
   GNN_LAYERINFO_SAGE* sageinfo;
   GNN_LAYERINFO_POOL* poolinfo;
   GNN_LAYERINFO_DENSE* denseinfo;
   GNN_LAYERTYPE layertype;

   layertype = SCIPgetGNNLayerType(gnndata, layeridx);
   assert((layeridx == 0) == (layertype == GNN_LAYERTYPE_INPUT));

   /* compute bounds from bounds on previous layer */
   switch( layertype )
   {
   case GNN_LAYERTYPE_INPUT:
      inputinfo = SCIPgetGNNLayerinfoInput(gnndata, layeridx);
      SCIP_CALL( freeBoundsGNNInputLayer(scip, inputinfo, ngraphnodes,
            lbgnnoutputvars, lbauxvars, lbnodecontent,
            ubgnnoutputvars, ubauxvars, ubnodecontent) );
      break;
   case GNN_LAYERTYPE_SAGE:
      sageinfo = SCIPgetGNNLayerinfoSage(gnndata, layeridx);
      SCIP_CALL( freeBoundsGNNSageLayer(scip, sageinfo, ngraphnodes,
            lbgnnoutputvars, lbauxvars, lbnodecontent,
            ubgnnoutputvars, ubauxvars, ubnodecontent) );
      break;
   case GNN_LAYERTYPE_POOL:
      poolinfo = SCIPgetGNNLayerinfoPool(gnndata, layeridx);
      SCIP_CALL( freeBoundsGNNPoolLayer(scip, poolinfo,
            lbgnnoutputvars, lbauxvars, lbnodecontent,
            ubgnnoutputvars, ubauxvars, ubnodecontent) );
      break;
   case GNN_LAYERTYPE_DENSE:
      denseinfo = SCIPgetGNNLayerinfoDense(gnndata, layeridx);
      SCIP_CALL( freeBoundsGNNDenseLayer(scip, denseinfo,
            lbgnnoutputvars, lbauxvars, lbnodecontent,
            ubgnnoutputvars, ubauxvars, ubnodecontent) );
      break;
   default:
      assert(FALSE);
   }

   return SCIP_OKAY;
}

/** frees arrays containing bounds for all variables and expressions in GNN */
SCIP_RETCODE SCIPfreeBoundsGNN(
   SCIP*                 scip,               /**< SCIP data structure */
   GNN_DATA*             gnndata,            /**< data about GNN */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   SCIP_Real***          lbgnnoutputvars,    /**< pointer to array storing lower bounds output at GNN nodes */
   SCIP_Real***          lbauxvars,          /**< pointer to array storing lower bound on auxiliary variables */
   SCIP_Real***          lbnodecontent,      /**< pointer to array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real***          ubgnnoutputvars,    /**< pointer to array storing upper bounds output at GNN nodes */
   SCIP_Real***          ubauxvars,          /**< pointer to array storing upper bound on auxiliary variables */
   SCIP_Real***          ubnodecontent       /**< pointer to array storing upper bounds on node content before
                                              *   applying an activation function */
   )
{
   int nlayers;
   int l;

   assert(scip != NULL);
   assert(gnndata != NULL);
   assert(ngraphnodes > 0);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);

   nlayers = SCIPgetGNNNLayers(gnndata);

   /* free the bounds for each layer */
   for( l = 0; l < nlayers; ++l )
   {
      SCIP_CALL( SCIPfreeBoundsGNNLayer(scip, gnndata, ngraphnodes, l,
            &(*lbgnnoutputvars)[l], &(*lbauxvars)[l], &(*lbnodecontent)[l],
            &(*ubgnnoutputvars)[l], &(*ubauxvars)[l], &(*ubnodecontent)[l]) );
   }

   /* free memory for bounds for layers */
   SCIPfreeBlockMemoryArray(scip, lbgnnoutputvars, nlayers);
   SCIPfreeBlockMemoryArray(scip, lbauxvars, nlayers);
   SCIPfreeBlockMemoryArray(scip, lbnodecontent, nlayers);
   SCIPfreeBlockMemoryArray(scip, ubgnnoutputvars, nlayers);
   SCIPfreeBlockMemoryArray(scip, ubauxvars, nlayers);
   SCIPfreeBlockMemoryArray(scip, ubnodecontent, nlayers);

   return SCIP_OKAY;
}

/** writes bounds on GNN variables to a file in Python dictionary format */
SCIP_RETCODE SCIPwriteBoundsPython(
   const char*           name,               /**< name of file to be created */
   GNN_DATA*             gnndata,            /**< data about GNN */
   GNNPROB_DATA*         gnnprobdata,        /**< data about optimization problem on GNN */
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
   FILE* fptr;
   int cnt;
   int l;
   int f;
   int v;
   int w;

   assert(gnndata != NULL);
   assert(lbgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(lbnodecontent != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(ubauxvars != NULL);
   assert(ubnodecontent != NULL);

   fptr = fopen(name, "w");

   fprintf(fptr, "def get_debug_bounds():\n");
   fprintf(fptr, "  bounds = dict()\n");
   fprintf(fptr, "  bounds['x'] = dict()\n");
   fprintf(fptr, "  bounds['aux'] = dict()\n");
   fprintf(fptr, "  bounds['relu_c'] = dict()\n");

   for( l = 0; l < gnndata->nlayers; ++l )
   {
      if( gnndata->layertypes[l] == GNN_LAYERTYPE_INPUT )
      {
         cnt = 0;
         for( v = 0; v < gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes; ++v)
         {
            for( f = 0; f < gnndata->layerinfo[l]->inputinfo.ninputfeatures; ++f)
            {
               fprintf(fptr, "  bounds['x'][%d,%d,'L%d_F%d'] = (%13.10f,%13.10f)\n",
                  v, l, l, f, lbgnnoutputvars[l][cnt], ubgnnoutputvars[l][cnt]);
               ++cnt;
            }
         }

         for( v = 0; v < gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes; ++v)
         {
            for( f = 0; f < gnndata->layerinfo[l]->inputinfo.ninputfeatures; ++f)
            {
               for( w = 0; w < gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes; ++w)
               {
                  fprintf(fptr, "  bounds['aux'][%d,%d,'L%d_F%d',%d] = (%13.10f,%13.10f)\n", v, l, l, f, w,
                     lbauxvars[l][SCIPgetGNNNodevarIdxLayer(gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes, gnndata->layerinfo[l]->inputinfo.ninputfeatures, v, f)],
                     ubauxvars[l][SCIPgetGNNNodevarIdxLayer(gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes, gnndata->layerinfo[l]->inputinfo.ninputfeatures, v, f)]);
               }
            }
         }
      }
      else if( gnndata->layertypes[l] == GNN_LAYERTYPE_SAGE )
      {
         cnt = 0;
         for( v = 0; v < gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes; ++v)
         {
            for( f = 0; f < gnndata->layerinfo[l]->sageinfo.noutputfeatures; ++f)
            {
               fprintf(fptr, "  bounds['x'][%d,%d,'L%d_F%d'] = (%13.10f,%13.10f)\n",
                  v, l, l, f, lbgnnoutputvars[l][cnt], ubgnnoutputvars[l][cnt]);
               fprintf(fptr, "  bounds['relu_c'][%d,%d,'L%d_F%d'] = (%13.10f,%13.10f)\n",
                  v, l, l, f, lbnodecontent[l][cnt], ubnodecontent[l][cnt]);
               ++cnt;
            }
         }

         for( v = 0; v < gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes; ++v)
         {
            for( f = 0; f < gnndata->layerinfo[l]->sageinfo.noutputfeatures; ++f)
            {
               for( w = 0; w < gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes; ++w)
               {
                  fprintf(fptr, "  bounds['aux'][%d,%d,'L%d_F%d',%d] = (%13.10f,%13.10f)\n", v, l, l, f, w,
                     lbauxvars[l][SCIPgetGNNNodevarIdxLayer(gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes, gnndata->layerinfo[l]->sageinfo.noutputfeatures, v, f)],
                     ubauxvars[l][SCIPgetGNNNodevarIdxLayer(gnnprobdata->probtypeinfo->robustclassifyinfo.nnodes, gnndata->layerinfo[l]->sageinfo.noutputfeatures, v, f)]);
               }
            }
         }
      }
      else if ( gnndata->layertypes[l] == GNN_LAYERTYPE_POOL )
      {
         cnt = 0;
         for( f = 0; f < gnndata->layerinfo[l]->poolinfo.noutputfeatures; ++f)
         {
            fprintf(fptr, "  bounds['x'][%d,'L%d_F%d'] = (%13.10f,%13.10f)\n",
               l, l, f, lbgnnoutputvars[l][cnt], ubgnnoutputvars[l][cnt]);
            ++cnt;
         }
      }
      else
      {
         cnt = 0;
         for( f = 0; f < gnndata->layerinfo[l]->denseinfo.noutputfeatures; ++f)
         {
            fprintf(fptr, "  bounds['x'][%d,'L%d_F%d'] = (%13.10f,%13.10f)\n",
               l, l, f, lbgnnoutputvars[l][cnt], ubgnnoutputvars[l][cnt]);
            fprintf(fptr, "  bounds['relu_c'][%d,'L%d_F%d'] = (%13.10f,%13.10f)\n",
               l, l, f, lbnodecontent[l][cnt], ubnodecontent[l][cnt]);
            ++cnt;
         }
      }
   }

   fprintf(fptr, "  return bounds\n\n");
   fclose(fptr);

   return SCIP_OKAY;
}
