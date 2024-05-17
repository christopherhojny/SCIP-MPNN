/**@file   sepa_relu_nodeclassify.c
 * @brief  separator for strengthened ReLU linearization cuts for node classification problems
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "gnn.h"
#include "gnn_bounds_nodeclassify.h"
#include "probdata_nodeclassify.h"
#include "sepa_relu_nodeclassify.h"


#define SEPA_NAME              "ReLUnode"
#define SEPA_DESC              "separator for strengthened ReLU linearization cuts for node classification problems"
#define SEPA_PRIORITY                 0
#define SEPA_FREQ                     1
#define SEPA_MAXBOUNDDIST           1.0
#define SEPA_USESSUBSCIP          FALSE /**< does the separator use a secondary SCIP instance? */
#define SEPA_DELAY                FALSE /**< should separation method be delayed, if other separators found cuts? */

#define DEFAULT_SEPALINAUXCUT      TRUE /**< whether cuts linearizing auxiliary variables should be separated */
#define DEFAULT_SEPARELUCUT        TRUE /**< whether cuts linearizing ReLU expressions should be separated */
#define DEFAULT_SEPADEPTH            -1 /**< up to which depth of the tree cuts are separated (-1: unlimited) */
#define DEFAULT_SEPARATEDENSE      TRUE /**< whether dense layers shall be separated */
#define DEFAULT_MAXNCUTS             -1 /**< maximum cuts to be separated per separation round (-1: unlimited) */

/*
 * Data structures
 */

/** separator data */
struct SCIP_SepaData
{
   /* information regarding GNN */
   GNN_DATA*             gnndata;            /**< data about GNN */

   /* information regarding underlying graph */
   SCIP_Bool**           adjacencymatrix;    /**< adjacency matrix of underlying graph */
   int                   ngraphnodes;        /**< number of nodes of underlying graph */
   int                   globalbudget;       /**< global attack budget on graph */
   int                   localbudget;        /**< local attack budget for each node in graph */
   SCIP_VAR**            adjacencyvars;      /**< variables modeling adjacency */

   /* variables used in ReLU constraints */
   SCIP_VAR***           gnnoutputvars;      /**< variables modeling output values at nodes of GNN per layer*/
   SCIP_VAR***           auxvars;            /**< variables to linearize products of variables per layer */
   SCIP_VAR***           isactivevars;       /**< variables modeling whether ReLU is active per layer */

   /* parameters */
   SCIP_Bool             sepalinauxcut;      /**< whether cuts linearizing auxiliary variables should be separated */
   SCIP_Bool             separelucut;        /**< whether cuts linearizing ReLU expressions should be separated */
   int                   sepadepth;          /**< up to which depth of the tree cuts are separated (-1: unlimited) */
   SCIP_Bool             separatedense;      /**< whether dense layers shall be separated */
   int                   maxncuts;           /**< maximum cuts to be separated per separation round (-1: unlimited) */

   /* statistics */
   int                   nlocalcuts;         /**< counter to store number of separated local cuts */
   int                   nglobalcuts;        /**< counter to store number of separated global cuts */
};


/*
 * Local methods
 */

/** returns index of edge in list of edges */
static
int getEdgeIdx(
   int                   u,                  /**< first node in edge */
   int                   v,                  /**< second node in edge */
   int                   nnodes              /**< number of nodes in complete graph */
   )
{
   assert(0 <= u && u < nnodes);
   assert(0 <= v && v < nnodes);

   return u*nnodes + v;
}

/** frees data of separator */
static
SCIP_RETCODE sepadataFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA**       sepadata            /**< pointer to sepadata */
   )
{
   assert(scip != NULL);
   assert(sepadata != NULL);

   SCIPfreeBlockMemory(scip, sepadata);

   return SCIP_OKAY;
}


/** creates data structure of separator */
static
SCIP_RETCODE sepadataCreate(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA**       sepadata            /**< pointer to store separator data */
   )
{
   assert(scip != NULL);
   assert(sepadata != NULL);

   SCIP_CALL( SCIPallocBlockMemory(scip, sepadata) );

   return SCIP_OKAY;
}

/** set data structure of separator */
static
SCIP_RETCODE sepadataSet(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< pointer to store separator data */
   GNN_DATA*             gnndata,            /**< GNN data */
   SCIP_Bool**           adjacencymatrix,    /**< adjacency matrix of underlying graph */
   int                   ngraphnodes,        /**< number of nodes of underlying graph */
   int                   globalbudget,       /**< global attack budget on graph */
   int                   localbudget,        /**< local attack budget for each node in graph */
   SCIP_VAR***           gnnoutputvars,      /**< variables modeling output values at nodes of GNN per layer*/
   SCIP_VAR***           auxvars,            /**< variables to linearize products of variables per layer */
   SCIP_VAR***           isactivevars,       /**< variables modeling whether ReLU is active per layer */
   SCIP_VAR**            adjacencyvars       /**< variables modeling adjacency */
   )
{
   assert(scip != NULL);
   assert(sepadata != NULL);
   assert(gnndata != NULL);
   assert(adjacencymatrix != NULL);
   assert(ngraphnodes > 0);
   assert(globalbudget >= 0);
   assert(localbudget >= 0);
   assert(gnnoutputvars != NULL);
   assert(auxvars != NULL);
   assert(isactivevars != NULL);
   assert(adjacencyvars != NULL);

   sepadata->gnndata = gnndata;
   sepadata->adjacencymatrix = adjacencymatrix;
   sepadata->ngraphnodes = ngraphnodes;
   sepadata->globalbudget = globalbudget;
   sepadata->localbudget = localbudget;
   sepadata->gnnoutputvars = gnnoutputvars;
   sepadata->auxvars = auxvars;
   sepadata->isactivevars = isactivevars;
   sepadata->adjacencyvars = adjacencyvars;

   return SCIP_OKAY;
}

/** separates cuts for a sage layer */
static
SCIP_RETCODE separateCutsLayerSage(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_SOL*             sol,                /**< solution to be separated (or NULL for LP solution) */
   int                   lidx,               /**< index of layer for which cuts shall be generated */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< information about sage layer */
   SCIP_Real*            lbnodecontent,      /**< lower bounds on input to activation function */
   SCIP_Real*            ubnodecontent,      /**< upper bounds on input to activation function */
   SCIP_Real*            lbgnnoutputvars,    /**< lower bounds on output of activation function */
   SCIP_Real*            ubgnnoutputvars,    /**< upper bounds on output of activation function */
   SCIP_Real*            lbauxvars,          /**< lower bounds on auxiliary variables */
   SCIP_Real*            ubauxvars,          /**< upper bounds on auxiliary variables */
   SCIP_VAR**            auxvarsprev,        /**< auxiliary variables of previous layer */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency */
   SCIP_VAR**            gnnoutputvars,      /**< variables for output of ReLU in this layer */
   SCIP_VAR**            gnnoutputvarsprev,  /**< variables for output of ReLU in previous layer */
   SCIP_VAR**            auxvars,            /**< auxiliary variables of this layer */
   SCIP_VAR**            isactivevars,       /**< variables modeling activation of ReLU */
   int*                  ngen,               /**< pointer to store number of generated cuts */
   SCIP_Bool*            infeasible,         /**< pointer to store whether infeasibility is detected */
   SCIP_Bool             separelucut,        /**< whether cuts linearizing auxiliary variables should be separated */
   SCIP_Bool             sepalinauxcut,      /**< up to which depth of the tree cuts are separated (-1: unlimited) */
   int                   maxncuts,           /**< maximum cuts to be separated per separation round (-1: unlimited) */
   int*                  nlocalcuts,         /**< pointer to increment number of separated local cuts */
   int*                  nglobalcuts         /**< pointer to increment number of separated global cuts */
   )
{
   char cutname[SCIP_MAXSTRLEN];
   SCIP_Bool islocal;
   SCIP_ROW* cut;
   SCIP_VAR* var;
   SCIP_Real weight;
   SCIP_Real val;
   SCIP_Real viol;
   int ninputfeatures;
   int noutputfeatures;
   int edgeidx;
   int auxidx;
   int idx;
   int v;
   int w;
   int fout;
   int fin;

   assert(scip != NULL);
   assert(sepa != NULL);
   assert(lidx > 0);
   assert(ngraphnodes > 0);
   assert(layerinfo != NULL);
   assert(lbnodecontent != NULL);
   assert(ubnodecontent != NULL);
   assert(lbgnnoutputvars != NULL);
   assert(ubgnnoutputvars != NULL);
   assert(lbauxvars != NULL);
   assert(ubauxvars != NULL);
   assert(auxvarsprev != NULL || lidx == 1);
   assert(adjacencyvars != NULL);
   assert(gnnoutputvars != NULL);
   assert(isactivevars != NULL);
   assert(ngen != NULL);
   assert(infeasible != NULL);
   assert(nlocalcuts != NULL);
   assert(nglobalcuts != NULL);

   *ngen = 0;
   *infeasible = FALSE;

   /* collect information about layer */
   ninputfeatures = SCIPgetNInputFeaturesSageLayer(layerinfo);
   noutputfeatures = SCIPgetNOutputFeaturesSageLayer(layerinfo);
   islocal = SCIPgetDepth(scip) != 0;

   /* separate ReLU cuts */
   if( separelucut )
   {
      /* separate sparse ReLU cuts */
      for( v = 0; v < ngraphnodes; ++v )
      {
         for( fout = 0; fout < noutputfeatures; ++fout )
         {
            idx = v * noutputfeatures + fout;
            viol = SCIPgetSolVal(scip, sol, gnnoutputvars[idx]);
            viol -= ubnodecontent[idx] * SCIPgetSolVal(scip, sol, isactivevars[idx]);

            /* check whether sparse cut is violated */
            if( SCIPisEfficacious(scip, viol) )
            {
               (void) SCIPsnprintf(cutname, SCIP_MAXSTRLEN, "ReLUsage_sparse_%d_%d_%d_%" SCIP_LONGINT_FORMAT,
                  lidx, v, fout, SCIPgetNLPs(scip));

               /* create empty cut */
               SCIP_CALL( SCIPcreateEmptyRowSepa(scip, &cut, sepa, cutname, -SCIPinfinity(scip), 0.0, islocal, FALSE, TRUE) );

               /* cache the row extension and only flush them if the cut gets added */
               SCIP_CALL( SCIPcacheRowExtensions(scip, cut) );

               /* collect all non-zero coefficients */
               SCIP_CALL( SCIPaddVarToRow(scip, cut, gnnoutputvars[idx], 1.0) );
               SCIP_CALL( SCIPaddVarToRow(scip, cut, isactivevars[idx], -ubnodecontent[idx]) );

               /* flush all changes before adding the cut */
               SCIP_CALL( SCIPflushRowExtensions(scip, cut) );

               /* globally valid cuts are added to the pool, locally valid cuts to the sepastore */
               if( !islocal )
               {
                  SCIP_CALL( SCIPaddPoolCut(scip, cut) );
                  ++(*nglobalcuts);
               }
               else
               {
                  SCIP_CALL( SCIPaddRow(scip, cut, FALSE, infeasible) );
                  ++(*nlocalcuts);
               }

               if( *infeasible )
                  return SCIP_OKAY;
               *ngen += 1;

               if( *ngen >= maxncuts )
                  goto TERMINATE;
            }
         }
      }

      /* separate dense ReLU cuts */
      for( v = 0; v < ngraphnodes; ++v )
      {
         for( fout = 0; fout < noutputfeatures; ++fout )
         {
            idx = v * noutputfeatures + fout;

            viol = SCIPgetSolVal(scip, sol, gnnoutputvars[idx]);
            viol -= SCIPgetSageLayerFeatureBias(layerinfo, fout) + lbnodecontent[idx];
            viol -= SCIPgetSolVal(scip, sol, isactivevars[idx]) * lbnodecontent[idx];

            /* get contibution of output variables via node weights */
            for( fin = 0; fin < ninputfeatures; ++fin )
            {
               idx = v * ninputfeatures + fin;
               weight = SCIPgetSageLayerFeatureNodeweight(layerinfo, fin, fout);
               viol -= SCIPgetSolVal(scip, sol, gnnoutputvarsprev[idx]) * weight;
            }

            /* get contibution of edges incl. self-loops */
            for( w = 0; w < ngraphnodes; ++w )
            {
               for( fin = 0; fin < ninputfeatures; ++fin )
               {
                  /* distinguish whether we are in the first layer, where no aux. variables are present  */
                  if( lidx == 1 )
                  {
                     idx = w * ninputfeatures + fin;
                     weight = SCIPgetSageLayerFeatureEdgeweight(layerinfo, fin, fout);
                     weight *= SCIPgetSolVal(scip, sol, gnnoutputvarsprev[idx]);
                     viol -= SCIPgetSolVal(scip, sol, adjacencyvars[getEdgeIdx(w, v, ngraphnodes)]) * weight;
                  }
                  else
                  {
                     auxidx = SCIPgetAuxvarIdxLayer(ngraphnodes, ninputfeatures, v, w, fin);
                     weight = SCIPgetSageLayerFeatureEdgeweight(layerinfo, fin, fout);
                     viol -= weight * SCIPgetSolVal(scip, sol, auxvarsprev[auxidx]);
                  }
               }
            }

            /* check whether dense cut is violated */
            if( SCIPisEfficacious(scip, viol) )
            {
               SCIP_Real rhs;

               idx = v * noutputfeatures + fout;
               rhs = SCIPgetSageLayerFeatureBias(layerinfo, fout) - lbnodecontent[idx];

               (void) SCIPsnprintf(cutname, SCIP_MAXSTRLEN, "ReLUsage_dense_%d_%d_%d_%" SCIP_LONGINT_FORMAT,
                  lidx, v, fout, SCIPgetNLPs(scip));

               /* create empty cut */
               SCIP_CALL( SCIPcreateEmptyRowSepa(scip, &cut, sepa, cutname, -SCIPinfinity(scip), rhs, islocal, FALSE, TRUE) );

               /* cache the row extension and only flush them if the cut gets added */
               SCIP_CALL( SCIPcacheRowExtensions(scip, cut) );

               /* add output variable and isactive variable */
               SCIP_CALL( SCIPaddVarToRow(scip, cut, gnnoutputvars[idx], 1.0) );
               SCIP_CALL( SCIPaddVarToRow(scip, cut, isactivevars[idx], -lbnodecontent[idx]) );

               /* collect contribution of node weights */
               for( fin = 0; fin < ninputfeatures; ++fin )
               {
                  idx = v * ninputfeatures + fin;
                  var = gnnoutputvarsprev[idx];
                  val = -SCIPgetSageLayerFeatureNodeweight(layerinfo, fin, fout);
                  SCIP_CALL( SCIPaddVarToRow(scip, cut, var, val) );
               }

               /* collect contribution of edge weights including self-loops */
               for( w = 0; w < ngraphnodes; ++w )
               {
                  for( fin = 0; fin < ninputfeatures; ++fin )
                  {
                     /* distinguish whether we are in the first layer */
                     if( lidx == 1 )
                     {
                        idx = w * ninputfeatures + fin;
                        var = adjacencyvars[getEdgeIdx(w, v, ngraphnodes)];
                        val = -SCIPgetSageLayerFeatureEdgeweight(layerinfo, fin, fout);
                        val *= SCIPgetSolVal(scip, sol, gnnoutputvarsprev[idx]);
                        SCIP_CALL( SCIPaddVarToRow(scip, cut, var, val) );
                     }
                     else
                     {
                        auxidx = SCIPgetAuxvarIdxLayer(ngraphnodes, ninputfeatures, v, w, fin);
                        var = auxvarsprev[auxidx];
                        val = -SCIPgetSageLayerFeatureEdgeweight(layerinfo, fin, fout);
                        SCIP_CALL( SCIPaddVarToRow(scip, cut, var, val) );
                     }
                  }
               }

               /* flush all changes before adding the cut */
               SCIP_CALL( SCIPflushRowExtensions(scip, cut) );

               /* globally valid cuts are added to the pool, locally valid cuts to the sepastore */
               if( !islocal )
               {
                  SCIP_CALL( SCIPaddPoolCut(scip, cut) );
                  ++(*nglobalcuts);
               }
               else
               {
                  SCIP_CALL( SCIPaddRow(scip, cut, FALSE, infeasible) );
                  ++(*nlocalcuts);
               }

               if( *infeasible )
                  return SCIP_OKAY;
               *ngen += 1;

               if( *ngen >= maxncuts )
                  goto TERMINATE;
            }
         }
      }
   }

   /* in the first and last sage layer, there are no auxiliary variables */
   if( auxvars == NULL || !sepalinauxcut )
      return SCIP_OKAY;

   for( v = 0; v < ngraphnodes; ++v )
   {
      for( w = 0; w < ngraphnodes; ++w )
      {
         /* get index of arc entering v */
         edgeidx = getEdgeIdx(w, v, ngraphnodes);

         for( fout = 0; fout < noutputfeatures; ++fout )
         {
            auxidx = SCIPgetAuxvarIdxLayer(ngraphnodes, noutputfeatures, v, w, fout);
            idx = w * noutputfeatures + fout;

            /* first type of cut */
            viol = SCIPgetSolVal(scip, sol, auxvars[auxidx]);
            viol -= SCIPgetSolVal(scip, sol, gnnoutputvars[idx]);
            viol += lbgnnoutputvars[idx] * (1 - SCIPgetSolVal(scip, sol, adjacencyvars[edgeidx]));

            if( SCIPisEfficacious(scip, viol) )
            {
               SCIP_Real rhs;

               rhs = -lbgnnoutputvars[idx];

               (void) SCIPsnprintf(cutname, SCIP_MAXSTRLEN, "aux_type1_%d_%d_%d_%d_%" SCIP_LONGINT_FORMAT,
                  lidx, v, w, fout, SCIPgetNLPs(scip));

               /* create empty cut */
               SCIP_CALL( SCIPcreateEmptyRowSepa(scip, &cut, sepa, cutname, -SCIPinfinity(scip), rhs,
                     islocal, FALSE, TRUE) );

               /* cache the row extension and only flush them if the cut gets added */
               SCIP_CALL( SCIPcacheRowExtensions(scip, cut) );

               SCIP_CALL( SCIPaddVarToRow(scip, cut, auxvars[auxidx], 1.0) );
               SCIP_CALL( SCIPaddVarToRow(scip, cut, gnnoutputvars[idx], -1.0) );
               SCIP_CALL( SCIPaddVarToRow(scip, cut, adjacencyvars[edgeidx], -lbgnnoutputvars[idx]) );

               /* flush all changes before adding the cut */
               SCIP_CALL( SCIPflushRowExtensions(scip, cut) );

               /* globally valid cuts are added to the pool, locally valid cuts to the sepastore */
               if( !islocal )
               {
                  SCIP_CALL( SCIPaddPoolCut(scip, cut) );
                  ++(*nglobalcuts);
               }
               else
               {
                  SCIP_CALL( SCIPaddRow(scip, cut, FALSE, infeasible) );
                  ++(*nlocalcuts);
               }

               if( *infeasible )
                  return SCIP_OKAY;
               *ngen += 1;

               if( *ngen >= maxncuts )
                  goto TERMINATE;
            }

            /* second type of cut */
            viol = -SCIPgetSolVal(scip, sol, auxvars[auxidx]);
            viol += SCIPgetSolVal(scip, sol, gnnoutputvars[idx]);
            viol -= ubgnnoutputvars[idx] * (1 - SCIPgetSolVal(scip, sol, adjacencyvars[edgeidx]));

            if( SCIPisEfficacious(scip, viol) )
            {
               SCIP_Real rhs;

               rhs = ubgnnoutputvars[idx];

               (void) SCIPsnprintf(cutname, SCIP_MAXSTRLEN, "aux_type2_%d_%d_%d_%d_%" SCIP_LONGINT_FORMAT,
                  lidx, v, w, fout, SCIPgetNLPs(scip));

               /* create empty cut */
               SCIP_CALL( SCIPcreateEmptyRowSepa(scip, &cut, sepa, cutname, -SCIPinfinity(scip), rhs,
                     islocal, FALSE, TRUE) );

               /* cache the row extension and only flush them if the cut gets added */
               SCIP_CALL( SCIPcacheRowExtensions(scip, cut) );

               SCIP_CALL( SCIPaddVarToRow(scip, cut, auxvars[auxidx], -1.0) );
               SCIP_CALL( SCIPaddVarToRow(scip, cut, gnnoutputvars[idx], 1.0) );
               SCIP_CALL( SCIPaddVarToRow(scip, cut, adjacencyvars[edgeidx], ubgnnoutputvars[idx]) );

               /* flush all changes before adding the cut */
               SCIP_CALL( SCIPflushRowExtensions(scip, cut) );

               /* globally valid cuts are added to the pool, locally valid cuts to the sepastore */
               if( !islocal )
               {
                  SCIP_CALL( SCIPaddPoolCut(scip, cut) );
                  ++(*nglobalcuts);
               }
               else
               {
                  SCIP_CALL( SCIPaddRow(scip, cut, FALSE, infeasible) );
                  ++(*nlocalcuts);
               }

               if( *infeasible )
                  return SCIP_OKAY;
               *ngen += 1;

               if( *ngen >= maxncuts )
                  goto TERMINATE;
            }

            /* third type of cut */
            viol = SCIPgetSolVal(scip, sol, auxvars[auxidx]);
            viol -= ubauxvars[auxidx] * SCIPgetSolVal(scip, sol, adjacencyvars[edgeidx]);

            if( SCIPisEfficacious(scip, viol) )
            {
               SCIP_Real rhs;

               rhs = 0.0;

               (void) SCIPsnprintf(cutname, SCIP_MAXSTRLEN, "aux_type3_%d_%d_%d_%d_%" SCIP_LONGINT_FORMAT,
                  lidx, v, w, fout, SCIPgetNLPs(scip));

               /* create empty cut */
               SCIP_CALL( SCIPcreateEmptyRowSepa(scip, &cut, sepa, cutname, -SCIPinfinity(scip), rhs,
                     islocal, FALSE, TRUE) );

               /* cache the row extension and only flush them if the cut gets added */
               SCIP_CALL( SCIPcacheRowExtensions(scip, cut) );

               SCIP_CALL( SCIPaddVarToRow(scip, cut, auxvars[auxidx], 1.0) );
               SCIP_CALL( SCIPaddVarToRow(scip, cut, adjacencyvars[edgeidx], -ubauxvars[auxidx]) );

               /* flush all changes before adding the cut */
               SCIP_CALL( SCIPflushRowExtensions(scip, cut) );

               /* globally valid cuts are added to the pool, locally valid cuts to the sepastore */
               if( !islocal )
               {
                  SCIP_CALL( SCIPaddPoolCut(scip, cut) );
                  ++(*nglobalcuts);
               }
               else
               {
                  SCIP_CALL( SCIPaddRow(scip, cut, FALSE, infeasible) );
                  ++(*nlocalcuts);
               }

               if( *infeasible )
                  return SCIP_OKAY;
               *ngen += 1;

               if( *ngen >= maxncuts )
                  goto TERMINATE;
            }

            /* fourth type of cut */
            viol = -SCIPgetSolVal(scip, sol, auxvars[auxidx]);
            viol += lbauxvars[auxidx] * SCIPgetSolVal(scip, sol, adjacencyvars[edgeidx]);

            if( SCIPisEfficacious(scip, viol) )
            {
               SCIP_Real rhs;

               rhs = 0.0;

               (void) SCIPsnprintf(cutname, SCIP_MAXSTRLEN, "aux_type4_%d_%d_%d_%d_%" SCIP_LONGINT_FORMAT,
                  lidx, v, w, fout, SCIPgetNLPs(scip));

               /* create empty cut */
               SCIP_CALL( SCIPcreateEmptyRowSepa(scip, &cut, sepa, cutname, -SCIPinfinity(scip), rhs,
                     islocal, FALSE, TRUE) );

               /* cache the row extension and only flush them if the cut gets added */
               SCIP_CALL( SCIPcacheRowExtensions(scip, cut) );

               SCIP_CALL( SCIPaddVarToRow(scip, cut, auxvars[auxidx], -1.0) );
               SCIP_CALL( SCIPaddVarToRow(scip, cut, adjacencyvars[edgeidx], lbauxvars[auxidx]) );

               /* flush all changes before adding the cut */
               SCIP_CALL( SCIPflushRowExtensions(scip, cut) );

               /* globally valid cuts are added to the pool, locally valid cuts to the sepastore */
               if( !islocal )
               {
                  SCIP_CALL( SCIPaddPoolCut(scip, cut) );
                  ++(*nglobalcuts);
               }
               else
               {
                  SCIP_CALL( SCIPaddRow(scip, cut, FALSE, infeasible) );
                  ++(*nlocalcuts);
               }

               if( *infeasible )
                  return SCIP_OKAY;
               *ngen += 1;

               if( *ngen >= maxncuts )
                  goto TERMINATE;
            }
         }
      }
   }

 TERMINATE:

   return SCIP_OKAY;
}

/** separates cuts for a dense layer */
static
SCIP_RETCODE separateCutsLayerDense(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_SOL*             sol,                /**< solution to be separated (or NULL for LP solution) */
   int                   lidx,               /**< index of layer for which cuts shall be generated */
   GNN_LAYERINFO_DENSE*  layerinfo,          /**< information about sage layer */
   SCIP_Real*            lbnodecontent,      /**< lower bounds on input to activation function */
   SCIP_Real*            ubnodecontent,      /**< upper bounds on input to activation function */
   SCIP_VAR**            gnnoutputvarsprev,  /**< output variables of previous layer */
   SCIP_VAR**            gnnoutputvars,      /**< variables for output of ReLU in this layer */
   SCIP_VAR**            isactivevars,       /**< variables modeling activation of ReLU */
   int*                  ngen,               /**< pointer to store number of generated cuts */
   SCIP_Bool*            infeasible,         /**< pointer to store whether infeasibility is detected */
   int                   maxncuts,           /**< maximum cuts to be separated per separation round (-1: unlimited) */
   int*                  nlocalcuts,         /**< pointer to increment number of separated local cuts */
   int*                  nglobalcuts         /**< pointer to increment number of separated global cuts */
   )
{
   char cutname[SCIP_MAXSTRLEN];
   SCIP_Bool islocal;
   SCIP_ROW* cut;
   SCIP_VAR* var;
   SCIP_Real val;
   SCIP_Real viol;
   int ninputfeatures;
   int noutputfeatures;
   int fout;
   int fin;

   assert(scip != NULL);
   assert(sepa != NULL);
   assert(lidx > 0);
   assert(layerinfo != NULL);
   assert(lbnodecontent != NULL);
   assert(ubnodecontent != NULL);
   assert(gnnoutputvarsprev != NULL);
   assert(gnnoutputvars != NULL);
   assert(isactivevars != NULL);
   assert(ngen != NULL);
   assert(infeasible != NULL);
   assert(nglobalcuts != NULL);
   assert(nlocalcuts != NULL);

   *ngen = 0;
   *infeasible = FALSE;

   /* collect information about layer */
   ninputfeatures = SCIPgetNInputFeaturesDenseLayer(layerinfo);
   noutputfeatures = SCIPgetNOutputFeaturesDenseLayer(layerinfo);
   islocal = SCIPgetDepth(scip) != 0;

   /* separate sparse cuts */
   for( fout = 0; fout < noutputfeatures; ++fout )
   {
      viol = SCIPgetSolVal(scip, sol, gnnoutputvars[fout]);
      viol -= ubnodecontent[fout] * SCIPgetSolVal(scip, sol, isactivevars[fout]);

      /* check whether sparse cut is violated */
      if( SCIPisEfficacious(scip, viol) )
      {
         (void) SCIPsnprintf(cutname, SCIP_MAXSTRLEN, "ReLUdense_sparse_%d_%d_%" SCIP_LONGINT_FORMAT,
            lidx, fout, SCIPgetNLPs(scip));

         /* create empty cut */
         SCIP_CALL( SCIPcreateEmptyRowSepa(scip, &cut, sepa, cutname, -SCIPinfinity(scip), 0.0, islocal, FALSE, TRUE) );

         /* cache the row extension and only flush them if the cut gets added */
         SCIP_CALL( SCIPcacheRowExtensions(scip, cut) );

         /* collect all non-zero coefficients */
         SCIP_CALL( SCIPaddVarToRow(scip, cut, gnnoutputvars[fout], 1.0) );
         SCIP_CALL( SCIPaddVarToRow(scip, cut, isactivevars[fout], -ubnodecontent[fout]) );

         /* flush all changes before adding the cut */
         SCIP_CALL( SCIPflushRowExtensions(scip, cut) );

         /* globally valid cuts are added to the pool, locally valid cuts to the sepastore */
         if( !islocal )
         {
            SCIP_CALL( SCIPaddPoolCut(scip, cut) );
            ++(*nglobalcuts);
         }
         else
         {
            SCIP_CALL( SCIPaddRow(scip, cut, FALSE, infeasible) );
            ++(*nlocalcuts);
         }

         if( *infeasible )
            return SCIP_OKAY;
         *ngen += 1;

         if( *ngen >= maxncuts )
            goto TERMINATE;
      }
   }

   /* separate dense cuts */
   for( fout = 0; fout < noutputfeatures; ++fout )
   {
      viol = -SCIPgetDenseLayerFeatureBias(layerinfo, fout) + lbnodecontent[fout];

      /* get contibution of variables from previous layer */
      for( fin = 0; fin < ninputfeatures; ++fin )
         viol -= SCIPgetDenseLayerFeatureWeight(layerinfo, fin, fout) * SCIPgetSolVal(scip, sol, gnnoutputvarsprev[fin]);

      /* get contibution of variables from this layer */
      viol -= lbnodecontent[fout] * SCIPgetSolVal(scip, sol, isactivevars[fout]);
      viol += SCIPgetSolVal(scip, sol, gnnoutputvars[fout]);

      /* check whether dense cut is violated */
      if( SCIPisEfficacious(scip, viol) )
      {
         SCIP_Real rhs;

         rhs = SCIPgetDenseLayerFeatureBias(layerinfo, fout) - lbnodecontent[fout];

         (void) SCIPsnprintf(cutname, SCIP_MAXSTRLEN, "ReLUdense_dense_%d_%d_%" SCIP_LONGINT_FORMAT,
            lidx, fout, SCIPgetNLPs(scip));

         /* create empty cut */
         SCIP_CALL( SCIPcreateEmptyRowSepa(scip, &cut, sepa, cutname, -SCIPinfinity(scip), rhs, islocal, FALSE, TRUE) );

         /* cache the row extension and only flush them if the cut gets added */
         SCIP_CALL( SCIPcacheRowExtensions(scip, cut) );

         /* collect coefficients for variables of previous layer */
         for( fin = 0; fin < ninputfeatures; ++fin )
         {
            var = gnnoutputvarsprev[fin];
            val = -SCIPgetDenseLayerFeatureWeight(layerinfo, fin, fout);
            SCIP_CALL( SCIPaddVarToRow(scip, cut, var, val) );
         }

         /* also take isactivevar and gnnoutputvar into account */
         SCIP_CALL( SCIPaddVarToRow(scip, cut, isactivevars[fout], -lbnodecontent[fout]) );
         SCIP_CALL( SCIPaddVarToRow(scip, cut, gnnoutputvars[fout], 1.0) );

         /* flush all changes before adding the cut */
         SCIP_CALL( SCIPflushRowExtensions(scip, cut) );

         /* globally valid cuts are added to the pool, locally valid cuts to the sepastore */
         if( !islocal )
         {
            SCIP_CALL( SCIPaddPoolCut(scip, cut) );
            ++(*nglobalcuts);
         }
         else
         {
            SCIP_CALL( SCIPaddRow(scip, cut, FALSE, infeasible) );
            ++(*nlocalcuts);
         }

         if( *infeasible )
            return SCIP_OKAY;
         *ngen += 1;

         if( *ngen >= maxncuts )
            goto TERMINATE;
      }
   }

 TERMINATE:

   return SCIP_OKAY;
}

/** separates strengthened ReLU linearization cuts for a layer in a  robust classification problems */
static
SCIP_RETCODE separateCutsLayer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_SOL*             sol,                /**< solution to be separated (or NULL for LP solution) */
   GNN_DATA*             gnndata,            /**< data about GNN */
   int                   lidx,               /**< index of layer for which cuts shall be generated */
   int                   ngraphnodes,        /**< number of nodes in underlying graph */
   SCIP_VAR**            adjacencyvars,      /**< variables modeling adjacency */
   SCIP_VAR**            gnnoutputvars,      /**< output variables of layer */
   SCIP_VAR**            auxvars,            /**< auxiliary variables of this layer */
   SCIP_VAR**            isactivevars,       /**< variables modeling activity */
   SCIP_VAR**            gnnoutputvarsprev,  /**< output variables of previous layer */
   SCIP_VAR**            auxvarsprev,        /**< auxiliary variables of previous layer (or NULL) */
   SCIP_Real*            lbnodecontent,      /**< array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            ubnodecontent,      /**< array storing lower bounds on node content before
                                              *   applying an activation function */
   SCIP_Real*            lbgnnoutputvars,    /**< lower bounds on output of activation function */
   SCIP_Real*            ubgnnoutputvars,    /**< upper bounds on output of activation function */
   SCIP_Real*            lbauxvars,          /**< lower bounds on auxiliary variables */
   SCIP_Real*            ubauxvars,          /**< upper bounds on auxiliary variables */
   int*                  ngen,               /**< pointer to store number of generated cuts */
   SCIP_Bool*            infeasible,         /**< pointer to store whether infeasibility is detectted */
   SCIP_Bool             separelucut,        /**< whether cuts linearizing auxiliary variables should be separated */
   SCIP_Bool             sepalinauxcut,      /**< up to which depth of the tree cuts are separated (-1: unlimited) */
   SCIP_Bool             separatedense,      /**< whether dense layers shall be separated */
   int                   maxncuts,           /**< maximum cuts to be separated per separation round (-1: unlimited) */
   int*                  nlocalcuts,         /**< pointer to increment number of separated local cuts */
   int*                  nglobalcuts         /**< pointer to increment number of separated global cuts */
   )
{
   GNN_LAYERINFO_SAGE* sageinfo;
   GNN_LAYERINFO_DENSE* denseinfo;
   GNN_LAYERTYPE layertype;
   GNN_ACTIVATIONTYPE activation;

   assert(scip != NULL);
   assert(sepa != NULL);
   assert(gnndata != NULL);
   assert(lidx >= 1);
   assert(ngraphnodes >= 0);
   assert(ngen != NULL);
   assert(infeasible != NULL);

   layertype = SCIPgetGNNLayerType(gnndata, lidx);

   *ngen = 0;
   *infeasible = FALSE;

   /* ReLU constraints can only exist in sage or dense layers */
   if( layertype != GNN_LAYERTYPE_SAGE && layertype != GNN_LAYERTYPE_DENSE )
      return SCIP_OKAY;

   if( !separatedense && layertype == GNN_LAYERTYPE_DENSE )
      return SCIP_OKAY;

   if( layertype == GNN_LAYERTYPE_SAGE )
   {
      sageinfo = SCIPgetGNNLayerinfoSage(gnndata, lidx);
      activation = SCIPgetSageLayerActivationType(sageinfo);
   }
   else
   {
      assert(layertype == GNN_LAYERTYPE_DENSE);

      denseinfo = SCIPgetGNNLayerinfoDense(gnndata, lidx);
      activation = SCIPgetDenseLayerActivationType(denseinfo);
   }

   /* only separate cuts if there is a ReLU activation function */
   if( activation != GNN_ACTIVATIONTYPE_RELU )
      return SCIP_OKAY;

   if( layertype == GNN_LAYERTYPE_SAGE )
   {
      SCIP_CALL( separateCutsLayerSage(scip, sepa, sol, lidx, ngraphnodes, sageinfo, lbnodecontent, ubnodecontent,
            lbgnnoutputvars, ubgnnoutputvars, lbauxvars, ubauxvars, auxvarsprev, adjacencyvars, gnnoutputvars,
            gnnoutputvarsprev, auxvars, isactivevars, ngen, infeasible, separelucut, sepalinauxcut, maxncuts,
            nlocalcuts, nglobalcuts) );
   }
   else if( separelucut )
   {
      SCIP_CALL( separateCutsLayerDense(scip, sepa, sol, lidx, denseinfo, lbnodecontent, ubnodecontent,
            gnnoutputvarsprev, gnnoutputvars, isactivevars, ngen, infeasible, maxncuts, nlocalcuts, nglobalcuts) );
   }

   return SCIP_OKAY;
}

/** separates strengthened ReLU linearization cuts for robust classification problems */
static
SCIP_RETCODE separateCuts(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< separator */
   SCIP_SOL*             sol,                /**< solution to be separated (or NULL for LP solution) */
   int*                  ngen,               /**< pointer to store number of separated cuts */
   SCIP_Bool*            infeasible,         /**< pointer to store whether infeasibility is detected */
   SCIP_Bool*            hasrun              /**< pointer to store whether separation has been called */
   )
{
   SCIP_SEPADATA* sepadata;
   GNN_DATA* gnndata;
   SCIP_Real** lbgnnoutputvars;
   SCIP_Real** lbauxvars;
   SCIP_Real** lbnodecontent;
   SCIP_Real** ubgnnoutputvars;
   SCIP_Real** ubauxvars;
   SCIP_Real** ubnodecontent;
   int tmpmaxncuts = INT_MAX;
   int nlayers;
   int l;

   assert(scip != NULL);
   assert(sepa != NULL);
   assert(ngen != NULL);
   assert(infeasible != NULL);
   assert(hasrun != NULL);

   *ngen = 0;
   *infeasible = FALSE;
   *hasrun = FALSE;

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   if( sepadata->sepadepth != -1 && SCIPgetDepth(scip) > sepadata->sepadepth )
      return SCIP_OKAY;
   *hasrun = TRUE;

   gnndata = sepadata->gnndata;
   assert(gnndata != NULL);

   nlayers = SCIPgetGNNNLayers(gnndata);

   /* allocate memory for variable bounds, use block memory since it is filled by auxiliary function */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &lbgnnoutputvars, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &lbauxvars, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &lbnodecontent, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &ubgnnoutputvars, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &ubauxvars, nlayers) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &ubnodecontent, nlayers) );

   /* iterate over layers, compute bounds on variables, and separate cuts per layer */
   SCIP_CALL( SCIPcomputeBoundsGNNNodeClassifyLayer(scip, gnndata, sepadata->adjacencymatrix,
         sepadata->ngraphnodes, sepadata->globalbudget, sepadata->localbudget,
         sepadata->gnnoutputvars[0], sepadata->adjacencyvars, NULL, NULL, 0,
         &lbgnnoutputvars[0], &lbauxvars[0], &lbnodecontent[0],
         &ubgnnoutputvars[0], &ubauxvars[0], &ubnodecontent[0],
         NULL, NULL, NULL, NULL) );

   if( sepadata->maxncuts != -1 )
      tmpmaxncuts = sepadata->maxncuts;

   for( l = 1; l < nlayers; ++l )
   {
      int ngenlocal = 0;
      SCIP_Bool infeasiblelocal = FALSE;

      SCIP_CALL( SCIPcomputeBoundsGNNNodeClassifyLayer(scip, gnndata, sepadata->adjacencymatrix,
            sepadata->ngraphnodes, sepadata->globalbudget, sepadata->localbudget,
            sepadata->gnnoutputvars[l], sepadata->adjacencyvars, NULL, NULL, l,
            &lbgnnoutputvars[l], &lbauxvars[l], &lbnodecontent[l],
            &ubgnnoutputvars[l], &ubauxvars[l], &ubnodecontent[l],
            lbgnnoutputvars[l-1], lbauxvars[l-1], ubgnnoutputvars[l-1], ubauxvars[l-1]) );

      SCIP_CALL( separateCutsLayer(scip, sepa, sol, gnndata, l, sepadata->ngraphnodes,
            sepadata->adjacencyvars, sepadata->gnnoutputvars[l], sepadata->auxvars[l],
            sepadata->isactivevars[l], sepadata->gnnoutputvars[l-1], sepadata->auxvars[l-1],
            lbnodecontent[l], ubnodecontent[l], lbgnnoutputvars[l], ubgnnoutputvars[l],
            lbauxvars[l], ubauxvars[l], &ngenlocal, &infeasiblelocal, sepadata->separelucut,
            sepadata->sepalinauxcut, sepadata->separatedense, tmpmaxncuts, &sepadata->nlocalcuts,
            &sepadata->nglobalcuts) );

      if( infeasiblelocal )
      {
         *infeasible = TRUE;
         break;
      }
      if( sepadata->maxncuts != -1 )
         tmpmaxncuts -= ngenlocal;
      *ngen += ngenlocal;

      if( sepadata->maxncuts != -1 && *ngen >= sepadata->maxncuts )
         break;
   }

   /* free memory */
   SCIPfreeBlockMemoryArray(scip, &ubnodecontent, nlayers);
   SCIPfreeBlockMemoryArray(scip, &ubauxvars, nlayers);
   SCIPfreeBlockMemoryArray(scip, &ubgnnoutputvars, nlayers);
   SCIPfreeBlockMemoryArray(scip, &lbnodecontent, nlayers);
   SCIPfreeBlockMemoryArray(scip, &lbauxvars, nlayers);
   SCIPfreeBlockMemoryArray(scip, &lbgnnoutputvars, nlayers);

   return SCIP_OKAY;
}

/*
 * Callback methods of separator
 */

/** LP solution separation method of separator */
static
SCIP_DECL_SEPAEXECLP(sepaExeclpReLUNodeClassify)
{  /*lint --e{715}*/
   int ngen = 0;
   SCIP_Bool infeasible = FALSE;
   SCIP_Bool hasrun = FALSE;

   *result = SCIP_DIDNOTRUN;

   /* terminate if we are not allowed to produce local cuts */
   if( !allowlocal )
      return SCIP_OKAY;

   SCIP_CALL( separateCuts(scip, sepa, NULL, &ngen, &infeasible, &hasrun) );

   if( !hasrun )
      return SCIP_OKAY;
   *result = SCIP_DIDNOTFIND;

   if( infeasible )
      *result = SCIP_CUTOFF;
   else if( ngen > 0 )
      *result = SCIP_SEPARATED;

   return SCIP_OKAY;
}

/** arbitrary primal solution separation method of separator */
static
SCIP_DECL_SEPAEXECSOL(sepaExecsolReLUNodeClassify)
{  /*lint --e{715}*/
   int ngen = 0;
   SCIP_Bool infeasible = FALSE;
   SCIP_Bool hasrun = FALSE;

   *result = SCIP_DIDNOTRUN;

   /* terminate if we are not allowed to produce local cuts */
   if( !allowlocal )
      return SCIP_OKAY;

   SCIP_CALL( separateCuts(scip, sepa, sol, &ngen, &infeasible, &hasrun) );

   if( !hasrun )
      return SCIP_OKAY;
   *result = SCIP_DIDNOTFIND;

   if( infeasible )
      *result = SCIP_CUTOFF;
   else if( ngen > 0 )
      *result = SCIP_SEPARATED;

   return SCIP_OKAY;
}

/** initialization method of separator (called after problem was transformed) */
static
SCIP_DECL_SEPAINIT(sepaInitReLUNodeClassify)
{  /*lint --e{715}*/
   SCIP_PROBDATA* probdata;
   SCIP_SEPADATA* sepadata;
   GNN_DATA* gnndata;
   SCIP_Bool** adjacencymatrix;
   int nnodes;
   int globalbudget;
   int localbudget;
   SCIP_VAR*** gnnoutputvars;
   SCIP_VAR*** auxvars;
   SCIP_VAR*** isactivevars;
   SCIP_VAR** adjacencyvars;

   probdata = SCIPgetProbData(scip);
   assert(probdata != NULL);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   gnndata = SCIPgetProbdataNodeClassifyGNNNData(probdata);
   adjacencymatrix = SCIPgetProbdataNodeClassifyAdjacencyMatrix(probdata);
   nnodes = SCIPgetProbdataNodeClassifyNNodes(probdata);
   globalbudget = SCIPgetProbdataNodeClassifyGlobalBudget(probdata);
   localbudget = SCIPgetProbdataNodeClassifyLocalBudget(probdata);
   gnnoutputvars = SCIPgetProbdataNodeClassifyGNNOutputVars(probdata);
   auxvars = SCIPgetProbdataNodeClassifyAuxVars(probdata);
   isactivevars = SCIPgetProbdataNodeClassifyIsActiveVars(probdata);
   adjacencyvars = SCIPgetProbdataNodeClassifyAdjacencyVars(probdata);

   /* create separator data */
   SCIP_CALL( sepadataSet(scip, sepadata, gnndata, adjacencymatrix, nnodes,
         globalbudget, localbudget, gnnoutputvars, auxvars, isactivevars, adjacencyvars) );

   return SCIP_OKAY;
}

/** destructor of separator to free user data (called when SCIP is exiting) */
static
SCIP_DECL_SEPAFREE(sepaFreeReLUNodeClassify)
{  /*lint --e{715}*/
   SCIP_SEPADATA* sepadata;

   assert(scip != NULL);
   assert(sepa != NULL);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   SCIP_CALL( sepadataFree(scip, &sepadata) );

   return SCIP_OKAY;
}

/** deinitialization method of separator (called before transformed problem is freed) */
static
SCIP_DECL_SEPAEXIT(sepaExitReLUNodeClassify)
{  /*lint --e{715}*/
   SCIP_SEPADATA* sepadata;

   assert(scip != NULL);
   assert(sepa != NULL);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   /* print statistics */
   SCIPinfoMessage(scip, NULL, "Statistics ReLU Node Classification Separator\n");
   SCIPinfoMessage(scip, NULL, "number local cuts: %10d\n", sepadata->nlocalcuts);
   SCIPinfoMessage(scip, NULL, "number global cuts: %9d\n", sepadata->nglobalcuts);

   return SCIP_OKAY;
}

/*
 * separator specific interface methods
 */

/** creates the ReLU linearization separator and includes it in SCIP */
SCIP_RETCODE SCIPincludeSepaReLUNodeClassify(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_SEPADATA* sepadata;
   SCIP_SEPA* sepa;

   /* create separator data */
   sepadata = NULL;
   sepa = NULL;

   SCIP_CALL( sepadataCreate(scip, &sepadata) );
   sepadata->nglobalcuts = 0;
   sepadata->nlocalcuts = 0;

   SCIP_CALL( SCIPincludeSepaBasic(scip, &sepa, SEPA_NAME, SEPA_DESC, SEPA_PRIORITY, SEPA_FREQ, SEPA_MAXBOUNDDIST,
         SEPA_USESSUBSCIP, SEPA_DELAY,
         sepaExeclpReLUNodeClassify, sepaExecsolReLUNodeClassify,
         sepadata) );

   assert(sepa != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetSepaInit(scip, sepa, sepaInitReLUNodeClassify) );
   SCIP_CALL( SCIPsetSepaFree(scip, sepa, sepaFreeReLUNodeClassify) );
   SCIP_CALL( SCIPsetSepaExit(scip, sepa, sepaExitReLUNodeClassify) );

   /* define parameters specific to this separator */
   SCIP_CALL( SCIPaddBoolParam(scip, "separating/" SEPA_NAME "/sepalinauxcut",
         "whether cuts linearizing auxiliary variables should be separated",
         &sepadata->sepalinauxcut, TRUE, DEFAULT_SEPALINAUXCUT, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "separating/" SEPA_NAME "/separelucut",
         "whether cuts linearizing ReLU expressions should be separated",
         &sepadata->separelucut, TRUE, DEFAULT_SEPARELUCUT, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip, "separating/" SEPA_NAME "/sepadepth",
         "up to which depth of the tree cuts are separated (-1: unlimited) ",
         &sepadata->sepadepth, TRUE, DEFAULT_SEPADEPTH, -1, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "separating/" SEPA_NAME "/separatedense",
         "whether dense layers shall be separated",
         &sepadata->separatedense, TRUE, DEFAULT_SEPARATEDENSE, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip, "separating/" SEPA_NAME "/maxncuts",
         "maximum cuts to be separated per separation round (-1: unlimited)",
         &sepadata->maxncuts, TRUE, DEFAULT_MAXNCUTS, -1, INT_MAX, NULL, NULL) );

   return SCIP_OKAY;
}
