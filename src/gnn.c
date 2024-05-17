/**@file   gnn.c
 * @brief  functions to access data of GNN
 * @author Christopher Hojny
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scip/scip.h"
#include "gnn.h"
#include "type_gnn.h"
#include "type_problem.h"

#ifdef __cplusplus
extern "C" {
#endif

/** returns number of layers in GNN */
int SCIPgetGNNNLayers(
   GNN_DATA*             gnndata             /**< data of GNN */
   )
{
   assert(gnndata != NULL);
   return gnndata->nlayers;
}

/** returns type of a GNN layer */
GNN_LAYERTYPE SCIPgetGNNLayerType(
   GNN_DATA*             gnndata,            /**< data of GNN */
   int                   layeridx            /**< index of layer */
   )
{
   assert(gnndata != NULL);
   assert(0 <= layeridx && layeridx < gnndata->nlayers);
   return gnndata->layertypes[layeridx];
}

/** returns index of GNN node variable in a layer */
int SCIPgetGNNNodevarIdxLayer(
   int                   ngraphnodes,        /**< number of nodes in graph */
   int                   nfeatures,          /**< number of output features in layer */
   int                   nodeidx,            /**< node idx of GNN node variable */
   int                   featureidx          /**< index of feature of GNN node variable */
   )
{
   assert(0 <= nodeidx && nodeidx < ngraphnodes);
   assert(0 <= featureidx && featureidx < nfeatures);

   return nodeidx*nfeatures + featureidx;
}

/** returns index of auxiliary variable in a layer */
int SCIPgetAuxvarIdxLayer(
   int                   ngraphnodes,        /**< number of nodes in graph */
   int                   nfeatures,          /**< number of output features in layer */
   int                   nodeidx1,           /**< first node idx of GNN auxiliary variable */
   int                   nodeidx2,           /**< second node idx of GNN auxiliary variable */
   int                   featureidx          /**< index of feature of GNN node variable */
   )
{
   assert(0 <= nodeidx1 && nodeidx1 < ngraphnodes);
   assert(0 <= nodeidx2 && nodeidx2 < ngraphnodes);
   assert(0 <= featureidx && featureidx < nfeatures);

   return (nodeidx1*ngraphnodes + nodeidx2)*nfeatures + featureidx;
}

/** returns layer information of a GNN input layer */
GNN_LAYERINFO_INPUT* SCIPgetGNNLayerinfoInput(
   GNN_DATA*             gnndata,            /**< data of GNN */
   int                   layeridx            /**< index of layer */
   )
{
   assert(gnndata != 0);
   assert(0 <= layeridx && layeridx < gnndata->nlayers);
   assert(gnndata->layertypes[layeridx] == GNN_LAYERTYPE_INPUT);
   return &(gnndata->layerinfo[layeridx]->inputinfo);
}

/** returns layer information of a GNN sage layer */
GNN_LAYERINFO_SAGE* SCIPgetGNNLayerinfoSage(
   GNN_DATA*             gnndata,            /**< data of GNN */
   int                   layeridx            /**< index of layer */
   )
{
   assert(gnndata != 0);
   assert(0 <= layeridx && layeridx < gnndata->nlayers);
   assert(gnndata->layertypes[layeridx] == GNN_LAYERTYPE_SAGE);
   return &(gnndata->layerinfo[layeridx]->sageinfo);
}

/** returns layer information of a GNN pooling layer */
GNN_LAYERINFO_POOL* SCIPgetGNNLayerinfoPool(
   GNN_DATA*             gnndata,            /**< data of GNN */
   int                   layeridx            /**< index of layer */
   )
{
   assert(gnndata != 0);
   assert(0 <= layeridx && layeridx < gnndata->nlayers);
   assert(gnndata->layertypes[layeridx] == GNN_LAYERTYPE_POOL);
   return &(gnndata->layerinfo[layeridx]->poolinfo);
}

/** returns layer information of a GNN dense layer */
GNN_LAYERINFO_DENSE* SCIPgetGNNLayerinfoDense(
   GNN_DATA*             gnndata,            /**< data of GNN */
   int                   layeridx            /**< index of layer */
   )
{
   assert(gnndata != 0);
   assert(0 <= layeridx && layeridx < gnndata->nlayers);
   assert(gnndata->layertypes[layeridx] == GNN_LAYERTYPE_DENSE);
   return &(gnndata->layerinfo[layeridx]->denseinfo);
}

/** returns the number of features of an input layer */
int SCIPgetNInputFeaturesInputLayer(
   GNN_LAYERINFO_INPUT*  layerinfo           /**< data of input layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->ninputfeatures;
}

/** returns the number of input features of a sage layer */
int SCIPgetNInputFeaturesSageLayer(
   GNN_LAYERINFO_SAGE*   layerinfo           /**< data of sage layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->ninputfeatures;
}

/** returns the number of features of a sage layer */
int SCIPgetNOutputFeaturesSageLayer(
   GNN_LAYERINFO_SAGE*   layerinfo           /**< data of sage layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->noutputfeatures;
}

/** returns the activation type of a sage layer */
GNN_ACTIVATIONTYPE SCIPgetSageLayerActivationType(
   GNN_LAYERINFO_SAGE*   layerinfo           /**< data of sage layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->activation;
}

/** returns the bias of a feature of a sage layer */
SCIP_Real SCIPgetSageLayerFeatureBias(
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< data of sage layer */
   int                   featureidx          /**< index of feature for which bias is queried */
   )
{
   assert(layerinfo != NULL);
   assert(0 <= featureidx && featureidx < layerinfo->noutputfeatures);
   assert(layerinfo->bias != NULL);

   return layerinfo->bias[featureidx];
}

/** returns the node weight of a feature of a sage layer */
SCIP_Real SCIPgetSageLayerFeatureNodeweight(
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< data of sage layer */
   int                   prevfeatureidx,     /**< index of feature in previous layer for which weight is queried */
   int                   featureidx          /**< index of feature in this layer for which weight is queried */
   )
{
   assert(layerinfo != NULL);
   assert(0 <= featureidx && featureidx < layerinfo->noutputfeatures);
   assert(layerinfo->nodeweights != NULL);

   return layerinfo->nodeweights[featureidx][prevfeatureidx];
}

/** returns the edge weight of a feature of a sage layer */
SCIP_Real SCIPgetSageLayerFeatureEdgeweight(
   GNN_LAYERINFO_SAGE*   layerinfo,          /**< data of sage layer */
   int                   prevfeatureidx,     /**< index of feature in previous layer for which weight is queried */
   int                   featureidx          /**< index of feature in this layer for which weight is queried */
   )
{
   assert(layerinfo != NULL);
   assert(0 <= featureidx && featureidx < layerinfo->noutputfeatures);
   assert(layerinfo->edgeweights != NULL);

   return layerinfo->edgeweights[featureidx][prevfeatureidx];
}

/** returns the number of input features of a pooling layer */
int SCIPgetNInputFeaturesPoolLayer(
   GNN_LAYERINFO_POOL*   layerinfo           /**< data of pooling layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->ninputfeatures;
}

/** returns the number of features of a pooling layer */
int SCIPgetNOutputFeaturesPoolLayer(
   GNN_LAYERINFO_POOL*   layerinfo           /**< data of pooling layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->ninputfeatures;
}

/** returns the type of a pooling layer */
GNN_POOLTYPE SCIPgetTypePoolLayer(
   GNN_LAYERINFO_POOL*   layerinfo           /**< data of pool layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->type;
}

/** returns the bias of a feature of a dense layer */
SCIP_Real SCIPgetDenseLayerFeatureBias(
   GNN_LAYERINFO_DENSE*  layerinfo,          /**< data of sage layer */
   int                   featureidx          /**< index of feature for which bias is queried */
   )
{
   assert(layerinfo != NULL);
   assert(0 <= featureidx && featureidx < layerinfo->noutputfeatures);
   assert(layerinfo->bias != NULL);

   return layerinfo->bias[featureidx];

}

/** returns the weight of a feature of a dense layer */
SCIP_Real SCIPgetDenseLayerFeatureWeight(
   GNN_LAYERINFO_DENSE*  layerinfo,          /**< data of dense layer */
   int                   prevfeatureidx,     /**< index of feature in previous layer for which weight is queried */
   int                   featureidx          /**< index of feature in this layer for which weight is queried */
   )
{
   assert(layerinfo != NULL);
   assert(0 <= featureidx && featureidx < layerinfo->noutputfeatures);
   assert(layerinfo->weights != NULL);

   return layerinfo->weights[featureidx][prevfeatureidx];
}

/** returns the number of input features of a dense layer */
int SCIPgetNInputFeaturesDenseLayer(
   GNN_LAYERINFO_DENSE*  layerinfo           /**< data of dense layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->ninputfeatures;
}

/** returns the number of features of a dense layer */
int SCIPgetNOutputFeaturesDenseLayer(
   GNN_LAYERINFO_DENSE*  layerinfo           /**< data of dense layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->noutputfeatures;
}

/** returns the activation type of a dense layer */
GNN_ACTIVATIONTYPE SCIPgetDenseLayerActivationType(
   GNN_LAYERINFO_DENSE*  layerinfo           /**< data of dense layer */
   )
{
   assert(layerinfo != NULL);
   return layerinfo->activation;
}

/** returns type of problem on GNN */
GNNPROB_TYPE SCIPgetGNNProbType(
   GNNPROB_DATA*         gnnprobdata         /**< data of problem on GNN */
   )
{
   assert(gnnprobdata != NULL);
   return gnnprobdata->probtype;
}

/** returns information about robust classification problem on GNN */
GNNPROB_ROBUSTCLASSIFY* SCIPgetGNNProbDataRobustClassify(
   GNNPROB_DATA*         gnnprobdata         /**< data of problem on GNN */
   )
{
   assert(gnnprobdata != NULL);
   assert(gnnprobdata->probtype == GNNPROB_TYPE_ROBUSTCLASSIFY);
   assert(gnnprobdata->probtypeinfo != NULL);
   return &gnnprobdata->probtypeinfo->robustclassifyinfo;
}

/** returns number of nodes of graph in robust classification problem */
int SCIPgetNNodesRobustClassify(
   GNNPROB_ROBUSTCLASSIFY* probdata          /**< data of robust classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->nnodes;
}

/** returns number of features in robust classification problem */
int SCIPgetNFeaturesRobustClassify(
   GNNPROB_ROBUSTCLASSIFY* probdata          /**< data of robust classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->nfeatures;
}

/** returns global attack budget in robust classification problem */
int SCIPgetGlobalBudgetRobustClassify(
   GNNPROB_ROBUSTCLASSIFY* probdata          /**< data of robust classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->globalbudget;
}

/** returns local attack budget in robust classification problem */
int* SCIPgetLocalBudgetRobustClassify(
   GNNPROB_ROBUSTCLASSIFY* probdata          /**< data of robust classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->localbudget;
}

/** returns adjacency matrix of graph of in robust classification problem */
SCIP_Bool** SCIPgetAdjacencyMatrixRobustClassify(
   GNNPROB_ROBUSTCLASSIFY* probdata          /**< data of robust classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->adjacencymatrix;
}

/** returns lower bounds on feature assignment in robust classification problem */
SCIP_Real** SCIPgetFeatureLbRobustClassify(
   GNNPROB_ROBUSTCLASSIFY* probdata          /**< data of robust classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->featurelb;
}

/** returns upper bounds on feature assignment in robust classification problem */
SCIP_Real** SCIPgetFeatureUbRobustClassify(
   GNNPROB_ROBUSTCLASSIFY* probdata          /**< data of robust classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->featureub;
}

/** returns feature classification of graph in robust classification problem */
int SCIPgetGraphClassificationRobustClassify(
   GNNPROB_ROBUSTCLASSIFY* probdata          /**< data of robust classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->graphclassification;
}

/** returns targeted feature of graph in robust classification problem */
int SCIPgetTargetClassificationRobustClassify(
   GNNPROB_ROBUSTCLASSIFY* probdata          /**< data of robust classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->targetclassification;
}

/** returns information about node classification problem on GNN */
GNNPROB_NODECLASSIFY* SCIPgetGNNProbDataNodeClassify(
   GNNPROB_DATA*         gnnprobdata         /**< data of problem on GNN */
   )
{
   assert(gnnprobdata != NULL);
   assert(gnnprobdata->probtype == GNNPROB_TYPE_NODECLASSIFY);
   assert(gnnprobdata->probtypeinfo != NULL);
   return &gnnprobdata->probtypeinfo->nodeclassifyinfo;
}

/** returns number of nodes of graph in node classification problem */
int SCIPgetNNodesNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->nnodes;
}

/** returns number of features in node classification problem */
int SCIPgetNFeaturesNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->nfeatures;
}

/** returns global attack budget in node classification problem */
int SCIPgetGlobalBudgetNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->globalbudget;
}

/** returns local attack budget in node classification problem */
int SCIPgetLocalBudgetNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->localbudget;
}

/** returns adjacency matrix of graph of node classification problem */
SCIP_Bool** SCIPgetAdjacencyMatrixNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->adjacencymatrix;
}

/** returns lower bounds on feature assignment in node classification problem */
SCIP_Real** SCIPgetFeatureLbNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->featurelb;
}

/** returns upper bounds on feature assignment in node classification problem */
SCIP_Real** SCIPgetFeatureUbNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->featureub;
}

/** returns feature classification of graph in node classification problem */
int SCIPgetGraphClassificationNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->graphclassification;
}

/** returns targeted feature of graph in node classification problem */
int SCIPgetTargetClassificationNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->targetclassification;
}

/** returns index of node that shall be classified in node classification problem */
int SCIPgetNodeclassNodeClassify(
   GNNPROB_NODECLASSIFY* probdata          /**< data of node classification problem */
   )
{
   assert(probdata != NULL);
   return probdata->nodeclassification;
}
