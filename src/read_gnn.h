/**@file   read_gnn.h
 * @brief  functions to read GNN problems
 * @author Christopher Hojny
 */

// -*- C++ -*-

#ifndef READ_GNN_H
#define READ_GNN_H

#include <string>
#include "struct_gnn.h"
#include "struct_problem.h"

/** reads a GNN from a file and stores its data */
extern
SCIP_RETCODE readGNN(
   SCIP*                 scip,               //!< SCIP data structure
   std::string           filename,           //!< name of file encoding GNN
   GNN_DATA**            gnndata,            //!< pointer to GNN data
   SCIP_Bool*            success             //!< pointer to store whether GNN could be read
   );

/** frees GNN data */
extern
SCIP_RETCODE freeGNNData(
   SCIP*                 scip,               //!< SCIP pointer
   GNN_DATA*             gnndata             //!< pointer to data of GNN
   );

/** prints information about a GNN to screen */
extern
SCIP_RETCODE printGNN(
   SCIP*                 scip,               //!< SCIP pointer
   GNN_DATA*             gnndata             //!< data of GNN
   );

/** reads a GNN problem from a file and stores its data */
extern
SCIP_RETCODE readGNNProb(
   SCIP*                 scip,               //!< SCIP data structure
   std::string           filename,           //!< name of file encoding GNN
   GNNPROB_DATA**        gnnprobdata,        //!< pointer to GNN problem data
   SCIP_Bool*            success             //!< pointer to store whether GNN could be read
   );

/** frees GNN problem data */
extern
SCIP_RETCODE freeGNNProbData(
   SCIP*                 scip,               //!< SCIP pointer
   GNNPROB_DATA*         gnnprobdata         //!< pointer to GNN problem data
   );

#endif
