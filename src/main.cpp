/**@file   main.cpp
 * @brief  main file for SCIP-GNN
 * @author Christopher Hojny
 */

#include "parseOptions.h"
#include "read_gnn.h"
#include "scipgnn_plugins.h"
#include "scipgnn_params.h"
#include "struct_gnn.h"
#include "type_gnn.h"
#include "problem_gnn.h"
#include "gnn.h"

#include <scip/scip.h>
#include <scip/scip_debug.h>

#include <iostream>
#include <fstream>

/** creates and solves a GNN problem */
static
SCIP_RETCODE SCIPGNN(
   std::string           gnnfile,            //!< name of file encoding the GNN
   std::string           problemfile,        //!< file encoding the optimization problem over the GNN
   const char*           settings,           //!< Possible name of setting file
   double                timeLimit,          //!< time limit
   double                memLimit,           //!< memory limit
   SCIP_Longint          nodeLimit,          //!< node limit
   int                   displayFreq         //!< display frequency
   )
{  /*lint --e{429}*/

   // initialize SCIP
   SCIP* scip;
   SCIP_CALL( SCIPcreate(&scip) );

   // output SCIP banner
#ifdef SCIP_THREADSAFE_MESSAGEHDLRS
   SCIPprintVersion(scip, NULL);
#else
   SCIPprintVersion(NULL);
#endif
   std::cout << "\n" << std::endl;   // (to force flush)

   // create new params
   SCIP_CALL( createSCIPGNNParams(scip) );

   // create problem
   GNN_DATA* gnndata;
   GNNPROB_DATA* gnnprobdata;
   SCIP_Bool success;

   SCIP_CALL( readGNN(scip, gnnfile, &gnndata, &success) );
   SCIP_CALL( readGNNProb(scip, problemfile, &gnnprobdata, &success) );

   // get name of problem instance
   std::string instancename;
   size_t posdelim;
   posdelim = problemfile.rfind("/");
   if( posdelim < problemfile.npos )
      instancename = problemfile.substr(posdelim + 1, problemfile.npos - posdelim);
   else
      instancename = problemfile;

   char name[SCIP_MAXSTRLEN];
   strcpy(name, instancename.c_str());

   // load basic plugins
   SCIP_CALL( includeSCIPGNNPlugins(scip, SCIPgetGNNProbType(gnnprobdata)) );

   if( SCIPgetGNNProbType(gnnprobdata) == GNNPROB_TYPE_ROBUSTCLASSIFY )
   {
      SCIP_CALL( SCIPsetIntParam(scip, "separating/ReLUrc/expbackoff", 1) );
   }
   else
   {
      SCIP_CALL( SCIPsetIntParam(scip, "separating/ReLUnode/expbackoff", 1) );
   }

   if ( settings != 0 )
   {
      if ( ! SCIPfileExists(settings) )
      {
         SCIPerrorMessage("Setting file <%s> does not exist.\n\n", settings);
      }
      else
      {
         SCIPinfoMessage(scip, 0, "Reading parameters from <%s>.\n\n", settings);
         SCIP_CALL( SCIPreadParams(scip, settings) );
      }
   }

   /* output changed parameters */
   SCIPinfoMessage(scip, 0, "Changed settings:\n");
   SCIP_CALL( SCIPwriteParams(scip, 0, FALSE, TRUE) );
   SCIPinfoMessage(scip, 0, "\n");

   // set limits
   if ( timeLimit < 1e20 )
   {
      SCIPinfoMessage(scip, 0, "Setting time limit to %g.\n", timeLimit);
      SCIP_CALL( SCIPsetRealParam(scip, "limits/time", timeLimit) );
   }
   if ( memLimit < 1e20 )
   {
      SCIPinfoMessage(scip, 0, "Setting memory limit to %g.\n", memLimit);
      SCIP_CALL( SCIPsetRealParam(scip, "limits/memory", memLimit) );
   }
   if ( nodeLimit < SCIP_LONGINT_MAX )
   {
      SCIPinfoMessage(scip, 0, "Setting node limit to %lld.\n", nodeLimit);
      SCIP_CALL( SCIPsetLongintParam(scip, "limits/nodes", nodeLimit) );
   }
   if ( displayFreq < INT_MAX )
   {
      SCIPinfoMessage(scip, 0, "Setting display frequency to %d.\n", displayFreq);
      SCIP_CALL( SCIPsetIntParam(scip, "display/freq", displayFreq) );
   }
   SCIPinfoMessage(scip, 0, "\n");

   SCIP_CALL( SCIPcreateModel(scip, gnndata, gnnprobdata, name) );

   /* check whether we only write the model or whether we actually solve it */
   char* onlywrite;
   SCIP_CALL( SCIPgetStringParam(scip, "gnn/onlywritemodel", &onlywrite) );

   if( strcmp(onlywrite, "") != 0 )
   {
      SCIP_CALL( SCIPwriteOrigProblem(scip, onlywrite, "mps", FALSE) );
   }
   else
   {
      char* onlychecksol;
      SCIP_CALL( SCIPgetStringParam(scip, "gnn/onlychecksol", &onlychecksol) );

      if( strcmp(onlychecksol, "") != 0 )
      {
         SCIP_CALL( SCIPreadSol(scip, onlychecksol) );

         SCIP_CALL( SCIPsetIntParam(scip, "presolving/maxrounds", 0) );
         SCIP_CALL( SCIPpresolve(scip) );
      }
      else
      {
         // solve the problem ...
         SCIP_CALL( SCIPsolve(scip) );
         SCIP_CALL( SCIPprintStatistics(scip, NULL) );
         SCIP_CALL( SCIPprintBestSol(scip, NULL, FALSE) );
      }
   }

   // free data
   SCIP_CALL( freeGNNProbData(scip, gnnprobdata) );
   SCIP_CALL( freeGNNData(scip, gnndata) );

   SCIP_CALL( SCIPfreeTransform(scip) );

   BMScheckEmptyMemory();

   return SCIP_OKAY;
}


/** main function for creating and solving a GNN problem */
int main(int argc, const char** argv)
{
   // check parameters
   std::vector<std::string> mainArgs;
   std::vector<std::string> otherArgs;

   std::vector<std::string> knownOptions{"s", "t", "m", "n", "d"};

   if ( ! readOptions(argc, argv, 2, 2, mainArgs, otherArgs, knownOptions) )
   {
      std::cerr << "usage: " << argv[0] << " <GNN file> <problem file> [-s <settings>] [-t <time limit>] [-m <memory limit>] [-n <node limit>] ";
      std::cerr << "[-d <disp. freq>]" << std::endl;
      exit(1);
   }

   // get optional arguments
   const char* settings = 0;
   double timeLimit = 1e20;
   double memLimit = 1e20;
   SCIP_Longint nodeLimit = SCIP_LONGINT_MAX;
   int dispFreq = INT_MAX;
   for (long unsigned int j = 0; j < otherArgs.size(); ++j)
   {
      if ( otherArgs[j] == "s" )
         settings = otherArgs[++j].c_str();
      else if ( otherArgs[j] == "t" )
         timeLimit = atof(otherArgs[++j].c_str());
      else if ( otherArgs[j] == "m" )
         memLimit = atof(otherArgs[++j].c_str());
      else if ( otherArgs[j] == "n" )
         nodeLimit = atol(otherArgs[++j].c_str());
      else if ( otherArgs[j] == "d" )
         dispFreq = atoi(otherArgs[++j].c_str());
      else
         dispFreq = 1000;
   }

   // run code
   SCIP_RETCODE retcode;
   retcode = SCIPGNN(mainArgs[0], mainArgs[1], settings, timeLimit, memLimit, nodeLimit, dispFreq);

   if ( retcode != SCIP_OKAY )
   {
      SCIPprintError(retcode);
      return -1;
   }
   return 0;
}
