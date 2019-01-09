/*--------------------------------------------------------------------------
  Author: Mengchi Zhang
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  zhan2308@purdue.edu
  
  initial version: 2017-07-19
  
  --------------------------------------------------------------------------*/

//-----------------------------------------------------------------------
/*!  \file generateMPI.cc

  \brief Contains functions to generate code for running the
  simulation with MPI. Part of the code generation section.
*/
//--------------------------------------------------------------------------

#include "generateMPI.h"

// Standard C++ includes
#include <fstream>

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void genHeader(const NNmodel &model,    //!< Model description
               const std::string &path,      //!< Path for code generationn
               int localHostID)         //!< ID of local host
{
    //=======================
    // generate mpi.h
    //=======================

    // this file contains helpful macros and is separated out so that it can also be used by other code that is compiled separately
    std::string name= model.getGeneratedCodePath(path, "mpi.h");
    std::ofstream fs;
    fs.open(name.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    // **TODO** implement
    //writeHeader(os);
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file mpi.h" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing MPI function definition." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    os << "#ifndef MPI_H" << std::endl;
    os << "#define MPI_H" << std::endl;
    os << std::endl;
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things to remote" << std::endl;
    os << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        os << "void push" << n.first << "CurrentSpikesToRemote(int remote);" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things from remote" << std::endl;
    os << std::endl;
    for(const auto &n : model.getRemoteNeuronGroups()) {
        if(n.second.hasOutputToHost(localHostID)) {
            os << "void pull" << n.first << "CurrentSpikesFromRemote(int remote);" << std::endl;
        }
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global spikes communication" << std::endl;
    os << std::endl;
    os << "void synchroniseMPI();" << std::endl;
    os << std::endl;

    os << "#endif" << std::endl;
    fs.close();
}

void genCode(const NNmodel &model,  //!< Model description
             const std::string &path,    //!< Path for code generationn
             int localHostID)       //!< ID of local host
{
    //=======================
    // generate mpi.cc
    //=======================
    std::string name= model.getGeneratedCodePath(path, "mpi.cc");
    std::ofstream fs;
    fs.open(name.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    // **TODO** implement somewhere
    //writeHeader(os);
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file mpi.cc" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing MPI infrastructure code." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << std::endl;

    os << "#include <mpi.h>" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying spikes to remote" << std::endl << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        // neuron spike variables
        os << "void push" << n.first << "CurrentSpikesToRemote(int remote)" << std::endl;
        os << CodeStream::OB(1050);
        os << "const int spikeCountTag = " << (hashString("glbSpkCnt" + n.first) & 0x7FFFFFFF) << ";" << std::endl;
        os << "const int spikeTag = " << (hashString("glbSpk" + n.first) & 0x7FFFFFFF) << ";" << std::endl;
        os << "MPI_Request req;" << std::endl;
        // If delay is required
        if(n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
            os << "MPI_Isend(glbSpkCnt" << n.first << " + spkQuePtr" << n.first << ", 1";
            os << ", MPI_UNSIGNED, remote, spikeCountTag, MPI_COMM_WORLD, &req);" << std::endl;
            os << "MPI_Isend(glbSpk" << n.first << " + (spkQuePtr" << n.first << " * " << n.second.getNumNeurons() << ")";
            os << ", glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "]";
            os << ", MPI_UNSIGNED, remote, spikeTag, MPI_COMM_WORLD, &req);" << std::endl;
        }
        else {
            os << "MPI_Isend(glbSpkCnt" << n.first << ", 1";
            os << ", MPI_UNSIGNED, remote, spikeCountTag, MPI_COMM_WORLD, &req);" << std::endl;
            os << "MPI_Isend(glbSpk" << n.first << ", glbSpkCnt" << n.first << "[0]";
            os << ", MPI_UNSIGNED, remote, spikeTag, MPI_COMM_WORLD, &req);" << std::endl;
        }
        os << CodeStream::CB(1050);
        os << std::endl;
    }

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying spikes from remote" << std::endl << std::endl;
    for(const auto &n : model.getRemoteNeuronGroups()) {
        if(n.second.hasOutputToHost(localHostID)) {
            // neuron spike variables
            os << "void pull" << n.first << "CurrentSpikesFromRemote(int remote)" << std::endl;
            os << CodeStream::OB(1051);
            os << "const int spikeCountTag = " << (hashString("glbSpkCnt" + n.first) & 0x7FFFFFFF) << ";" << std::endl;
            os << "const int spikeTag = " << (hashString("glbSpk" + n.first) & 0x7FFFFFFF) << ";" << std::endl;
            if(n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
                os << "MPI_Recv(glbSpkCnt" << n.first << " + spkQuePtr" << n.first << ", 1";
                os << ", MPI_UNSIGNED, remote, spikeCountTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);" << std::endl;
                os << "MPI_Recv(glbSpk" << n.first << " + (spkQuePtr" << n.first << " * " << n.second.getNumNeurons() << ")";
                os << ", glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "]";
                os << ", MPI_UNSIGNED, remote, spikeTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);" << std::endl;
            }
            else {
                os << "MPI_Recv(glbSpkCnt" << n.first << ", 1";
                os << ", MPI_UNSIGNED, remote, spikeCountTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);" << std::endl;
                os << "MPI_Recv(glbSpk" << n.first << ", glbSpkCnt" << n.first << "[0]";
                os << ", MPI_UNSIGNED, remote, spikeTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);" << std::endl;
            }
            os << CodeStream::CB(1051);
            os << std::endl;
        }
    }

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// communication function to sync spikes" << std::endl << std::endl;

    os << "void synchroniseMPI()" << std::endl;
    os << CodeStream::OB(1054) << std::endl;
    // Loop through local neuron groups
    for(const auto &n : model.getLocalNeuronGroups()) {
        // Build set of unique remote targets who should receive spikes from this population
        std::set<int> remoteTargetIDs;
        for(auto *s : n.second.getOutSyn()) {
            const int trgClusterHostID = s->getTrgNeuronGroup()->getClusterHostID();
            if (trgClusterHostID != localHostID) {
                remoteTargetIDs.insert(trgClusterHostID);
            }
        }

        // If there are any remote targets
        if(!remoteTargetIDs.empty()) {
            os << "// Local neuron group '" << n.first << "' - outgoing connections" << std::endl;

            // Send current spikes to each remote target
            for(int t : remoteTargetIDs) {
                os << "// send to remote node " << t << std::endl;
                os << "push" << n.first << "CurrentSpikesToRemote(" << t << ");" << std::endl;
            }
        }
    }

    // Loop through remote neuron groups
    for(const auto &n : model.getRemoteNeuronGroups()) {
        // If neuron group provides output to local host pull
        if(n.second.hasOutputToHost(localHostID)) {
            os << "// Remote neuron group '" << n.first << "'" << std::endl;
            os << "pull" << n.first << "CurrentSpikesFromRemote(" << n.second.getClusterHostID() << ");" << std::endl;
        }
    }
    os << CodeStream::CB(1054);
    os << std::endl;

    fs.close();
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
/*!
  \brief A function that generates predominantly MPI infrastructure code.

  In this function MPI infrastructure code are generated,
  including: MPI send and receive functions.
*/
//--------------------------------------------------------------------------
void genMPI(const NNmodel &model,   //!< Model description
            const std::string &path,     //!< Path for code generation
            int localHostID)        //!< ID of local host
{
    genHeader(model, path, localHostID);
    genCode(model, path, localHostID);
}
