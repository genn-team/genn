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

#include "code_generator/generateMPI.h"

// Standard C++ includes
#include <fstream>

// Standard C includes
#include <cstring>

// GeNN includes
#include "code_generator/codeStream.h"
#include "modelSpecInternal.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{

//--------------------------------------------------------------------------
/*! \brief This function returns the 32-bit hash of a string - because these are used across MPI nodes which may have different libstdc++ it would be risky to use std::hash
 */
//--------------------------------------------------------------------------
//! https://stackoverflow.com/questions/19411742/what-is-the-default-hash-function-used-in-c-stdunordered-map
//! suggests that libstdc++ uses MurmurHash2 so this seems as good a bet as any
//! MurmurHash2, by Austin Appleby
//! It has a few limitations -
//! 1. It will not work incrementally.
//! 2. It will not produce the same results on little-endian and big-endian
//!    machines.
uint32_t hashString(const std::string &string)
{
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.

    const uint32_t m = 0x5bd1e995;
    const unsigned int r = 24;

    // Get string length
    size_t len = string.length();

    // Initialize the hash to a 'random' value

    uint32_t h = 0xc70f6907 ^ (uint32_t)len;

    // Mix 4 bytes at a time into the hash
    const char *data = string.c_str();
    while(len >= 4)
    {
        // **NOTE** one of the assumptions of the original MurmurHash2 was that
        // "We can read a 4-byte value from any address without crashing".
        // Bad experiance tells me this may not be the case on ARM so use memcpy
        uint32_t k;
        memcpy(&k, data, 4);

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        data += 4;
        len -= 4;
    }

    // Handle the last few bytes of the input array
    switch(len)
    {
        case 3: h ^= data[2] << 16; // falls through
        case 2: h ^= data[1] << 8;  // falls through
        case 1: h ^= data[0];
                h *= m;             // falls through
    };

    // Do a few final mixes of the hash to ensure the last few
    // bytes are well-incorporated.

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}

void genHeader(const ModelSpecInternal &model,    //!< Model description
               const std::string &path,      //!< Path for code generationn
               int localHostID)         //!< ID of local host
{
    using namespace CodeGenerator;

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

void genCode(const ModelSpecInternal &model,  //!< Model description
             const std::string &path,    //!< Path for code generationn
             int localHostID)       //!< ID of local host
{
    using namespace CodeGenerator;

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
void genMPI(const ModelSpecInternal &model,   //!< Model description
            const std::string &path,     //!< Path for code generation
            int localHostID)        //!< ID of local host
{
    genHeader(model, path, localHostID);
    genCode(model, path, localHostID);
}
