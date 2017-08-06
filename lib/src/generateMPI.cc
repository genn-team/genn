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
#include "codeStream.h"

static void genHeader(const NNmodel &model, //!< Model description
               const string &path) //!< Path for code generationn
{
    //=======================
    // generate infraMPI.h
    //=======================

    // this file contains helpful macros and is separated out so that it can also be used by other code that is compiled separately
    string infraMPIName= path + "/" + model.getName() + "_CODE/infraMPI.h";
    ofstream fs;
    fs.open(infraMPIName.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    writeHeader(os);
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file infraMPI.h" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing MPI function definition." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    os << "#ifndef INFRAMPI_H" << std::endl;
    os << "#define INFRAMPI_H" << std::endl;
    os << std::endl;

#ifdef MPI_ENABLE
    os << "#include <mpi.h>" << std::endl;
#endif

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things to remote" << std::endl;
    os << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        os << "void push" << n.first << "SpikesToRemote(int remote);" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things from remote" << std::endl;
    os << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        os << "void pull" << n.first << "SpikesFromRemote(int remote);" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes to remote" << std::endl;
    os << std::endl;
    os << "void copySpikesToRemote(int remote);" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes from remote" << std::endl;
    os << std::endl;
    os << "void copySpikesFromRemote(int remote);" << std::endl;
    os << std::endl;

    os << "#endif" << std::endl;
    fs.close();
}

static void genCode(const NNmodel &model, //!< Model description
               const string &path) //!< Path for code generationn
{
    //=======================
    // generate infraMPI.cc
    //=======================

    string infraMPICodeName= path + "/" + model.getName() + "_CODE/infraMPI.cc";
    ofstream fs;
    fs.open(infraMPICodeName.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    writeHeader(os);
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file inftraMPI.cc" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing MPI infrastructure code." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << std::endl;

#ifdef MPI_ENABLE
    os << "#include <mpi.h>" << std::endl;
#endif

    os << "#include \"infraMPI.h\"" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying spikes to remote" << std::endl << std::endl;

    for(const auto &n : model.getLocalNeuronGroups()) {
        // neuron spike variables
        os << "void push" << n.first << "SpikesToRemote(int remote)" << std::endl;
        os << CodeStream::OB(1050);

        const size_t glbSpkCntSize = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
        os << "MPI_Send(glbSpkCnt" << n.first;
        os << ", "<< glbSpkCntSize;
        os << ", MPI_INT";
        os << ", remote, 0, MPI_COMM_WORLD);" << std::endl;

        const size_t glbSpkSize = n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons();
        os << "MPI_Send(glbSpk" << n.first;
        os << ", "<< glbSpkSize;
        os << ", MPI_INT";
        os << ", remote, 0, MPI_COMM_WORLD);" << std::endl;

        os << CodeStream::CB(1050);
        os << std::endl;
    }

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying spikes from remote" << std::endl << std::endl;

    for(const auto &n : model.getLocalNeuronGroups()) {
        // neuron spike variables
        os << "void pull" << n.first << "SpikesFromRemote(int remote)" << std::endl;
        os << CodeStream::OB(1051);

        const size_t glbSpkCntSize = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
        os << "MPI_Recv(glbSpkCnt" << n.first;
        os << ", "<< glbSpkCntSize;
        os << ", MPI_INT";
        os << ", remote, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);" << std::endl;

        const size_t glbSpkSize = n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons();
        os << "MPI_Recv(glbSpk" << n.first;
        os << ", "<< glbSpkSize;
        os << ", MPI_INT";
        os << ", remote, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);" << std::endl;

        os << CodeStream::CB(1051);
        os << std::endl;
    }
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes to remote" << std::endl << std::endl;

    os << "void copySpikesToRemote(int remote)" << std::endl;
    os << CodeStream::OB(1052);
    for(const auto &n : model.getLocalNeuronGroups()) {
        os << "push" << n.first << "SpikesToRemote(int remote);" << std::endl;
    }
    os << CodeStream::CB(1052);
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes from remote" << std::endl << std::endl;

    os << "void copySpikesFromRemote(int remote)" << std::endl;
    os << CodeStream::OB(1053) << std::endl;

    for(const auto &n : model.getLocalNeuronGroups()) {
      os << "pull" << n.first << "SpikesFromRemote(int remote);" << std::endl;
    }
    os << CodeStream::CB(1053);
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// communication function to sync spikes" << std::endl << std::endl;

    os << "void CommunicateSpikes()" << std::endl;
    os << CodeStream::OB(1054) << std::endl;

    os << "    int localID;" << std::endl;
    os << "    MPI_Comm_rank(MPI_COMM_WORLD, &localID);" << std::endl;
    for(const auto &n : model.getNeuronGroups()) {
            os << "    // Handling neuron " << n.first << std::endl;
        for(auto *s : n.second.getOutSyn()) {
            os << "    // send to synapse" << s->getName()<< std::endl;
            os << "    if (" << " localID != " << s->getClusterHostID() << ")" << std::endl;
            os << CodeStream::OB(1055) << std::endl;
            os << "copySpikesToRemote(" << s->getClusterHostID() << ");" <<std::endl;
            os << CodeStream::CB(1055);
        }
    }
    for(const auto &n : model.getNeuronGroups()) {
            os << "    // Handling neuron " << n.first << std::endl;
        for(auto *s : n.second.getInSyn()) {
            os << "    // receive from synapse" << s->getName()<< std::endl;
            os << "    if (" << " localID != " << s->getClusterHostID() << ")" << std::endl;
            os << CodeStream::OB(1055) << std::endl;
            os << "copySpikesFromRemote(" << s->getClusterHostID() << ");" <<std::endl;
            os << CodeStream::CB(1055);
        }
    }
    os << CodeStream::CB(1054);
    os << std::endl;

    fs.close();
}

//--------------------------------------------------------------------------
/*!
  \brief A function that generates predominantly MPI infrastructure code.

  In this function MPI infrastructure code are generated,
  including: MPI send and receive functions.
*/
//--------------------------------------------------------------------------
void genMPI(const NNmodel &model, //!< Model description
               const string &path) //!< Path for code generationn
{
    genHeader(model, path);
    genCode(model, path);
}
