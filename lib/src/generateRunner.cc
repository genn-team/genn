/*--------------------------------------------------------------------------
  Author: Thomas Nowotny
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  T.Nowotny@sussex.ac.uk
  
  initial version: 2010-02-07
  
  --------------------------------------------------------------------------*/

//-----------------------------------------------------------------------
/*!  \file generateRunner.cc

  \brief Contains functions to generate code for running the
  simulation on the GPU, and for I/O convenience functions between GPU
  and CPU space. Part of the code generation section.
*/
//--------------------------------------------------------------------------

#include "generateRunner.h"
#include "global.h"
#include "utils.h"
#include "codeGenUtils.h"
#include "codeStream.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
//--------------------------------------------------------------------------
//! \brief This function generates host and device variable definitions, of the given type and name.
//--------------------------------------------------------------------------

void variable_def(CodeStream &os, const string &type, const string &name, VarMode mode)
{
#ifndef CPU_ONLY
    if(mode & VarLocation::HOST) {
        os << type << " " << name << ";" << std::endl;
    }
    if(mode & VarLocation::DEVICE) {
        os << type << " d_" << name << ";" << std::endl;
        os << "__device__ " << type << " dd_" << name << ";" << std::endl;
    }
#else
    USE(mode);
    os << type << " " << name << ";" << std::endl;
#endif
}


//--------------------------------------------------------------------------
//! \brief This function generates host extern variable definitions, of the given type and name.
//--------------------------------------------------------------------------

void extern_variable_def(CodeStream &os, const string &type, const string &name, VarMode mode)
{
    // In windows making variables extern isn't enough to export then as DLL symbols - you need to add __declspec(dllexport)
#ifdef _WIN32
    const std::string varExportPrefix = GENN_PREFERENCES::buildSharedLibrary ? "__declspec(dllexport) extern" : "extern";
#else
    const std::string varExportPrefix = "extern";
#endif

#ifndef CPU_ONLY
    if(mode & VarLocation::HOST) {
        os << varExportPrefix << " " << type << " " << name << ";" << std::endl;
    }
    if(mode & VarLocation::DEVICE) {
        os << varExportPrefix << " " << type << " d_" << name << ";" << std::endl;
    }
#else
    USE(mode);
    os << varExportPrefix << " " << type << " " << name << ";" << std::endl;
#endif
}

//--------------------------------------------------------------------------
//! \brief This function generates host allocation code
//--------------------------------------------------------------------------

void allocate_host_variable(CodeStream &os, const string &type, const string &name, VarMode mode, const string &size)
{
#ifndef CPU_ONLY
    if(mode & VarLocation::HOST) {
        const char *flags = (mode & VarLocation::ZERO_COPY) ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
        os << "cudaHostAlloc(&" << name << ", " << size << " * sizeof(" << type << "), " << flags << ");" << std::endl;
    }
#else
    USE(mode);

    os << name << " = new " << type << "[" << size << "];" << std::endl;
#endif
}

void allocate_host_variable(CodeStream &os, const string &type, const string &name, VarMode mode, size_t size)
{
    allocate_host_variable(os, type, name, mode, to_string(size));
}

void allocate_device_variable(CodeStream &os, const string &type, const string &name, VarMode mode, const string &size)
{
#ifndef CPU_ONLY
    // If variable is present on device at all
    if(mode & VarLocation::DEVICE) {
        // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
        if(mode & VarLocation::ZERO_COPY) {
            os << "deviceZeroCopy(" << name << ", &d_" << name << ", dd_" << name << ");" << std::endl;
        }
        else {
            os << "deviceMemAllocate(&d_" << name << ", dd_" << name << ", " << size << " * sizeof(" << type << "));" << std::endl;
        }
    }
#else
    USE(os);
    USE(type);
    USE(name);
    USE(mode);
    USE(size);
#endif
}

void allocate_device_variable(CodeStream &os, const string &type, const string &name, VarMode mode, size_t size)
{
    allocate_device_variable(os, type, name, mode, to_string(size));
}

//--------------------------------------------------------------------------
//! \brief This function generates host and device allocation with standard names (name, d_name, dd_name) and estimates size based on size known at generate-time
//--------------------------------------------------------------------------
unsigned int allocate_variable(CodeStream &os, const string &type, const string &name, VarMode mode, size_t size)
{
    // Allocate host and device variables
    allocate_host_variable(os, type, name, mode, size);
    allocate_device_variable(os, type, name, mode, size);

    // Return size estimate
    return size * theSize(type);
}

void allocate_variable(CodeStream &os, const string &type, const string &name, VarMode mode, const string &size)
{
    // Allocate host and device variables
    allocate_host_variable(os, type, name, mode, size);
    allocate_device_variable(os, type, name, mode, size);
}

void free_host_variable(CodeStream &os, const string &name, VarMode mode)
{
#ifndef CPU_ONLY
    if(mode & VarLocation::HOST) {
        os << "CHECK_CUDA_ERRORS(cudaFreeHost(" << name << "));" << std::endl;
    }
#else
    USE(mode);
    os << "delete[] " << name << ";" << std::endl;
#endif
}

void free_device_variable(CodeStream &os, const string &name, VarMode mode)
{
#ifndef CPU_ONLY
    // If this variable wasn't allocated in zero-copy mode, free it
    if(mode & VarLocation::DEVICE) {
        os << "CHECK_CUDA_ERRORS(cudaFree(d_" << name << "));" << std::endl;
    }
#else
    USE(os);
    USE(name);
    USE(mode);
#endif
}

//--------------------------------------------------------------------------
//! \brief This function generates code to free host and device allocations with standard names (name, d_name, dd_name)
//--------------------------------------------------------------------------
void free_variable(CodeStream &os, const string &name, VarMode mode)
{
    free_host_variable(os, name, mode);
    free_device_variable(os, name, mode);
}

void genHostSpikeQueueAdvance(CodeStream &os, const NNmodel &model, int localHostID)
{
    for(auto &n : model.getRemoteNeuronGroups()) {
        if(n.second.isDelayRequired() && n.second.hasOutputToHost(localHostID)) {
            os << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
        }
    }
    for(auto &n : model.getLocalNeuronGroups()) {
        if (n.second.isDelayRequired()) {
            os << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
        }
    }
}

//--------------------------------------------------------------------------
//! \brief Can a variable with this mode be pushed and pulled between device and host
//--------------------------------------------------------------------------
#ifndef CPU_ONLY
bool canPushPullVar(VarMode varMode)
{
    // A variable can be pushed and pulled if it is located
    // on both host and device and doesn't use zero-copy memory
    return ((varMode & VarLocation::HOST) &&
            (varMode & VarLocation::DEVICE) &&
            !(varMode & VarLocation::ZERO_COPY));
}

void genPushSpikeCode(CodeStream &os, const NeuronGroup &ng, bool spikeEvent)
{
    // Get variable mode
    const VarMode varMode = spikeEvent ? ng.getSpikeEventVarMode() : ng.getSpikeVarMode();

    // Is push required at all
    const bool pushRequired = spikeEvent ?
        (ng.isSpikeEventRequired() && canPushPullVar(varMode))
        : canPushPullVar(varMode);

    const char *spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
    const char *spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

    if(pushRequired) {
        // If spikes are initialised on device, only copy if hostInitialisedOnly isn't set
        if(varMode & VarInit::DEVICE) {
            os << "if(!hostInitialisedOnly)" << CodeStream::OB(1061);
        }
        const size_t spkCntSize = (spikeEvent || ng.isTrueSpikeRequired()) ? ng.getNumDelaySlots() : 1;
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikeCntPrefix << ng.getName();
        os << ", " << spikeCntPrefix << ng.getName();
        os << ", " << spkCntSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;

        const size_t spkSize = (spikeEvent || ng.isTrueSpikeRequired()) ? ng.getNumNeurons() * ng.getNumDelaySlots() : ng.getNumNeurons();
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikePrefix << ng.getName();
        os << ", " << spikePrefix << ng.getName();
        os << ", " << spkSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;

        if(varMode & VarInit::DEVICE) {
            os << CodeStream::CB(1061);
        }
    }
}

void genPushCurrentSpikeFunctions(CodeStream &os, const NeuronGroup &ng, bool spikeEvent)
{
    // Is push required at all
    const bool pushRequired = spikeEvent ?
        (ng.isSpikeEventRequired() && canPushPullVar(ng.getSpikeEventVarMode()))
        : canPushPullVar(ng.getSpikeVarMode());

    // Is delay required
    const bool delayRequired = spikeEvent ?
        ng.isDelayRequired() :
        (ng.isTrueSpikeRequired() && ng.isDelayRequired());

    const char *spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
    const char *spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

    // current neuron spike variables
    os << "void push" << ng.getName();
    if(spikeEvent) {
        os << "CurrentSpikeEventsToDevice";
    }
    else {
        os << "CurrentSpikesToDevice";
    }
    os << "()";
    {
        CodeStream::Scope b(os);
        if(pushRequired) {
            if (delayRequired) {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikeCntPrefix << ng.getName() << "+spkQuePtr" << ng.getName();
                os << ", " << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikePrefix << ng.getName() << " + (spkQuePtr" << ng.getName() << "*" << ng.getNumNeurons() << ")";
                os << ", " << spikePrefix << ng.getName();
                os << "+(spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
                os << ", " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << "] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            }
            else {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikeCntPrefix << ng.getName();
                os << ", " << spikeCntPrefix << ng.getName();
                os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << spikePrefix << ng.getName();
                os << ", " << spikePrefix << ng.getName();
                os << ", " << spikeCntPrefix << ng.getName() << "[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            }
        }
    }
    os << std::endl;
}

void genPullCurrentSpikeFunctions(CodeStream &os, const NeuronGroup &ng, bool spikeEvent)
{
    // Is push required at all
    const bool pullRequired = spikeEvent ?
        (ng.isSpikeEventRequired() && canPushPullVar(ng.getSpikeEventVarMode()))
        : canPushPullVar(ng.getSpikeVarMode());

    // Is delay required
    const bool delayRequired = spikeEvent ?
        ng.isDelayRequired() :
        (ng.isTrueSpikeRequired() && ng.isDelayRequired());

    const char *spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
    const char *spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

    os << "void pull" << ng.getName();
    if(spikeEvent) {
        os << "CurrentSpikeEventsFromDevice";
    }
    else {
        os << "CurrentSpikesFromDevice";
    }
    os << "()" << std::endl;
    {
        CodeStream::Scope b(os);
        if(pullRequired) {
            if (delayRequired) {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", d_" << spikeCntPrefix << ng.getName() << " + spkQuePtr" << ng.getName();
                os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;

                os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikePrefix << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
                os << ", d_" << spikePrefix << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
                os << ", " << spikeCntPrefix << ng.getName() << "[spkQuePtr" << ng.getName() << "] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }
            else {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikeCntPrefix << ng.getName();
                os << ", d_" << spikeCntPrefix << ng.getName();
                os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << spikePrefix << ng.getName();
                os << ", d_" << spikePrefix << ng.getName();
                os << ", " << spikeCntPrefix << ng.getName() << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }
        }
    }
    os << std::endl;
}
#endif  // CPU_ONLY
}   // Anonymous namespace

//--------------------------------------------------------------------------
/*!
  \brief A function that generates predominantly host-side code.

  In this function host-side functions and other code are generated,
  including: Global host variables, "allocatedMem()" function for
  allocating memories, "freeMem" function for freeing the allocated
  memories, "initialize" for initializing host variables, "gFunc" and
  "initGRaw()" for use with plastic synapses if such synapses exist in
  the model.  
*/
//--------------------------------------------------------------------------
void genDefinitions(const NNmodel &model,   //!< Model description
                    const string &path,     //!< Path for code generationn
                    int localHostID)        //!< Host ID of local machine
{
    string SCLR_MIN;
    string SCLR_MAX;
    if (model.getPrecision() == "float") {
        SCLR_MIN= to_string(FLT_MIN)+"f";
        SCLR_MAX= to_string(FLT_MAX)+"f";
    }

    if (model.getPrecision() == "double") {
        SCLR_MIN= to_string(DBL_MIN);
        SCLR_MAX= to_string(DBL_MAX);
    }

    //=======================
    // generate definitions.h
    //=======================
    // this file contains helpful macros and is separated out so that it can also be used by other code that is compiled separately
    string definitionsName= model.getGeneratedCodePath(path, "definitions.h");
    ofstream fs;
    fs.open(definitionsName.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    writeHeader(os);
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file definitions.h" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing useful Macros used for both GPU amd CPU versions." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    os << "#ifndef DEFINITIONS_H" << std::endl;
    os << "#define DEFINITIONS_H" << std::endl;
    os << std::endl;

    os << "#include \"utils.h\"" << std::endl;
    if (model.isTimingEnabled()) {
        os << "#include \"hr_time.h\"" << std::endl;
    }
#ifdef MPI_ENABLE
    os << "#include \"mpi.h\"" << std::endl;
#endif
    os << "#include \"sparseUtils.h\"" << std::endl << std::endl;
    os << "#include \"sparseProjection.h\"" << std::endl;
    os << "#include <cstdint>" << std::endl;
    os << "#include <random>" << std::endl;
#ifndef CPU_ONLY
    os << "#include <curand_kernel.h>" << std::endl;
#endif
    os << std::endl;
#ifndef CPU_ONLY
    // write CUDA error handler macro
    os << "#ifndef CHECK_CUDA_ERRORS" << std::endl;
    os << "#define CHECK_CUDA_ERRORS(call) {\\" << std::endl;
    os << "    cudaError_t error = call;\\" << std::endl;
    os << "    if (error != cudaSuccess) {\\" << std::endl;
    os << "        fprintf(stderr, \"%s: %i: cuda error %i: %s\\n\", __FILE__, __LINE__, (int) error, cudaGetErrorString(error));\\" << std::endl;
    os << "        exit(EXIT_FAILURE);\\" << std::endl;
    os << "    }\\" << std::endl;
    os << "}" << std::endl;
    os << "#endif" << std::endl;
    os << std::endl;
#else
    // define CUDA device and function type qualifiers
    os << "#define __device__" << std::endl;
    os << "#define __global__" << std::endl;
    os << "#define __host__" << std::endl;
    os << "#define __constant__" << std::endl;
    os << "#define __shared__" << std::endl;
#endif // CPU_ONLY

    // write DT macro
    os << "#undef DT" << std::endl;
    if (model.getPrecision() == "float") {
        os << "#define DT " << to_string(model.getDT()) << "f" << std::endl;
    } else {
        os << "#define DT " << to_string(model.getDT()) << std::endl;
    }

    // write MYRAND macro
    os << "#ifndef MYRAND" << std::endl;
    os << "#define MYRAND(Y,X) Y = Y * 1103515245 + 12345; X = (Y >> 16);" << std::endl;
    os << "#endif" << std::endl;
    os << "#ifndef MYRAND_MAX" << std::endl;
    os << "#define MYRAND_MAX 0x0000FFFFFFFFFFFFLL" << std::endl;
    os << "#endif" << std::endl;
    os << std::endl;

    os << "#ifndef scalar" << std::endl;
    os << "typedef " << model.getPrecision() << " scalar;" << std::endl;
    os << "#endif" << std::endl;
    os << "#ifndef SCALAR_MIN" << std::endl;
    os << "#define SCALAR_MIN " << SCLR_MIN << std::endl;
    os << "#endif" << std::endl;
    os << "#ifndef SCALAR_MAX" << std::endl;
    os << "#define SCALAR_MAX " << SCLR_MAX << std::endl;
    os << "#endif" << std::endl;
    os << std::endl;
  
    // Begin extern C block around ALL definitions
    if(GENN_PREFERENCES::buildSharedLibrary) {
        os << "extern \"C\" {" << std::endl;
    }
        
    // In windows making variables extern isn't enough to export then as DLL symbols - you need to add __declspec(dllexport)
#ifdef _WIN32
    const std::string varExportPrefix = GENN_PREFERENCES::buildSharedLibrary ? "__declspec(dllexport) extern" : "extern";
#else
    const std::string varExportPrefix = "extern";
#endif

    //-----------------
    // GLOBAL VARIABLES

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global variables" << std::endl;
    os << std::endl;

    os << varExportPrefix << " unsigned long long iT;" << std::endl;
    os << varExportPrefix << " " << model.getPrecision() << " t;" << std::endl;
    if (model.isTimingEnabled()) {
#ifndef CPU_ONLY
        os << varExportPrefix << " cudaEvent_t neuronStart, neuronStop;" << std::endl;
#endif
        os << varExportPrefix << " double neuron_tme;" << std::endl;
        os << varExportPrefix << " CStopWatch neuron_timer;" << std::endl;
        if (!model.getLocalSynapseGroups().empty()) {
#ifndef CPU_ONLY
            os << varExportPrefix << " cudaEvent_t synapseStart, synapseStop;" << std::endl;
#endif
            os << varExportPrefix << " double synapse_tme;" << std::endl;
            os << varExportPrefix << " CStopWatch synapse_timer;" << std::endl;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
#ifndef CPU_ONLY
            os << varExportPrefix << " cudaEvent_t learningStart, learningStop;" << std::endl;
#endif
            os << varExportPrefix << " double learning_tme;" << std::endl;
            os << varExportPrefix << " CStopWatch learning_timer;" << std::endl;
        }
        if (!model.getSynapseDynamicsGroups().empty()) {
#ifndef CPU_ONLY
            os << varExportPrefix << " cudaEvent_t synDynStart, synDynStop;" << std::endl;
#endif
            os << varExportPrefix << " double synDyn_tme;" << std::endl;
            os << varExportPrefix << " CStopWatch synDyn_timer;" << std::endl;
        }
#ifndef CPU_ONLY
        if(model.isDeviceInitRequired(localHostID)) {
            os << "extern cudaEvent_t initDeviceStart, initDeviceStop;" << std::endl;
        }
        if(model.isDeviceSparseInitRequired()) {
            os << "extern cudaEvent_t sparseInitDeviceStart, sparseInitDeviceStop;" << std::endl;
        }
#endif
        os << "extern double initHost_tme;" << std::endl;
        os << "extern double initDevice_tme;" << std::endl;
        os << "extern CStopWatch initHost_timer;" << std::endl;
        os << "extern double sparseInitHost_tme;" << std::endl;
        os << "extern double sparseInitDevice_tme;" << std::endl;
        os << "extern CStopWatch sparseInitHost_timer;" << std::endl;
    }
    os << std::endl;
    if(model.isHostRNGRequired()) {
        os << "extern std::mt19937 rng;" << std::endl;

        os << "extern std::uniform_real_distribution<" << model.getPrecision() << "> standardUniformDistribution;" << std::endl;
        os << "extern std::normal_distribution<" << model.getPrecision() << "> standardNormalDistribution;" << std::endl;
        os << "extern std::exponential_distribution<" << model.getPrecision() << "> standardExponentialDistribution;" << std::endl;
    }
#ifndef CPU_ONLY
    if(model.isDeviceRNGRequired()) {
        os << "extern curandStatePhilox4_32_10_t *d_rng;" << std::endl;
    }
#endif  // CPU_ONLY
    os << std::endl;

    //---------------------------------
    // REMOTE NEURON GROUPS

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// remote neuron groups" << std::endl;
    os << std::endl;

    // Loop through remote neuron groups
    for(const auto &n : model.getRemoteNeuronGroups()) {
        // Write macro so whether a neuron group is remote or not can be determined at compile time
        // **NOTE** we do this for REMOTE groups so #ifdef GROUP_NAME_REMOTE is backward compatible
        os << "#define " << n.first << "_REMOTE" << std::endl;

        // If this neuron group has outputs to local host
        if(n.second.hasOutputToHost(localHostID)) {
            // Check that, whatever variable mode is set for these variables,
            // they are instantiated on host so they can be copied using MPI
            if(!(n.second.getSpikeVarMode() & VarLocation::HOST)) {
                gennError("Remote neuron group '" + n.first + "' has its spike variable mode set so it is not instantiated on the host - this is not supported");
            }

            extern_variable_def(os, "unsigned int *", "glbSpkCnt"+n.first, n.second.getSpikeVarMode());
            extern_variable_def(os, "unsigned int *", "glbSpk"+n.first, n.second.getSpikeVarMode());
        }
    }
    os << std::endl;

    //---------------------------------
    // HOST AND DEVICE NEURON VARIABLES

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// neuron variables" << std::endl;
    os << std::endl;

    for(const auto &n : model.getLocalNeuronGroups()) {
        extern_variable_def(os, "unsigned int *", "glbSpkCnt"+n.first, n.second.getSpikeVarMode());
        extern_variable_def(os, "unsigned int *", "glbSpk"+n.first, n.second.getSpikeVarMode());
        if (n.second.isSpikeEventRequired()) {
            extern_variable_def(os, "unsigned int *", "glbSpkCntEvnt"+n.first, n.second.getSpikeEventVarMode());
            extern_variable_def(os, "unsigned int *", "glbSpkEvnt"+n.first, n.second.getSpikeEventVarMode());
        }
        if (n.second.isDelayRequired()) {
            os << varExportPrefix << " unsigned int spkQuePtr" << n.first << ";" << std::endl;
        }
        if (n.second.isSpikeTimeRequired()) {
            extern_variable_def(os, model.getPrecision()+" *", "sT"+n.first, n.second.getSpikeTimeVarMode());
        }
#ifndef CPU_ONLY
        if(n.second.isSimRNGRequired()) {
            os << "extern curandState *d_rng" << n.first << ";" << std::endl;
        }
#endif
        auto neuronModel = n.second.getNeuronModel();
        for(auto const &v : neuronModel->getVars()) {
            extern_variable_def(os, v.second +" *", v.first + n.first, n.second.getVarMode(v.first));
        }
        for(auto const &v : neuronModel->getExtraGlobalParams()) {
            os << "extern " << v.second << " " << v.first + n.first << ";" << std::endl;
        }
    }
    os << std::endl;
    for(auto &n : model.getLocalNeuronGroups()) {
        os << "#define glbSpkShift" << n.first;
        if (n.second.isDelayRequired()) {
            os << " spkQuePtr" << n.first << "*" << n.second.getNumNeurons();
        }
        else {
            os << " 0";
        }
        os << std::endl;
    }

    for(const auto &n : model.getLocalNeuronGroups()) {
        // convenience macros for accessing spike count
        os << "#define spikeCount_" << n.first << " glbSpkCnt" << n.first;
        if (n.second.isDelayRequired() && n.second.isTrueSpikeRequired()) {
            os << "[spkQuePtr" << n.first << "]" << std::endl;
        }
        else {
            os << "[0]" << std::endl;
        }
        // convenience macro for accessing spikes
        os << "#define spike_" << n.first;
        if (n.second.isDelayRequired() && n.second.isTrueSpikeRequired()) {
            os << " (glbSpk" << n.first << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << "))" << std::endl;
        }
        else {
            os << " glbSpk" << n.first << std::endl;
        }
        if (n.second.isSpikeEventRequired()) {
            // convenience macros for accessing spike count
            os << "#define spikeEventCount_" << n.first << " glbSpkCntEvnt" << n.first;
            if (n.second.isDelayRequired()) {
                os << "[spkQuePtr" << n.first << "]" << std::endl;
            }
            else {
                os << "[0]" << std::endl;
            }
            // convenience macro for accessing spikes
            os << "#define spikeEvent_" << n.first;
            if (n.second.isDelayRequired()) {
                os << " (glbSpkEvnt" << n.first << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << "))" << std::endl;
            }
            else {
                os << " glbSpkEvnt" << n.first << std::endl;
            }
        }
    }
    os << std::endl;


    //----------------------------------
    // HOST AND DEVICE SYNAPSE VARIABLES

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// synapse variables" << std::endl;
    os << std::endl;

    for(const auto &s : model.getLocalSynapseGroups()) {
        extern_variable_def(os, model.getPrecision() + " *", "inSyn" + s.first, s.second.getInSynVarMode());

        if (s.second.isDendriticDelayRequired()) {
            extern_variable_def(os, model.getPrecision() + " *", "denDelay" + s.first, s.second.getDendriticDelayVarMode());
        }

        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            extern_variable_def(os, "uint32_t *", "gp" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);
        }
        else if (s.second.getMatrixType() & SynapseMatrixConnectivity::YALE) {
            os << varExportPrefix << " SparseProjection C" << s.first << ";" << std::endl;
        }
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
            // **TODO** different types
            os << varExportPrefix << " RaggedProjection<unsigned int> C" << s.first << ";" << std::endl;
        }

        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            for(const auto &v : s.second.getWUModel()->getVars()) {
                extern_variable_def(os, v.second + " *", v.first + s.first, s.second.getWUVarMode(v.first));
            }
        }
        
        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
            for(const auto &v : s.second.getPSModel()->getVars()) {
                extern_variable_def(os, v.second + " *", v.first + s.first, s.second.getPSVarMode(v.first));
            }
        }

        for(auto const &p : s.second.getWUModel()->getExtraGlobalParams()) {
            os << "extern " << p.second << " " << p.first + s.first << ";" << std::endl;
        }
    }
    os << std::endl;

    os << "#define Conductance SparseProjection" << std::endl;
    os << "/*struct Conductance is deprecated. \n\
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. \n\
  Please consider updating your user code by renaming Conductance as SparseProjection \n\
  and making g member a synapse variable.*/" << std::endl;
    os << std::endl;

    // In windows wrapping functions in extern "C" isn't enough to export then as DLL symbols - you need to add __declspec(dllexport)
#ifdef _WIN32
    const string funcExportPrefix = GENN_PREFERENCES::buildSharedLibrary ? "__declspec(dllexport) " : "";
#else
    const string funcExportPrefix = "";
#endif


#ifndef CPU_ONLY
    // generate headers for the communication utility functions such as
    // pullXXXStateFromDevice() etc. This is useful for the brian2genn
    // interface where we do more proper compile/link and do not want
    // to include runnerGPU.cc into all relevant code_objects (e.g.
    // spike and state monitors
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying remote data to device" << std::endl << std::endl;
    for(const auto &n : model.getRemoteNeuronGroups()) {
        if(n.second.hasOutputToHost(localHostID)) {
            os << "void push" << n.first << "SpikesToDevice(bool hostInitialisedOnly = false);" << std::endl;
            os << "void push" << n.first << "CurrentSpikesToDevice();" << std::endl;
        }
    }
    os << std::endl;
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things to device" << std::endl;
    os << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        os << funcExportPrefix << "void push" << n.first << "StateToDevice(bool hostInitialisedOnly = false);" << std::endl;
        os << funcExportPrefix << "void push" << n.first << "SpikesToDevice(bool hostInitialisedOnly = false);" << std::endl;
        os << funcExportPrefix << "void push" << n.first << "SpikeEventsToDevice(bool hostInitialisedOnly = false);" << std::endl;
        os << funcExportPrefix << "void push" << n.first << "CurrentSpikesToDevice();" << std::endl;
        os << funcExportPrefix << "void push" << n.first << "CurrentSpikeEventsToDevice();" << std::endl;
    }
    for(const auto &s : model.getLocalSynapseGroups()) {
        os << "#define push" << s.first << "ToDevice push" << s.first << "StateToDevice" << std::endl;
        os << funcExportPrefix << "void push" << s.first << "StateToDevice(bool hostInitialisedOnly = false);" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things from device" << std::endl;
    os << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        os << funcExportPrefix << "void pull" << n.first << "StateFromDevice();" << std::endl;
        os << funcExportPrefix << "void pull" << n.first << "SpikesFromDevice();" << std::endl;
        os << funcExportPrefix << "void pull" << n.first << "SpikeEventsFromDevice();" << std::endl;
        os << funcExportPrefix << "void pull" << n.first << "CurrentSpikesFromDevice();" << std::endl;
        os << funcExportPrefix << "void pull" << n.first << "CurrentSpikeEventsFromDevice();" << std::endl;
    }
    for(const auto &s : model.getLocalSynapseGroups()) {
        os << "#define pull" << s.first << "FromDevice pull" << s.first << "StateFromDevice" << std::endl;
        os << funcExportPrefix << "void pull" << s.first << "StateFromDevice();" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying values to device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copyStateToDevice(bool hostInitialisedOnly = false);" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes to device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copySpikesToDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikes to device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copyCurrentSpikesToDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spike events to device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copySpikeEventsToDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikes to device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copyCurrentSpikeEventsToDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying values from device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copyStateFromDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes from device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copySpikesFromDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikes from device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copyCurrentSpikesFromDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying spike numbers from device (note, only use when only interested"<< std::endl;
    os << "// in spike numbers; copySpikesFromDevice() already includes this)" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copySpikeNFromDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------"<< std::endl;
    os << "// global copying spikeEvents from device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copySpikeEventsFromDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikeEvents from device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copyCurrentSpikeEventsFromDevice();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spike event numbers from device (note, only use when only interested" << std::endl;
    os << "// in spike numbers; copySpikeEventsFromDevice() already includes this)" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copySpikeEventNFromDevice();" << std::endl;
    os << std::endl;
#endif

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Function for setting the CUDA device and the host's global variables." << std::endl;
    os << "// Also estimates memory usage on device." << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void allocateMem();" << std::endl;
    os << std::endl;

    for(const auto &s : model.getLocalSynapseGroups()) {
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::YALE) {
            os << funcExportPrefix << "void allocate" << s.first << "(unsigned int connN);" << std::endl;
            os << std::endl;
        }
    }

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Function to (re)set all model variables to their compile-time, homogeneous initial" << std::endl;
    os << "// values. Note that this typically includes synaptic weight values. The function" << std::endl;
    os << "// (re)sets host side variables and copies them to the GPU device." << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void initialize();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Method for cleaning up and resetting device while quitting GeNN." << std::endl;
    os << std::endl;
    os << "void exitGeNN();" << std::endl;
    os << std::endl;

#ifndef CPU_ONLY
    os << funcExportPrefix << "void initializeAllSparseArrays();" << std::endl;
    os << std::endl;
#endif

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// initialization of variables, e.g. reverse sparse arrays etc." << std::endl;
    os << "// that the user would not want to worry about" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void init" << model.getName() << "();" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Function to free all global memory structures." << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void freeMem();" << std::endl;
    os << std::endl;

    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "// Function to convert a firing probability (per time step) to an integer of type uint64_t" << std::endl;
    os << "// that can be used as a threshold for the GeNN random number generator to generate events with the given probability." << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void convertProbabilityToRandomNumberThreshold(" << model.getPrecision() << " *p_pattern, " << model.getRNType() << " *pattern, int N);" << std::endl;
    os << std::endl;

    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "// Function to convert a firing rate (in kHz) to an integer of type uint64_t that can be used" << std::endl;
    os << "// as a threshold for the GeNN random number generator to generate events with the given rate." << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void convertRateToRandomNumberThreshold(" << model.getPrecision() << " *rateKHz_pattern, " << model.getRNType() << " *pattern, int N);" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// the actual time stepping procedure (using CPU)" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void stepTimeCPU();" << std::endl;
    os << std::endl;

#ifndef CPU_ONLY
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// the actual time stepping procedure (using GPU)" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void stepTimeGPU();" << std::endl;
    os << std::endl;
#endif

    // End extern C block around definitions
    if(GENN_PREFERENCES::buildSharedLibrary) {
        os << "}\t// extern \"C\"" << std::endl;
    }

    //--------------------------
    // HELPER FUNCTIONS
#ifndef CPU_ONLY
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Throw an error for \"old style\" time stepping calls (using GPU)" << std::endl;
    os << std::endl;
    os << "template <class T>" << std::endl;
    os << "void stepTimeGPU(T arg1, ...)";
    {
        CodeStream::Scope b(os);
        os << "gennError(\"Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.\");" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper function for allocating memory blocks on the GPU device" << std::endl;
    os << std::endl;
    os << "template<class T>" << std::endl;
    os << "void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)";
    {
        CodeStream::Scope b(os);
        os << "void *devptr;" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper function for getting the device pointer corresponding to a zero-copied host pointer and assigning it to a symbol" << std::endl;
    os << std::endl;
    os << "template<class T>" << std::endl;
    os << "void deviceZeroCopy(T hostPtr, const T *devPtr, const T &devSymbol)";
    {
        CodeStream::Scope b(os);
        os << "CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void **)devPtr, (void*)hostPtr, 0));" << std::endl;
        os << "void *devSymbolPtr;" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devSymbolPtr, devSymbol));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(devSymbolPtr, devPtr, sizeof(void*), cudaMemcpyHostToDevice));" << std::endl;
    }
    os << std::endl;
#endif

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Throw an error for \"old style\" time stepping calls (using CPU)" << std::endl;
    os << std::endl;
    os << "template <class T>" << std::endl;
    os << "void stepTimeCPU(T arg1, ...)";
    {
        CodeStream::Scope b(os);
        os << "gennError(\"Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.\");" << std::endl;
    }
    os<< std::endl;

    os << "#endif" << std::endl;
    fs.close();
}

void genSupportCode(const NNmodel &model, //!< Model description
               const string &path) //!< Path for code generationn
{
    //========================
    // generate support_code.h
    //========================

    string supportCodeName= model.getGeneratedCodePath(path, "support_code.h");
    ofstream fs;
    fs.open(supportCodeName.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    writeHeader(os);
    os << std::endl;

       // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file support_code.h" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing support code provided by the user and used for both GPU amd CPU versions." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    os << "#ifndef SUPPORT_CODE_H" << std::endl;
    os << "#define SUPPORT_CODE_H" << std::endl;
    // write the support codes
    os << "// support code for neuron and synapse models" << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        if (!n.second.getNeuronModel()->getSupportCode().empty()) {
            os << "namespace " << n.first << "_neuron";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(n.second.getNeuronModel()->getSupportCode(), model.getPrecision()) << std::endl;
            }
        }
    }
    for(const auto &s : model.getLocalSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        if (!wu->getSimSupportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_simCode";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(wu->getSimSupportCode(), model.getPrecision()) << std::endl;
            }
        }
        if (!wu->getLearnPostSupportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_simLearnPost";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(wu->getLearnPostSupportCode(), model.getPrecision()) << std::endl;
            }
        }
        if (!wu->getSynapseDynamicsSuppportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_synapseDynamics";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(wu->getSynapseDynamicsSuppportCode(), model.getPrecision()) << std::endl;
            }
        }
        if (!psm->getSupportCode().empty()) {
            os << "namespace " << s.first << "_postsyn";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(psm->getSupportCode(), model.getPrecision()) << std::endl;
            }
        }

    }
    os << "#endif" << std::endl;
    fs.close();
}

void genRunner(const NNmodel &model,    //!< Model description
               const string &path,      //!< Path for code generationn
               int localHostID)         //!< ID of local host

{
    // Counter used for tracking memory allocations
    unsigned int mem = 0;

    //cout << "entering genRunner" << std::endl;
    string runnerName= model.getGeneratedCodePath(path, "runner.cc");
    ofstream fs;
    fs.open(runnerName.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    writeHeader(os);
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file runner.cc" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing general control code." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << std::endl;

    os << "#define RUNNER_CC_COMPILE" << std::endl;
    os << std::endl;
    os << "#include \"definitions.h\"" << std::endl;
    os << "#include <cstdlib>" << std::endl;
    os << "#include <cstdio>" << std::endl;
    os << "#include <cmath>" << std::endl;
    os << "#include <ctime>" << std::endl;
    os << "#include <cassert>" << std::endl;
    os << "#include <stdint.h>" << std::endl;

    // **NOTE** if we are using GCC on x86_64, bugs in some version of glibc can cause
    // bad performance issues so need this to allow us to perform a runtime check
    os << "#if defined(__GNUG__) && !defined(__clang__) && defined(__x86_64__)" << std::endl;
    os << "    #include <gnu/libc-version.h>" << std::endl;
    os << "#endif" << std::endl;
    os << std::endl;


    //-----------------
    // GLOBAL VARIABLES

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global variables" << std::endl;
    os << std::endl;
    
    // Begin extern C block around variable declarations
    if(GENN_PREFERENCES::buildSharedLibrary) {
        os << "extern \"C\" {" << std::endl;
    }
    
    os << "unsigned long long iT;" << std::endl;
    os << model.getPrecision() << " t;" << std::endl;
    if (model.isTimingEnabled()) {
#ifndef CPU_ONLY
        os << "cudaEvent_t neuronStart, neuronStop;" << std::endl;
#endif
        os << "double neuron_tme;" << std::endl;
        os << "CStopWatch neuron_timer;" << std::endl;
        if (!model.getLocalSynapseGroups().empty()) {
#ifndef CPU_ONLY
            os << "cudaEvent_t synapseStart, synapseStop;" << std::endl;
#endif
            os << "double synapse_tme;" << std::endl;
            os << "CStopWatch synapse_timer;" << std::endl;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
#ifndef CPU_ONLY
            os << "cudaEvent_t learningStart, learningStop;" << std::endl;
#endif
            os << "double learning_tme;" << std::endl;
            os << "CStopWatch learning_timer;" << std::endl;
        }
        if (!model.getSynapseDynamicsGroups().empty()) {
#ifndef CPU_ONLY
            os << "cudaEvent_t synDynStart, synDynStop;" << std::endl;
#endif
            os << "double synDyn_tme;" << std::endl;
            os << "CStopWatch synDyn_timer;" << std::endl;
        }
#ifndef CPU_ONLY
        if(model.isDeviceInitRequired(localHostID)) {
            os << "cudaEvent_t initDeviceStart, initDeviceStop;" << std::endl;
        }
        if(model.isDeviceSparseInitRequired()) {
            os << "cudaEvent_t sparseInitDeviceStart, sparseInitDeviceStop;" << std::endl;
        }
#endif
        os << "double initHost_tme;" << std::endl;
        os << "double initDevice_tme;" << std::endl;
        os << "CStopWatch initHost_timer;" << std::endl;
        os << "double sparseInitHost_tme;" << std::endl;
        os << "double sparseInitDevice_tme;" << std::endl;
        os << "CStopWatch sparseInitHost_timer;" << std::endl;
    } 
    os << std::endl;
    if(model.isHostRNGRequired()) {
        os << "std::mt19937 rng;" << std::endl;

        // Construct standard host distributions as recreating them each call is slow
        os << "std::uniform_real_distribution<" << model.getPrecision() << "> standardUniformDistribution(" << model.scalarExpr(0.0) << ", " << model.scalarExpr(1.0) << ");" << std::endl;
        os << "std::normal_distribution<" << model.getPrecision() << "> standardNormalDistribution(" << model.scalarExpr(0.0) << ", " << model.scalarExpr(1.0) << ");" << std::endl;
        os << "std::exponential_distribution<" << model.getPrecision() << "> standardExponentialDistribution(" << model.scalarExpr(1.0) << ");" << std::endl;
    }
#ifndef CPU_ONLY
    if(model.isDeviceRNGRequired()) {
        os << "curandStatePhilox4_32_10_t *d_rng;" << std::endl;
        os << "__device__ curandStatePhilox4_32_10_t *dd_rng;" << std::endl;
    }
#endif
    os << std::endl;

    //---------------------------------
    // REMOTE NEURON GROUPS
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// remote neuron groups" << std::endl;
    os << std::endl;

    // Loop through remote neuron groups
    for(const auto &n : model.getRemoteNeuronGroups()) {
        // If this neuron group has outputs to local host
        if(n.second.hasOutputToHost(localHostID)) {
            variable_def(os, "unsigned int *", "glbSpkCnt"+n.first, n.second.getSpikeVarMode());
            variable_def(os, "unsigned int *", "glbSpk"+n.first, n.second.getSpikeVarMode());

            if (n.second.isDelayRequired()) {
                os << "unsigned int spkQuePtr" << n.first << ";" << std::endl;
#ifndef CPU_ONLY
                os << "__device__ volatile unsigned int dd_spkQuePtr" << n.first << ";" << std::endl;
#endif
            }
        }
    }
    os << std::endl;

    //---------------------------------
    // HOST AND DEVICE NEURON VARIABLES

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// neuron variables" << std::endl;
    os << std::endl;

#ifndef CPU_ONLY
    os << "__device__ volatile unsigned int d_done;" << std::endl;
#endif
    for(const auto &n : model.getLocalNeuronGroups()) {
        variable_def(os, "unsigned int *", "glbSpkCnt"+n.first, n.second.getSpikeVarMode());
        variable_def(os, "unsigned int *", "glbSpk"+n.first, n.second.getSpikeVarMode());
        if (n.second.isSpikeEventRequired()) {
            variable_def(os, "unsigned int *", "glbSpkCntEvnt"+n.first, n.second.getSpikeEventVarMode());
            variable_def(os, "unsigned int *", "glbSpkEvnt"+n.first, n.second.getSpikeEventVarMode());
        }
        if (n.second.isDelayRequired()) {
            os << "unsigned int spkQuePtr" << n.first << ";" << std::endl;
#ifndef CPU_ONLY
            os << "__device__ volatile unsigned int dd_spkQuePtr" << n.first << ";" << std::endl;
#endif
        }
        if (n.second.isSpikeTimeRequired()) {
            variable_def(os, model.getPrecision()+" *", "sT"+n.first, n.second.getSpikeTimeVarMode());
        }
#ifndef CPU_ONLY
        if(n.second.isSimRNGRequired()) {
            os << "curandState *d_rng" << n.first << ";" << std::endl;
            os << "__device__ curandState *dd_rng" << n.first << ";" << std::endl;
        }
#endif

        auto neuronModel = n.second.getNeuronModel();
        for(auto const &v : neuronModel->getVars()) {
            variable_def(os, v.second + " *", v.first + n.first, n.second.getVarMode(v.first));
        }
        for(auto const &v : neuronModel->getExtraGlobalParams()) {
            os << v.second << " " <<  v.first << n.first << ";" << std::endl;
        }
    }
    os << std::endl;


    //----------------------------------
    // HOST AND DEVICE SYNAPSE VARIABLES

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// synapse variables" << std::endl;
    os << std::endl;

   for(const auto &s : model.getLocalSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        variable_def(os, model.getPrecision() + " *", "inSyn" + s.first, s.second.getInSynVarMode());

        if(s.second.isDendriticDelayRequired()) {
            variable_def(os, model.getPrecision() + " *", "denDelay" + s.first, s.second.getDendriticDelayVarMode());
        }

        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            variable_def(os, "uint32_t *", "gp"+s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);
        }
        else if (s.second.getMatrixType() & SynapseMatrixConnectivity::YALE) {
            os << "SparseProjection C" << s.first << ";" << std::endl;
#ifndef CPU_ONLY
            os << "unsigned int *d_indInG" << s.first << ";" << std::endl;
            os << "__device__ unsigned int *dd_indInG" << s.first << ";" << std::endl;
            os << "unsigned int *d_ind" << s.first << ";" << std::endl;
            os << "__device__ unsigned int *dd_ind" << s.first << ";" << std::endl;
            if (model.isSynapseGroupDynamicsRequired(s.first)) {
                os << "unsigned int *d_preInd" << s.first << ";" << std::endl;
                os << "__device__ unsigned int *dd_preInd" << s.first << ";" << std::endl;
            }
            if (model.isSynapseGroupPostLearningRequired(s.first)) {
                // TODO: make conditional on post-spike driven learning actually taking place
                os << "unsigned int *d_revIndInG" << s.first << ";" << std::endl;
                os << "__device__ unsigned int *dd_revIndInG" << s.first << ";" << std::endl;
                os << "unsigned int *d_revInd" << s.first << ";" << std::endl;
                os << "__device__ unsigned int *dd_revInd" << s.first << ";" << std::endl;
                os << "unsigned int *d_remap" << s.first << ";" << std::endl;
                os << "__device__ unsigned int *dd_remap" << s.first << ";" << std::endl;
            }
#endif
        }
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
            // **TODO** other index types
            os << "RaggedProjection<unsigned int> C" << s.first << "(" << s.second.getMaxConnections() << "," << s.second.getMaxSourceConnections() << ");" << std::endl;
#ifndef CPU_ONLY
            os << "unsigned int *d_rowLength" << s.first << ";" << std::endl;
            os << "__device__ unsigned int *dd_rowLength" << s.first << ";" << std::endl;
            os << "unsigned int *d_ind" << s.first << ";" << std::endl;
            os << "__device__ unsigned int *dd_ind" << s.first << ";" << std::endl;

            if (model.isSynapseGroupDynamicsRequired(s.first)) {
                os << "unsigned int *d_synRemap" << s.first << ";" << std::endl;
                os << "__device__ unsigned int *dd_synRemap" << s.first << ";" << std::endl;
            }

            if (model.isSynapseGroupPostLearningRequired(s.first)) {
                os << "unsigned int *d_colLength" << s.first << ";" << std::endl;
                os << "__device__ unsigned int *dd_colLength" << s.first << ";" << std::endl;
                os << "unsigned int *d_remap" << s.first << ";" << std::endl;
                os << "__device__ unsigned int *dd_remap" << s.first << ";" << std::endl;
            }
#endif  // CPU_ONLY
        }


        // If weight update variables should be individual
        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            for(const auto &v : wu->getVars()) {
                variable_def(os, v.second + " *", v.first + s.first, s.second.getWUVarMode(v.first));
            }
        }
        // If postsynaptic model variables should be individual
        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
            for(const auto &v : psm->getVars()) {
                variable_def(os, v.second+" *", v.first + s.first, s.second.getPSVarMode(v.first));
            }
        }

        for(const auto &v : wu->getExtraGlobalParams()) {
            os << v.second << " " <<  v.first << s.first << ";" << std::endl;
        }
    }
    os << std::endl;
    
    // End extern C block around variable declarations
    if(GENN_PREFERENCES::buildSharedLibrary) {
        os << "}\t// extern \"C\"" << std::endl;
    }

    //--------------------------
    // HOST AND DEVICE FUNCTIONS

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\brief Function to convert a firing probability (per time step) " << std::endl;
    os << "to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given probability." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;
    os << "void convertProbabilityToRandomNumberThreshold(" << model.getPrecision() << " *p_pattern, " << model.getRNType() << " *pattern, int N)";
    {
        CodeStream::Scope b(os);
        os << model.getPrecision() << " fac= pow(2.0, (double) sizeof(" << model.getRNType() << ")*8-16);" << std::endl;
        os << "for (int i= 0; i < N; i++)";
        {
            CodeStream::Scope b(os);
            os << "pattern[i]= (" << model.getRNType() << ") (p_pattern[i]*fac);" << std::endl;
        }
    }
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\brief Function to convert a firing rate (in kHz) " << std::endl;
    os << "to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given rate." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;
    os << "void convertRateToRandomNumberThreshold(" << model.getPrecision() << " *rateKHz_pattern, " << model.getRNType() << " *pattern, int N)";
    {
        CodeStream::Scope b(os);
        os << model.getPrecision() << " fac= pow(2.0, (double) sizeof(" << model.getRNType() << ")*8-16)*DT;" << std::endl;
        os << "for (int i= 0; i < N; i++)";
        {
            CodeStream::Scope b(os);
            os << "pattern[i]= (" << model.getRNType() << ") (rateKHz_pattern[i]*fac);" << std::endl;
        }
    }
    os << std::endl;

    // include simulation kernels
#ifndef CPU_ONLY
    os << "#include \"runnerGPU.cc\"" << std::endl;
#endif
#ifdef MPI_ENABLE
    os << "#include \"mpi.cc\"" << std::endl;
#endif
    os << "#include \"init.cc\"" << std::endl;

    // If model can be run on GPU, include CPU simulation functions
    if(model.canRunOnCPU()) {
        os << "#include \"neuronFnct.cc\"" << std::endl;
        if (!model.getLocalSynapseGroups().empty()) {
            os << "#include \"synapseFnct.cc\"" << std::endl;
        }
    }
    os << std::endl;

    // ---------------------------------------------------------------------
    // Function for setting the CUDA device and the host's global variables.
    // Also estimates memory usage on device ...
    os << "void allocateMem()";
    {
        CodeStream::Scope b(os);
#ifndef CPU_ONLY
        os << "CHECK_CUDA_ERRORS(cudaSetDevice(" << theDevice << "));" << std::endl;

        // If the model requires zero-copy
        if(model.zeroCopyInUse()) {
            // If device doesn't support mapping host memory error
            if(!deviceProp[theDevice].canMapHostMemory) {
                gennError("Device does not support mapping CPU host memory!");
            }

            // set appropriate device flags
            os << "CHECK_CUDA_ERRORS(cudaSetDeviceFlags(cudaDeviceMapHost));" << std::endl;
        }

        // If RNG is required, allocate memory for global philox RNG
        if(model.isDeviceRNGRequired()) {
            allocate_device_variable(os, "curandStatePhilox4_32_10_t", "rng", VarMode::LOC_DEVICE_INIT_DEVICE, 1);
        }
#endif
        //cout << "model.neuronGroupN " << model.neuronGrpN << std::endl;
        //os << "    " << model.getPrecision() << " free_m, total_m;" << std::endl;
        //os << "    cudaMemGetInfo((size_t*) &free_m, (size_t*) &total_m);" << std::endl;

        if (model.isTimingEnabled()) {
#ifndef CPU_ONLY
            os << "cudaEventCreate(&neuronStart);" << std::endl;
            os << "cudaEventCreate(&neuronStop);" << std::endl;
#endif
            os << "neuron_tme= 0.0;" << std::endl;
            if (!model.getLocalSynapseGroups().empty()) {
#ifndef CPU_ONLY
                os << "cudaEventCreate(&synapseStart);" << std::endl;
                os << "cudaEventCreate(&synapseStop);" << std::endl;
#endif
                os << "synapse_tme= 0.0;" << std::endl;
            }
            if (!model.getSynapsePostLearnGroups().empty()) {
#ifndef CPU_ONLY
                os << "cudaEventCreate(&learningStart);" << std::endl;
                os << "cudaEventCreate(&learningStop);" << std::endl;
#endif
                os << "learning_tme= 0.0;" << std::endl;
            }
            if (!model.getSynapseDynamicsGroups().empty()) {
#ifndef CPU_ONLY
                os << "cudaEventCreate(&synDynStart);" << std::endl;
                os << "cudaEventCreate(&synDynStop);" << std::endl;
#endif
                os << "synDyn_tme= 0.0;" << std::endl;
            }
#ifndef CPU_ONLY
            if(model.isDeviceInitRequired(localHostID)) {
                os << "cudaEventCreate(&initDeviceStart);" << std::endl;
                os << "cudaEventCreate(&initDeviceStop);" << std::endl;
            }
            if(model.isDeviceSparseInitRequired()) {
                os << "cudaEventCreate(&sparseInitDeviceStart);" << std::endl;
                os << "cudaEventCreate(&sparseInitDeviceStop);" << std::endl;
            }
#endif
            os << "initHost_tme = 0.0;" << std::endl;
            os << "initDevice_tme = 0.0;" << std::endl;
            os << "sparseInitHost_tme = 0.0;" << std::endl;
            os << "sparseInitDevice_tme = 0.0;" << std::endl;
        }

        // ALLOCATE REMOTE NEURON VARIABLES
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// remote neuron groups" << std::endl;
        os << std::endl;

        // Loop through remote neuron groups
        for(const auto &n : model.getRemoteNeuronGroups()) {
            // If this neuron group has outputs to local host
            if(n.second.hasOutputToHost(localHostID)) {
                // Allocate population spike count
                mem += allocate_variable(os, "unsigned int", "glbSpkCnt" + n.first, n.second.getSpikeVarMode(),
                                        n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1);

                // Allocate population spike output buffer
                mem += allocate_variable(os, "unsigned int", "glbSpk" + n.first, n.second.getSpikeVarMode(),
                                        n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());
            }
        }
        os << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// local neuron groups" << std::endl;

        // ALLOCATE NEURON VARIABLES
        for(const auto &n : model.getLocalNeuronGroups()) {
            // Allocate population spike count
            mem += allocate_variable(os, "unsigned int", "glbSpkCnt" + n.first, n.second.getSpikeVarMode(),
                                    n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1);

            // Allocate population spike output buffer
            mem += allocate_variable(os, "unsigned int", "glbSpk" + n.first, n.second.getSpikeVarMode(),
                                    n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());


            if (n.second.isSpikeEventRequired()) {
                // Allocate population spike-like event counters
                mem += allocate_variable(os, "unsigned int", "glbSpkCntEvnt" + n.first, n.second.getSpikeEventVarMode(),
                                        n.second.getNumDelaySlots());

                // Allocate population spike-like event output buffer
                mem += allocate_variable(os, "unsigned int", "glbSpkEvnt" + n.first, n.second.getSpikeEventVarMode(),
                                        n.second.getNumNeurons() * n.second.getNumDelaySlots());
            }

            // Allocate buffer to hold last spike times if required
            if (n.second.isSpikeTimeRequired()) {
                mem += allocate_variable(os, model.getPrecision(), "sT" + n.first, n.second.getSpikeTimeVarMode(),
                                        n.second.getNumNeurons() * n.second.getNumDelaySlots());
            }

#ifndef CPU_ONLY
            if(n.second.isSimRNGRequired()) {
                allocate_device_variable(os, "curandState", "rng" + n.first, VarMode::LOC_DEVICE_INIT_DEVICE,
                                        n.second.getNumNeurons());
            }
#endif  // CPU_ONLY

            // Allocate memory for neuron model's state variables
            for(const auto &v : n.second.getNeuronModel()->getVars()) {
                mem += allocate_variable(os, v.second, v.first + n.first, n.second.getVarMode(v.first),
                                        n.second.isVarQueueRequired(v.first) ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());
            }
            os << std::endl;
        }

        // ALLOCATE SYNAPSE VARIABLES
        for(const auto &s : model.getLocalSynapseGroups()) {
            const auto *wu = s.second.getWUModel();
            const auto *psm = s.second.getPSModel();

            // Allocate buffer to hold input coming from this synapse population
            mem += allocate_variable(os, model.getPrecision(), "inSyn" + s.first, s.second.getInSynVarMode(),
                                     s.second.getTrgNeuronGroup()->getNumNeurons());

            // Allocate buffer to delay input coming from this synapse population
            if(s.second.isDendriticDelayRequired()) {
                mem += allocate_variable(os, model.getPrecision(), "denDelay" + s.first, s.second.getDendriticDelayVarMode(),
                                         s.second.getMaxDendriticDelaySlots() * s.second.getTrgNeuronGroup()->getNumNeurons());
            }
            // If connectivity is defined using a bitmask, allocate memory for bitmask
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                const size_t gpSize = (s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumNeurons()) / 32 + 1;
                mem += allocate_variable(os, "uint32_t", "gp" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST, gpSize);
            }
            else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                const size_t size = s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getMaxConnections();

                // Allocate row lengths
                allocate_host_variable(os, "unsigned int", "C" + s.first + ".rowLength", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                    s.second.getSrcNeuronGroup()->getNumNeurons());
                allocate_device_variable(os, "unsigned int", "rowLength" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                        s.second.getSrcNeuronGroup()->getNumNeurons());

                // Allocate target indices
                const std::string postIndexType = "unsigned int";
                allocate_host_variable(os, postIndexType, "C" + s.first + ".ind", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                       size);
                allocate_device_variable(os, postIndexType, "ind" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                         size);

                if(model.isSynapseGroupPostLearningRequired(s.first)) {
                    const size_t postSize = s.second.getTrgNeuronGroup()->getNumNeurons() * s.second.getMaxSourceConnections();
                    
                    // Allocate column lengths
                    allocate_host_variable(os,  "unsigned int", "C" + s.first + ".colLength", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                           s.second.getTrgNeuronGroup()->getNumNeurons());
                    allocate_device_variable(os,  "unsigned int", "colLength" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                             s.second.getTrgNeuronGroup()->getNumNeurons());
                    
                    // Allocate remap
                    allocate_host_variable(os,  "unsigned int", "C" + s.first + ".remap", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                           postSize);
                    allocate_device_variable(os,  "unsigned int", "remap" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                             postSize);
                }

                if(model.isSynapseGroupDynamicsRequired(s.first)) {
                    // Allocate synRemap
                    allocate_host_variable(os,  "unsigned int", "C" + s.first + ".synRemap", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                           size + 1);
                    allocate_device_variable(os,  "unsigned int", "synRemap" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                             size + 1);
                }
                
                if(s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                    for(const auto &v : wu->getVars()) {
                        mem += allocate_variable(os, v.second, v.first + s.first, s.second.getWUVarMode(v.first), size);
                    }
                }

            }
            // Otherwise, if matrix connectivity is defined using a dense matrix, allocate user-defined weight model variables
            // **NOTE** if matrix is sparse, allocate later in the allocatesparsearrays function when we know the size of the network
            else if ((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
                const size_t size = s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumNeurons();

                for(const auto &v : wu->getVars()) {
                    mem += allocate_variable(os, v.second, v.first + s.first, s.second.getWUVarMode(v.first), size);
                }
            }

            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                const size_t size = s.second.getTrgNeuronGroup()->getNumNeurons();

                for(const auto &v : psm->getVars()) {
                    mem += allocate_variable(os, v.second, v.first + s.first, s.second.getPSVarMode(v.first), size);
                }
            }
            os << std::endl;
        }
    }
    os << std::endl;
    
    // ------------------------------------------------------------------------
    // allocating conductance arrays for sparse matrices

    for(const auto &s : model.getLocalSynapseGroups()) {
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::YALE) {
            os << "void allocate" << s.first << "(unsigned int connN)";
            {
                CodeStream::Scope b(os);
                os << "// Allocate host side variables" << std::endl;
                os << "C" << s.first << ".connN= connN;" << std::endl;

                // Allocate indices pointing to synapses in each presynaptic neuron's sparse matrix row
                allocate_host_variable(os, "unsigned int", "C" + s.first + ".indInG", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                    s.second.getSrcNeuronGroup()->getNumNeurons() + 1);

                // Allocate the postsynaptic neuron indices that make up sparse matrix
                allocate_host_variable(os, "unsigned int", "C" + s.first + ".ind", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                    "connN");

                if (model.isSynapseGroupDynamicsRequired(s.first)) {
                    allocate_host_variable(os, "unsigned int", "C" + s.first + ".preInd", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                        "connN");
                } else {
                    os << "C" << s.first << ".preInd= NULL;" << std::endl;
                }
                if (model.isSynapseGroupPostLearningRequired(s.first)) {
                    // Allocate indices pointing to synapses in each postsynaptic neuron's sparse matrix column
                    allocate_host_variable(os, "unsigned int", "C" + s.first + ".revIndInG", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                        s.second.getTrgNeuronGroup()->getNumNeurons() + 1);

                    // Allocate presynaptic neuron indices that make up postsynaptically indexed sparse matrix
                    allocate_host_variable(os, "unsigned int", "C" + s.first + ".revInd", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                        "connN");

                    // Allocate array mapping from postsynaptically to presynaptically indexed sparse matrix
                    allocate_host_variable(os, "unsigned int", "C" + s.first + ".remap", VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                        "connN");
                } else {
                    os << "C" << s.first << ".revIndInG= NULL;" << std::endl;
                    os << "C" << s.first << ".revInd= NULL;" << std::endl;
                    os << "C" << s.first << ".remap= NULL;" << std::endl;
                }

                const string numConnections = "C" + s.first + ".connN";

                allocate_device_variable(os, "unsigned int", "indInG" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                        s.second.getSrcNeuronGroup()->getNumNeurons() + 1);

                allocate_device_variable(os, "unsigned int", "ind" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                        numConnections);

                if (model.isSynapseGroupDynamicsRequired(s.first)) {
                    allocate_device_variable(os, "unsigned int", "preInd" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                            numConnections);
                }
                if (model.isSynapseGroupPostLearningRequired(s.first)) {
                    allocate_device_variable(os, "unsigned int", "revIndInG" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                            s.second.getTrgNeuronGroup()->getNumNeurons() + 1);
                    allocate_device_variable(os, "unsigned int", "revInd" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                            numConnections);
                    allocate_device_variable(os, "unsigned int", "remap" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST,
                                            numConnections);
                }

                // Allocate synapse variables
                if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                    for(const auto &v : s.second.getWUModel()->getVars()) {
                        allocate_variable(os, v.second, v.first + s.first, s.second.getWUVarMode(v.first), numConnections);
                    }
                }
            }
            os << std::endl;
            //setup up helper fn for this (specific) popn to generate sparse from dense
            os << "void createSparseConnectivityFromDense" << s.first << "(int preN,int postN, " << model.getPrecision() << " *denseMatrix)";
            {
                CodeStream::Scope b(os);
                os << "gennError(\"The function createSparseConnectivityFromDense" << s.first << "() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \\n Please use your own logic and use the general tools allocate" << s.first << "(), countEntriesAbove(), and setSparseConnectivityFromDense().\");" << std::endl;
            }
            os << std::endl;
        }
    }

    // ------------------------------------------------------------------------
    // freeing global memory structures

    os << "void freeMem()";
    {
        CodeStream::Scope b(os);
#ifndef CPU_ONLY
        if(model.isDeviceRNGRequired()) {
            free_device_variable(os, "rng", VarMode::LOC_DEVICE_INIT_DEVICE);
        }
#endif
        // FREE REMOTE NEURON VARIABLES
        for(const auto &n : model.getRemoteNeuronGroups()) {
            if(n.second.hasOutputToHost(localHostID)) {
                free_variable(os, "glbSpkCnt" + n.first, n.second.getSpikeVarMode());
                free_variable(os, "glbSpk" + n.first, n.second.getSpikeVarMode());
            }
        }

        // FREE LOCAL NEURON VARIABLES
        for(const auto &n : model.getLocalNeuronGroups()) {
            // Free spike buffer
            free_variable(os, "glbSpkCnt" + n.first, n.second.getSpikeVarMode());
            free_variable(os, "glbSpk" + n.first, n.second.getSpikeVarMode());

            // Free spike-like event buffer if allocated
            if (n.second.isSpikeEventRequired()) {
                free_variable(os, "glbSpkCntEvnt" + n.first, n.second.getSpikeEventVarMode());
                free_variable(os, "glbSpkEvnt" + n.first, n.second.getSpikeEventVarMode());
            }

            // Free last spike time buffer if allocated
            if (n.second.isSpikeTimeRequired()) {
                free_variable(os, "sT" + n.first, n.second.getSpikeTimeVarMode());
            }

#ifndef CPU_ONLY
            if(n.second.isSimRNGRequired()) {
                free_device_variable(os, "rng" + n.first, VarMode::LOC_DEVICE_INIT_DEVICE);
            }
#endif
            // Free neuron state variables
            for (auto const &v : n.second.getNeuronModel()->getVars()) {
                free_variable(os, v.first + n.first,
                            n.second.getVarMode(v.first));
            }
        }

        // FREE SYNAPSE VARIABLES
        for(const auto &s : model.getLocalSynapseGroups()) {
            free_variable(os, "inSyn" + s.first, s.second.getInSynVarMode());

            if(s.second.isDendriticDelayRequired()) {
                free_variable(os, "denDelay" + s.first, s.second.getDendriticDelayVarMode());
            }

            if (s.second.getMatrixType() & SynapseMatrixConnectivity::YALE) {
                os << "C" << s.first << ".connN= 0;" << std::endl;

                free_host_variable(os, "C" + s.first + ".indInG", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                free_device_variable(os, "indInG" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);

                free_host_variable(os, "C" + s.first + ".ind", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                free_device_variable(os, "ind" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);

                if (model.isSynapseGroupPostLearningRequired(s.first)) {
                    free_host_variable(os, "C" + s.first + ".revIndInG", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                    free_device_variable(os, "revIndInG" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);

                    free_host_variable(os, "C" + s.first + ".revInd", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                    free_device_variable(os, "revInd" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);

                    free_host_variable(os, "C" + s.first + ".remap", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                    free_device_variable(os, "remap" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);
                }

                if (model.isSynapseGroupDynamicsRequired(s.first)) {
                    free_host_variable(os, "C" + s.first + ".preInd", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                    free_device_variable(os, "preInd" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);
                }
            }
            else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                free_host_variable(os, "C" + s.first + ".rowLength", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                free_device_variable(os, "rowLength" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);

                free_host_variable(os, "C" + s.first + ".ind", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                free_device_variable(os, "ind" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);

                if (model.isSynapseGroupPostLearningRequired(s.first)) {
                    free_host_variable(os, "C" + s.first + ".colLength", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                    free_device_variable(os, "colLength" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);

                    free_host_variable(os, "C" + s.first + ".remap", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                    free_device_variable(os, "remap" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);
                }

                if (model.isSynapseGroupDynamicsRequired(s.first)) {
                    free_host_variable(os, "C" + s.first + ".synRemap", VarMode::LOC_HOST_DEVICE_INIT_HOST);
                    free_device_variable(os, "synRemap" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);
                }
            }
            else if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                free_variable(os, "gp" + s.first, VarMode::LOC_HOST_DEVICE_INIT_HOST);
            }

            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : s.second.getWUModel()->getVars()) {
                    free_variable(os, v.first + s.first, s.second.getWUVarMode(v.first));
                }
            }
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                for(const auto &v : s.second.getPSModel()->getVars()) {
                    free_variable(os, v.first + s.first, s.second.getPSVarMode(v.first));
                }
            }
        }
    }
    os << std::endl;

    // ------------------------------------------------------------------------
    //! \brief Method for cleaning up and resetting device while quitting GeNN

    os << "void exitGeNN()";
    {
        CodeStream::Scope b(os);
        os << "freeMem();" << std::endl;
#ifndef CPU_ONLY
        os << "cudaDeviceReset();" << std::endl;
#endif
#ifdef MPI_ENABLE
        os << "MPI_Finalize();" << std::endl;
        os << "printf(\"MPI finalized.\\n\");" << std::endl;
#endif
    }
    os << std::endl;

    // If model can be run on CPU
    if(model.canRunOnCPU()) {
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// the actual time stepping procedure (using CPU)" << std::endl;
        os << "void stepTimeCPU()";
        {
            CodeStream::Scope b(os);
            if (!model.getLocalSynapseGroups().empty()) {
                if (!model.getSynapseDynamicsGroups().empty()) {
                    if (model.isTimingEnabled()) os << "synDyn_timer.startTimer();" << std::endl;
                    os << "calcSynapseDynamicsCPU(t);" << std::endl;
                    if (model.isTimingEnabled()) {
                        os << "synDyn_timer.stopTimer();" << std::endl;
                        os << "synDyn_tme+= synDyn_timer.getElapsedTime();" << std::endl;
                    }
                }
                if (model.isTimingEnabled()) os << "synapse_timer.startTimer();" << std::endl;
                os << "calcSynapsesCPU(t);" << std::endl;
                if (model.isTimingEnabled()) {
                    os << "synapse_timer.stopTimer();" << std::endl;
                    os << "synapse_tme+= synapse_timer.getElapsedTime();"<< std::endl;
                }
                if (!model.getSynapsePostLearnGroups().empty()) {
                    if (model.isTimingEnabled()) os << "learning_timer.startTimer();" << std::endl;
                    os << "learnSynapsesPostHost(t);" << std::endl;
                    if (model.isTimingEnabled()) {
                        os << "learning_timer.stopTimer();" << std::endl;
                        os << "learning_tme+= learning_timer.getElapsedTime();" << std::endl;
                    }
                }
            }

            // Generate code to advance host-side spike queues
            genHostSpikeQueueAdvance(os, model, localHostID);

            if (model.isTimingEnabled()) os << "neuron_timer.startTimer();" << std::endl;
            os << "calcNeuronsCPU(t);" << std::endl;
            if (model.isTimingEnabled()) {
                os << "neuron_timer.stopTimer();" << std::endl;
                os << "neuron_tme+= neuron_timer.getElapsedTime();" << std::endl;
            }
            os << "iT++;" << std::endl;
            os << "t= iT*DT;" << std::endl;
        }
    }
    fs.close();


    // ------------------------------------------------------------------------
    // finish up

#ifndef CPU_ONLY
    cout << "Global memory required for core model: " << mem/1e6 << " MB. " << std::endl;
    cout << deviceProp[theDevice].totalGlobalMem << " for device " << theDevice << std::endl;
  
    if (0.5 * deviceProp[theDevice].totalGlobalMem < mem) {
        cout << "memory required for core model (" << mem/1e6;
        cout << "MB) is more than 50% of global memory on the chosen device";
        cout << "(" << deviceProp[theDevice].totalGlobalMem/1e6 << "MB)." << std::endl;
        cout << "Experience shows that this is UNLIKELY TO WORK ... " << std::endl;
    }
#endif
}


//----------------------------------------------------------------------------
/*!
  \brief A function to generate the code that simulates the model on the GPU

  The function generates functions that will spawn kernel grids onto the GPU (but not the actual kernel code which is generated in "genNeuronKernel()" and "genSynpaseKernel()"). Generated functions include "copyGToDevice()", "copyGFromDevice()", "copyStateToDevice()", "copyStateFromDevice()", "copySpikesFromDevice()", "copySpikeNFromDevice()" and "stepTimeGPU()". The last mentioned function is the function that will initialize the execution on the GPU in the generated simulation engine. All other generated functions are "convenience functions" to handle data transfer from and to the GPU.
*/
//----------------------------------------------------------------------------

#ifndef CPU_ONLY
void genRunnerGPU(const NNmodel &model, //!< Model description
                  const string &path,   //!< Path for code generation
                  int localHostID)      //!< ID of local host
{
    string name = model.getGeneratedCodePath(path, "runnerGPU.cc");
    ofstream fs;
    fs.open(name.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    writeHeader(os);

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file runnerGPU.cc" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing the host side code for a GPU simulator version." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;
    os << std::endl;
    int version;
    cudaRuntimeGetVersion(&version); 
    if ((deviceProp[theDevice].major < 6) || (version < 8000)){
        //os << "#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600" << std::endl;
        //os << "#else"<< std::endl;
        //os << "#if __CUDA_ARCH__ < 600" << std::endl;
        os << "// software version of atomic add for double precision" << std::endl;
        os << "__device__ double atomicAddSW(double* address, double val)";
        {
            CodeStream::Scope b(os);
            os << "unsigned long long int* address_as_ull = (unsigned long long int*)address;" << std::endl;
            os << "unsigned long long int old = *address_as_ull, assumed;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "assumed = old;" << std::endl;
                os << "old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));" << std::endl;
            }
            os << "while (assumed != old);" << std::endl;
            os << "return __longlong_as_double(old);" << std::endl;
        }
        os << std::endl;
    }

    if (deviceProp[theDevice].major < 2) {
        os << "// software version of atomic add for single precision float" << std::endl;
        os << "__device__ float atomicAddSW(float* address, float val)" << std::endl;
        {
            CodeStream::Scope b(os);
            os << "int* address_as_ull = (int*)address;" << std::endl;
            os << "int old = *address_as_ull, assumed;" << std::endl;
            os << "do";
            {
                CodeStream::Scope b(os);
                os << "assumed = old;" << std::endl;
                os << "old = atomicCAS(address_as_ull, assumed, __float_as_int(val + __int_as_float(assumed)));" << std::endl;
            }
            os << "while (assumed != old);" << std::endl;
            os << "return __int_as_float(old);" << std::endl;
        }
        os << std::endl;
    }

    os << "template<typename RNG>" << std::endl;
    os << "__device__ float exponentialDistFloat(RNG *rng)";
    {
        CodeStream::Scope b(os);
        os << "float a = 0.0f;" << std::endl;
        os << "while (true)";
        {
            CodeStream::Scope b(os);
            os << "float u = curand_uniform(rng);" << std::endl;
            os << "const float u0 = u;" << std::endl;
            os << "while (true)";
            {
                CodeStream::Scope b(os);
                os << "float uStar = curand_uniform(rng);" << std::endl;
                os << "if (u < uStar)";
                {
                    CodeStream::Scope b(os);
                    os << "return  a + u0;" << std::endl;
                }
                os << "u = curand_uniform(rng);" << std::endl;
                os << "if (u >= uStar)";
                {
                    CodeStream::Scope b(os);
                    os << "break;" << std::endl;
                }
            }
            os << "a += 1.0f;" << std::endl;
        }
    }
    os << std::endl;
    os << "template<typename RNG>" << std::endl;
    os << "__device__ double exponentialDistDouble(RNG *rng)";
    {
        CodeStream::Scope b(os);
        os << "double a = 0.0f;" << std::endl;
        os << "while (true)";
        {
            CodeStream::Scope b(os);
            os << "double u = curand_uniform_double(rng);" << std::endl;
            os << "const double u0 = u;" << std::endl;
            os << "while (true)";
            {
                CodeStream::Scope b(os);
                os << "double uStar = curand_uniform_double(rng);" << std::endl;
                os << "if (u < uStar)" << std::endl;
                {
                    CodeStream::Scope b(os);
                    os << "return  a + u0;" << std::endl;
                }
                os << "u = curand_uniform_double(rng);" << std::endl;
                os << "if (u >= uStar)" << std::endl;
                {
                    CodeStream::Scope b(os);
                    os << "break;" << std::endl;
                }
            }
            os << "a += 1.0;" << std::endl;
        }
    }
    os << std::endl;

    os << "#include \"neuronKrnl.cc\"" << std::endl;
    if (!model.getLocalSynapseGroups().empty()) {
        os << "#include \"synapseKrnl.cc\"" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying remote data to device" << std::endl << std::endl;
    // Loop through remote neuron groups
    for(const auto &n : model.getRemoteNeuronGroups()) {
        // If this neuron group has outputs to local host
        if(n.second.hasOutputToHost(localHostID)) {
            // Write spike pushing function
            os << "void push" << n.first << "SpikesToDevice(bool hostInitialisedOnly)";
            {
                CodeStream::Scope b(os);
                genPushSpikeCode(os, n.second, false);
            }

            // Write current spike pushing function
            genPushCurrentSpikeFunctions(os, n.second, false);
        }

    }

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things to device" << std::endl << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        // neuron state variables
        os << "void push" << n.first << "StateToDevice(bool hostInitialisedOnly)";
        {
            CodeStream::Scope b(os);
            for(const auto &v : n.second.getNeuronModel()->getVars()) {
                // only copy variables which aren't pointers (pointers don't transport between GPU and CPU)
                // and are present on both device and host.
                const VarMode varMode = n.second.getVarMode(v.first);
                if (v.second.find("*") == string::npos && canPushPullVar(varMode)){
                    const size_t size = n.second.isVarQueueRequired(v.first)
                        ? n.second.getNumNeurons() * n.second.getNumDelaySlots()
                        : n.second.getNumNeurons();
                    // If variable is initialised on device, only copy if hostInitialisedOnly isn't set
                    if(varMode & VarInit::DEVICE) {
                        os << "if(!hostInitialisedOnly)" << CodeStream::OB(1051);
                    }

                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << n.first;
                    os << ", " << v.first << n.first;
                    os << ", " << size << " * sizeof(" << v.second << "), cudaMemcpyHostToDevice));" << std::endl;

                    if(varMode & VarInit::DEVICE) {
                        os << CodeStream::CB(1051);
                    }
                }
            }
        }
        os << std::endl;

        // neuron spike variables
        os << "void push" << n.first << "SpikesToDevice(bool hostInitialisedOnly)";
        {
            CodeStream::Scope b(os);

            genPushSpikeCode(os, n.second, false);

            if (n.second.isSpikeEventRequired()) {
                os << "push" << n.first << "SpikeEventsToDevice(hostInitialisedOnly);" << std::endl;
            }

            const VarMode spikeTimeVarMode = n.second.getSpikeTimeVarMode();
            if (n.second.isSpikeTimeRequired() && canPushPullVar(spikeTimeVarMode)) {
                // If spikes times are initialised on device, only copy if hostInitialisedOnly isn't set
                if(spikeTimeVarMode & VarInit::DEVICE) {
                    os << "if(!hostInitialisedOnly)" << CodeStream::OB(1062);
                }

                size_t size = n.second.getNumNeurons() * n.second.getNumDelaySlots();
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_sT" << n.first;
                os << ", sT" << n.first;
                os << ", " << size << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;

                if(spikeTimeVarMode & VarInit::DEVICE) {
                    os << CodeStream::CB(1062);
                }
            }
        }
        os << std::endl;

        // neuron spike variables
        os << "void push" << n.first << "SpikeEventsToDevice(bool hostInitialisedOnly)";
        {
            CodeStream::Scope b(os);
            genPushSpikeCode(os, n.second, true);
        }
        os << std::endl;

        // Generate functions to push current spikes and spike events to device
        genPushCurrentSpikeFunctions(os, n.second, false);
        genPushCurrentSpikeFunctions(os, n.second, true);
    }
    // synapse variables
    for(const auto &s : model.getLocalSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        os << "void push" << s.first << "StateToDevice(bool hostInitialisedOnly)";
        {
            CodeStream::Scope b(os);

            const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
            const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) { // INDIVIDUALG
                if (s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                    os << "const size_t size = " << numSrcNeurons * numTrgNeurons << ";" << std::endl;
                }
                else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                    os << "const size_t size = " << numSrcNeurons * s.second.getMaxConnections() << ";" << std::endl;
                }
                else {
                    os << "const size_t size = C" << s.first << ".connN;" << std::endl;
                }

                for(const auto &v : wu->getVars()) {
                    const VarMode varMode = s.second.getWUVarMode(v.first);

                    // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                    if (v.second.find("*") == string::npos && canPushPullVar(varMode)) {
                        // If variable is initialised on device, only copy if hostInitialisedOnly isn't set
                        if(varMode & VarInit::DEVICE) {
                            os << "if(!hostInitialisedOnly)" << CodeStream::OB(1101);
                        }

                        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << s.first;
                        os << ", " << v.first << s.first;
                        os << ", size * sizeof(" << v.second << "), cudaMemcpyHostToDevice));" << std::endl;

                        if(varMode & VarInit::DEVICE) {
                            os << CodeStream::CB(1101);
                        }
                    }
                }
            }
            
            if(s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                for(const auto &v : psm->getVars()) {
                    const VarMode varMode = s.second.getPSVarMode(v.first);
                    // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                    if (v.second.find("*") == string::npos && canPushPullVar(varMode)) {
                        // If variable is initialised on device, only copy if hostInitialisedOnly isn't set
                        if(varMode & VarInit::DEVICE) {
                            os << "if(!hostInitialisedOnly)" << CodeStream::OB(1102);
                        }

                        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << s.first;
                        os << ", " << v.first << s.first;
                        os << ", " << numTrgNeurons << " * sizeof(" << v.second << "), cudaMemcpyHostToDevice));" << std::endl;

                        if(varMode & VarInit::DEVICE) {
                            os << CodeStream::CB(1102);
                        }
                    }
                }
            }
            
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                const size_t size = (numSrcNeurons * numTrgNeurons) / 32 + 1;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << s.first;
                os << ", gp" << s.first;
                os << ", " << size << " * sizeof(uint32_t), cudaMemcpyHostToDevice));" << std::endl;
            }

            // If synapse input variables can be pushed and pulled add copy code
            if(canPushPullVar(s.second.getInSynVarMode())) {
                // If variable is initialised on device, only copy if hostInitialisedOnly isn't set
                if(s.second.getInSynVarMode() & VarInit::DEVICE) {
                    os << "if(!hostInitialisedOnly)" << CodeStream::OB(1103);
                }
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyn" << s.first;
                os << ", inSyn" << s.first;
                os << ", " << numTrgNeurons << " * sizeof(" << model.getPrecision() << "), cudaMemcpyHostToDevice));" << std::endl;

                if(s.second.getInSynVarMode() & VarInit::DEVICE) {
                    os << CodeStream::CB(1103);
                }
            }

            // If dendritic delay variables can be pushed and pulled add copy code
            if(s.isDendriticDelayRequired() && canPushPullVar(s.second.getDendriticDelayVarMode())) {
                // If variable is initialised on device, only copy if hostInitialisedOnly isn't set
                if(s.second.getDendriticDelayVarMode() & VarInit::DEVICE) {
                    os << "if(!hostInitialisedOnly)" << CodeStream::OB(1104);
                }
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_denDelay" << s.first;
                os << ", denDelay" << s.first;
                os << ", " << s.second.getMaxDendriticDelaySlots() * numTrgNeurons << " * sizeof(" << model.getPrecision() << "), cudaMemcpyHostToDevice));" << std::endl;

                if(s.second.getDendriticDelayVarMode() & VarInit::DEVICE) {
                    os << CodeStream::CB(1104);
                }
            }
        }
        os << std::endl;
    }


    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things from device" << std::endl << std::endl;

    for(const auto &n : model.getLocalNeuronGroups()) {
        // neuron state variables
        os << "void pull" << n.first << "StateFromDevice()";
        {
            CodeStream::Scope b(os);
            for(const auto &v : n.second.getNeuronModel()->getVars()) {
                // only copy non-zero-copied, non-pointers. Pointers don't transport between GPU and CPU
                if (v.second.find("*") == string::npos && canPushPullVar(n.second.getVarMode(v.first))) {
                    const size_t size = n.second.isVarQueueRequired(v.first)
                        ? n.second.getNumNeurons() * n.second.getNumDelaySlots()
                        : n.second.getNumNeurons();

                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << v.first << n.first;
                    os << ", d_" << v.first << n.first;
                    os << ", " << size << " * sizeof(" << v.second << "), cudaMemcpyDeviceToHost));" << std::endl;
                }
            }
        }
        os << std::endl;

        // spike event variables
        os << "void pull" << n.first << "SpikeEventsFromDevice()";
        {
            CodeStream::Scope b(os);

            if (n.second.isSpikeEventRequired() && canPushPullVar(n.second.getSpikeEventVarMode())) {
                const size_t glbSpkCntEvntSize = n.second.getNumDelaySlots();
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
                os << ", d_glbSpkCntEvnt" << n.first;
                os << ", " << glbSpkCntEvntSize << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;

                const size_t glbSpkEvntSize = n.second.getNumNeurons() * n.second.getNumDelaySlots();
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvnt" << n.first;
                os << ", d_glbSpkEvnt" << n.first;
                os << ", " << glbSpkEvntSize << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }
        }
        os << std::endl;

        // neuron spike variables (including spike events)
        os << "void pull" << n.first << "SpikesFromDevice()";
        {
            CodeStream::Scope b(os);

            if(canPushPullVar(n.second.getSpikeVarMode())) {
                const size_t glbSpkCntSize = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << n.first;
                os << ", d_glbSpkCnt" << n.first;
                os << ", " << glbSpkCntSize << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;

                os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << n.first;
                os << ", d_glbSpk" << n.first;
                os << ", " << "glbSpkCnt" << n.first << " [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }

            if (n.second.isSpikeEventRequired()) {
                os << "pull" << n.first << "SpikeEventsFromDevice();" << std::endl;
            }
        }
        os << std::endl;

        // neuron spike times
        os << "void pull" << n.first << "SpikeTimesFromDevice()";
        {
            CodeStream::Scope b(os);

            os << "//Assumes that spike numbers are already copied back from the device" << std::endl;
            if (n.second.isSpikeTimeRequired() && canPushPullVar(n.second.getSpikeTimeVarMode())) {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(sT" << n.first;
                os << ", d_sT" << n.first;
                os << ", " << "glbSpkCnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }

        }
        os << std::endl;

        genPullCurrentSpikeFunctions(os, n.second, false);
        genPullCurrentSpikeFunctions(os, n.second, true);
    }

    // synapse variables
    for(const auto &s : model.getLocalSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
        const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();

        os << "void pull" << s.first << "StateFromDevice()";
        {
            CodeStream::Scope b(os);
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) { // INDIVIDUALG
                if (s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                    os << "size_t size = " << numSrcNeurons * numTrgNeurons << ";" << std::endl;
                }
                else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                    os << "size_t size = " << numSrcNeurons * s.second.getMaxConnections() << ";" << std::endl;
                }
                else {
                    os << "size_t size = C" << s.first << ".connN;" << std::endl;
                }

                for(const auto &v : wu->getVars()) {
                    // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                    if (v.second.find("*") == string::npos && canPushPullVar(s.second.getWUVarMode(v.first))) {
                        os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << v.first << s.first;
                        os << ", d_"  << v.first << s.first;
                        os << ", size * sizeof(" << v.second << "), cudaMemcpyDeviceToHost));" << std::endl;
                    }
                }
            }
            
            if(s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                for(const auto &v : psm->getVars()) {
                    // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                    if (v.second.find("*") == string::npos && canPushPullVar(s.second.getPSVarMode(v.first))) {
                        os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << v.first << s.first;
                        os << ", d_"  << v.first << s.first;
                        os << ", " << numTrgNeurons << " * sizeof(" << v.second << "), cudaMemcpyDeviceToHost));" << std::endl;
                    }
                }
            }
            
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                const size_t size = (numSrcNeurons * numTrgNeurons) / 32 + 1;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(gp" << s.first;
                os << ", d_gp" << s.first;
                os << ", " << size << " * sizeof(uint32_t), cudaMemcpyDeviceToHost));" << std::endl;
            }

            if(canPushPullVar(s.second.getInSynVarMode())) {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(inSyn" << s.first;
                os << ", d_inSyn" << s.first;
                os << ", " << numTrgNeurons << " * sizeof(" << model.getPrecision() << "), cudaMemcpyDeviceToHost));" << std::endl;
            }

            if(s.isDendriticDelayRequired() &&canPushPullVar(s.second.getDendriticDelayVarMode())) {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(denDelay" << s.first;
                os << ", d_denDelay" << s.first;
                os << ", " << s.second.getMaxDendriticDelaySlots() * numTrgNeurons << " * sizeof(" << model.getPrecision() << "), cudaMemcpyDeviceToHost));" << std::endl;
            }
        }
        os << std::endl;
    }


    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying values to device" << std::endl;
    os << "void copyStateToDevice(bool hostInitialisedOnly)";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getRemoteNeuronGroups()) {
            if(n.second.hasOutputToHost(localHostID)) {
                os << "push" << n.first << "SpikesToDevice(hostInitialisedOnly);" << std::endl;
            }
        }

        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "push" << n.first << "StateToDevice(hostInitialisedOnly);" << std::endl;
            os << "push" << n.first << "SpikesToDevice(hostInitialisedOnly);" << std::endl;
        }

        for(const auto &s : model.getLocalSynapseGroups()) {
            os << "push" << s.first << "StateToDevice(hostInitialisedOnly);" << std::endl;
        }
    }
    os << std::endl;
    
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes to device" << std::endl;
    os << "void copySpikesToDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "push" << n.first << "SpikesToDevice();" << std::endl;
        }
    }
   
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikes to device" << std::endl;
    os << "void copyCurrentSpikesToDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
        os << "push" << n.first << "CurrentSpikesToDevice();" << std::endl;
        }
    }
   
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spike events to device" << std::endl;
    os << "void copySpikeEventsToDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "push" << n.first << "SpikeEventsToDevice();" << std::endl;
        }
    }
   
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikes to device" << std::endl;
    os << "void copyCurrentSpikeEventsToDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
        os << "push" << n.first << "CurrentSpikeEventsToDevice();" << std::endl;
        }
    }

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying values from device" << std::endl;
    os << "void copyStateFromDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "pull" << n.first << "StateFromDevice();" << std::endl;
            os << "pull" << n.first << "SpikesFromDevice();" << std::endl;
        }

        for(const auto &s : model.getLocalSynapseGroups()) {
            os << "pull" << s.first << "StateFromDevice();" << std::endl;
        }
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes from device" << std::endl;
    os << "void copySpikesFromDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "pull" << n.first << "SpikesFromDevice();" << std::endl;
        }
    }
    os << std::endl;
    
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikes from device" << std::endl;
    os << "void copyCurrentSpikesFromDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "pull" << n.first << "CurrentSpikesFromDevice();" << std::endl;
        }
    }
    os << std::endl;
    
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying spike numbers from device (note, only use when only interested"<< std::endl;
    os << "// in spike numbers; copySpikesFromDevice() already includes this)" << std::endl;
    os << "void copySpikeNFromDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
            if(canPushPullVar(n.second.getSpikeVarMode())) {
                size_t size = (n.second.isTrueSpikeRequired() && n.second.isDelayRequired())
                    ? n.second.getNumDelaySlots() : 1;

                os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << n.first;
                os << ", d_glbSpkCnt" << n.first << ", " << size << "* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }
        }
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------"<< std::endl;
    os << "// global copying spikeEvents from device" << std::endl;
    os << "void copySpikeEventsFromDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "pull" << n.first << "SpikeEventsFromDevice();" << std::endl;
        }
    }
    os << std::endl;
    
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikeEvents from device" << std::endl;
    os << "void copyCurrentSpikeEventsFromDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "pull" << n.first << "CurrentSpikeEventsFromDevice();" << std::endl;
        }
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spike event numbers from device (note, only use when only interested" << std::endl;
    os << "// in spike numbers; copySpikeEventsFromDevice() already includes this)" << std::endl;
    os << "void copySpikeEventNFromDevice()";
    {
        CodeStream::Scope b(os);
        for(const auto &n : model.getLocalNeuronGroups()) {
            if (n.second.isSpikeEventRequired() && canPushPullVar(n.second.getSpikeEventVarMode())) {
                const size_t size = n.second.isDelayRequired() ? n.second.getNumDelaySlots() : 1;

                os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
                os << ", d_glbSpkCntEvnt" << n.first << ", " << size << "* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }
        }
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// the time stepping procedure (using GPU)" << std::endl;
    os << "void stepTimeGPU()";
    {
        CodeStream::Scope b(os);
        if (!model.getLocalSynapseGroups().empty()) {
            unsigned int synapseGridSz = model.getSynapseKernelGridSize();
            os << "//model.padSumSynapseTrgN[model.synapseGrpN - 1] is " << synapseGridSz << std::endl;
            synapseGridSz = synapseGridSz / synapseBlkSz;
            os << "dim3 sThreads(" << synapseBlkSz << ", 1);" << std::endl;
            os << "dim3 sGrid(" << synapseGridSz << ", 1);" << std::endl;
            os << std::endl;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
            const unsigned int learnGridSz = ceil((float)model.getSynapsePostLearnGridSize() / learnBlkSz);
            os << "dim3 lThreads(" << learnBlkSz << ", 1);" << std::endl;
            os << "dim3 lGrid(" << learnGridSz << ", 1);" << std::endl;
            os << std::endl;
        }

        if (!model.getSynapseDynamicsGroups().empty()) {
            const unsigned int synDynGridSz = ceil((float)model.getSynapseDynamicsGridSize() / synDynBlkSz);
            os << "dim3 sDThreads(" << synDynBlkSz << ", 1);" << std::endl;
            os << "dim3 sDGrid(" << synDynGridSz << ", 1);" << std::endl;
            os << std::endl;
        }

        const unsigned int neuronGridSz = ceil((float) model.getNeuronGridSize() / neuronBlkSz);
        os << "dim3 nThreads(" << neuronBlkSz << ", 1);" << std::endl;
        if (neuronGridSz < (unsigned int)deviceProp[theDevice].maxGridSize[1]) {
            os << "dim3 nGrid(" << neuronGridSz << ", 1);" << std::endl;
        }
        else {
            int sqGridSize = ceil((float) sqrt((float) neuronGridSz));
            os << "dim3 nGrid(" << sqGridSize << ","<< sqGridSize <<");" << std::endl;
        }
        os << std::endl;
        if (!model.getLocalSynapseGroups().empty()) {
            if (!model.getSynapseDynamicsGroups().empty()) {
                if (model.isTimingEnabled()) {
                    os << "cudaEventRecord(synDynStart);" << std::endl;
                }
                os << "calcSynapseDynamics <<< sDGrid, sDThreads >>> (";
                for(const auto &p : model.getSynapseDynamicsKernelParameters()) {
                    os << p.first << ", ";
                }
                os << "t);" << std::endl;
                if (model.isTimingEnabled()) {
                    os << "cudaEventRecord(synDynStop);" << std::endl;
                }
            }
            if (model.isTimingEnabled()) {
                os << "cudaEventRecord(synapseStart);" << std::endl;
            }
            os << "calcSynapses <<< sGrid, sThreads >>> (";
            for(const auto &p : model.getSynapseKernelParameters()) {
                os << p.first << ", ";
            }
            os << "t);" << std::endl;
            if (model.isTimingEnabled()) {
                os << "cudaEventRecord(synapseStop);" << std::endl;
            }

            if (!model.getSynapsePostLearnGroups().empty()) {
                if (model.isTimingEnabled()) {
                    os << "cudaEventRecord(learningStart);" << std::endl;
                }
                os << "learnSynapsesPost <<< lGrid, lThreads >>> (";
                for(const auto &p : model.getSimLearnPostKernelParameters()) {
                    os << p.first << ", ";
                }
                os << "t);" << std::endl;
                if (model.isTimingEnabled()) {
                    os << "cudaEventRecord(learningStop);" << std::endl;
                }
            }
        }
        // Generate code to advance host-side spike queues
        genHostSpikeQueueAdvance(os, model, localHostID);

        if (model.isTimingEnabled()) {
            os << "cudaEventRecord(neuronStart);" << std::endl;
        }

        os << "calcNeurons <<< nGrid, nThreads >>> (";
        for(const auto &p : model.getNeuronKernelParameters()) {
            os << p.first << ", ";
        }
        os << "t);" << std::endl;
        if (model.isTimingEnabled()) {
            os << "cudaEventRecord(neuronStop);" << std::endl;
            os << "cudaEventSynchronize(neuronStop);" << std::endl;
            os << "float tmp;" << std::endl;
            if (!model.getLocalSynapseGroups().empty()) {
                os << "cudaEventElapsedTime(&tmp, synapseStart, synapseStop);" << std::endl;
                os << "synapse_tme+= tmp/1000.0;" << std::endl;
            }
            if (!model.getSynapsePostLearnGroups().empty()) {
                os << "cudaEventElapsedTime(&tmp, learningStart, learningStop);" << std::endl;
                os << "learning_tme+= tmp/1000.0;" << std::endl;
            }
            if (!model.getSynapseDynamicsGroups().empty()) {
                os << "cudaEventElapsedTime(&tmp, synDynStart, synDynStop);" << std::endl;
                os << "synDyn_tme+= tmp/1000.0;" << std::endl;
            }
            os << "cudaEventElapsedTime(&tmp, neuronStart, neuronStop);" << std::endl;
            os << "neuron_tme+= tmp/1000.0;" << std::endl;
        }

        // Synchronise if zero-copy is in use
        if(model.zeroCopyInUse()) {
            os << "cudaDeviceSynchronize();" << std::endl;
        }

        os << "iT++;" << std::endl;
        os << "t= iT*DT;" << std::endl;
    }
    fs.close();
    //cout << "done with generating GPU runner" << std::endl;
}
#endif // CPU_ONLY


//----------------------------------------------------------------------------
/*!
  \brief A function that generates an MSBuild script for all generated GeNN code.
*/
//----------------------------------------------------------------------------
void genMSBuild(const NNmodel &model,   //!< Model description
                const string &path)     //!< Path for code generation
{
    string name = model.getGeneratedCodePath(path, "generated_code.props");
    ofstream fs;
    fs.open(name.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    os << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << endl;
    os << "<Project DefaultTargets=\"Build\" ToolsVersion=\"12.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">" << endl;
#ifdef CPU_ONLY
    os << "  <Import Project=\"$(GENN_PATH)\\userproject\\include\\genn_cpu_only.props\"/>" << endl;
    os << endl;
    os << "  <!-- Compile runner using C++ compiler -->" << endl;
    os << "  <ItemGroup>" << endl;
    os << "    <ClCompile Include=\"" << model.getName() + "_CODE\\runner.cc\"/>";
    os << "  </ItemGroup>" << endl;
#else
    os << "  <Import Project=\"$(GENN_PATH)\\userproject\\include\\genn.props\"/>" << endl;
    os << endl;
    const string computeCapability = to_string(deviceProp[theDevice].major) + to_string(deviceProp[theDevice].minor);
	os << "  <!-- Set CUDA code generation options based on selected device -->" << endl;
    os << "  <ItemDefinitionGroup>" << endl;
    os << "    <CudaCompile>" << endl;
    os << "      <CodeGeneration>compute_" << computeCapability <<",sm_" << computeCapability << "</CodeGeneration>" << endl;
    os << "    </CudaCompile>" << endl;
    os << "  </ItemDefinitionGroup>" << endl;
    os << "  <!-- Compile runner using CUDA compiler -->" << endl;
    os << "  <ItemGroup>" << endl;
    os << "    <CudaCompile Include=\"" << model.getName() + "_CODE\\runner.cc\">" << endl;
    // **YUCK** for some reasons you can't call .Contains on %(BaseCommandLineTemplate) directly
    // Solution suggested by https://stackoverflow.com/questions/9512577/using-item-functions-on-metadata-values
    os << "      <AdditionalOptions Condition=\" !$([System.String]::new('%(BaseCommandLineTemplate)').Contains('-x cu')) \">-x cu %(AdditionalOptions)</AdditionalOptions>" << endl;
    os << "    </CudaCompile>" << endl;
    os << "  </ItemGroup>" << endl;
#endif  // !CPU_ONLY
    os << "</Project>" << endl;
    
}

//----------------------------------------------------------------------------
/*!
\brief A function that generates the Makefile for all generated GeNN code.
*/
//----------------------------------------------------------------------------
void genMakefile(const NNmodel &model, //!< Model description
                 const string &path)  //!< Path for code generation
{
    string name = model.getGeneratedCodePath(path, "Makefile");
    ofstream fs;
    fs.open(name.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

#ifdef _WIN32
#ifdef CPU_ONLY
    string cxxFlags = GENN_PREFERENCES::buildSharedLibrary ? "/LD" : "/C";
    cxxFlags += " /DCPU_ONLY";
    cxxFlags += " " + GENN_PREFERENCES::userCxxFlagsWIN;
    if (GENN_PREFERENCES::optimizeCode) {
        cxxFlags += " /O2";
    }
    if (GENN_PREFERENCES::debugCode) {
        cxxFlags += " /debug /Zi /Od";
    }

    os << endl;
    os << "CXXFLAGS       =/nologo /EHsc " << cxxFlags << endl;
    os << endl;
    os << "INCLUDEFLAGS   =/I\"$(GENN_PATH)\\lib\\include\"" << endl;
    os << endl;
    
    // Add correct rules for building either shared library or object file
    // **NOTE** no idea how Visual C++ figures out that the dll should be called runner.dll but...it does
    // **NOTE** adding /OUT:runner.dll to make this explicit actually causes it to complain about ignoring options
    if(GENN_PREFERENCES::buildSharedLibrary) {
        os << "all: runner.dll" << endl;
        os << endl;
        os << "runner.dll: runner.cc" << endl;
        os << "\t$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) runner.cc $(GENN_PATH)\\lib\\src\\sparseUtils.cc" << endl;
        os << endl;
        os << "clean:" << endl;
        os << "\t-del runner.dll *.obj 2>nul" << endl;
    }
    else
    {
        os << "all: runner.obj" << endl;
        os << endl;
        os << "runner.obj: runner.cc" << endl;
        os << "\t$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) runner.cc" << endl;
        os << endl;
        os << "clean:" << endl;
        os << "\t-del runner.obj 2>nul" << endl;
    }
#else
    // Start with correct NVCC flags to build shared library or object file as appropriate
    // **NOTE** -c = compile and assemble, don't link
    string nvccFlags = GENN_PREFERENCES::buildSharedLibrary ? "--shared" : "-c";
    nvccFlags += " -x cu -arch sm_";
    nvccFlags += to_string(deviceProp[theDevice].major) + to_string(deviceProp[theDevice].minor);
    nvccFlags += " " + GENN_PREFERENCES::userNvccFlags;
    
    if (GENN_PREFERENCES::optimizeCode) {
        nvccFlags += " -O3 -use_fast_math";
    }
    if (GENN_PREFERENCES::debugCode) {
        nvccFlags += " -O0 -g -G";
    }
    if (GENN_PREFERENCES::showPtxInfo) {
        nvccFlags += " -Xptxas \"-v\"";
    }

    os << endl;
    os << "NVCC           =\"" << NVCC << "\"" << endl;
    os << "NVCCFLAGS      =" << nvccFlags << endl;
    os << endl;
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)\\lib\\include\"" << endl;
    os << endl;
    
    // Add correct rules for building either shared library or object file
    if(GENN_PREFERENCES::buildSharedLibrary) {
        os << "all: runner.dll" << endl;
        os << endl;
        os << "runner.dll: runner.cc" << endl;
        os << "\t$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc $(GENN_PATH)\\lib\\src\\sparseUtils.cc -o runner.dll" << endl;
        os << endl;
        os << "clean:" << endl;
        os << "\t-del runner.dll 2>nul" << endl;
    }
    else {
        os << "all: runner.obj" << endl;
        os << endl;
        os << "runner.obj: runner.cc" << endl;
        os << "\t$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc" << endl;
        os << endl;
        os << "clean:" << endl;
        os << "\t-del runner.obj 2>nul" << endl;
    }
#endif

#else // UNIX

#ifdef CPU_ONLY
    // Start with correct NVCC flags to build shared library or object file as appropriate
    // **NOTE** -c = compile and assemble, don't link
    string cxxFlags = GENN_PREFERENCES::buildSharedLibrary ? "-shared -fPIC" : "-c";
    cxxFlags += " -DCPU_ONLY -std=c++11";
    cxxFlags += " " + GENN_PREFERENCES::userCxxFlagsGNU;
    if (GENN_PREFERENCES::optimizeCode) {
        cxxFlags += " -O3 -ffast-math";
    }
    if (GENN_PREFERENCES::debugCode) {
        cxxFlags += " -O0 -g";
    }

    os << endl;
    os << "CXXFLAGS       :=" << cxxFlags << endl;
    os << endl;
#ifdef MPI_ENABLE
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)/lib/include\" -I\"$(MPI_PATH)/include\"" << endl;
#else
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)/lib/include\"" << endl;
#endif
    os << endl;

    // Add correct rules for building either shared library or object file
    if(GENN_PREFERENCES::buildSharedLibrary) {
        os << "all: librunner.so" << endl;
        os << endl;
        os << "librunner.so: runner.cc" << endl;
        os << "\t$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) runner.cc $(GENN_PATH)/lib/src/sparseUtils.cc -o librunner.so" << endl;
        os << endl;
        os << "clean:" << endl;
        os << "\trm -f librunner.so" << endl;
    }
    else {
        os << "all: runner.o" << endl;
        os << endl;
        os << "runner.o: runner.cc" << endl;
        os << "\t$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) runner.cc" << endl;
        os << endl;
        os << "clean:" << endl;
        os << "\trm -f runner.o" << endl;
    }
#else
    // Start with correct NVCC flags to build shared library or object file as appropriate
    // **NOTE** -c = compile and assemble, don't link
    string nvccFlags = GENN_PREFERENCES::buildSharedLibrary ? "--shared --compiler-options '-fPIC'" : "-c";

    nvccFlags += " -std=c++11 -x cu -arch sm_";
    nvccFlags += to_string(deviceProp[theDevice].major) + to_string(deviceProp[theDevice].minor);
    nvccFlags += " " + GENN_PREFERENCES::userNvccFlags;
    if (GENN_PREFERENCES::optimizeCode) {
        nvccFlags += " -O3 -use_fast_math -Xcompiler \"-ffast-math\"";
    }
    if (GENN_PREFERENCES::debugCode) {
        nvccFlags += " -O0 -g -G";
    }
    if (GENN_PREFERENCES::showPtxInfo) {
        nvccFlags += " -Xptxas \"-v\"";
    }

    os << endl;
    os << "NVCC           :=\"" << NVCC << "\"" << endl;
    os << "NVCCFLAGS      :=" << nvccFlags << endl;
    os << endl;
#ifdef MPI_ENABLE
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)/lib/include\" -I\"$(MPI_PATH)/include\"" << endl;
#else
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)/lib/include\"" << endl;
#endif
    os << endl;

    // Add correct rules for building either shared library or object file
    if(GENN_PREFERENCES::buildSharedLibrary) {
        os << "all: librunner.so" << endl;
        os << endl;
        os << "librunner.so: runner.cc" << endl;
        os << "\t$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc $(GENN_PATH)/lib/src/sparseUtils.cc -o librunner.so" << endl;
        os << endl;
        os << "clean:" << endl;
        os << "\trm -f librunner.so" << endl;
    }
    else {
        os << "all: runner.o" << endl;
        os << endl;
        os << "runner.o: runner.cc" << endl;
        os << "\t$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc" << endl;
        os << endl;
        os << "clean:" << endl;
        os << "\trm -f runner.o" << endl;
    }
#endif

#endif

    fs.close();
}
