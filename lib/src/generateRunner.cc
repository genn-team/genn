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

void variable_def(CodeStream &os, const string &type, const string &name)
{
    os << type << " " << name << ";" << std::endl;
#ifndef CPU_ONLY
    os << type << " d_" << name << ";" << std::endl;
    os << "__device__ " << type << " dd_" << name << ";" << std::endl;
#endif
}


//--------------------------------------------------------------------------
//! \brief This function generates host extern variable definitions, of the given type and name.
//--------------------------------------------------------------------------

void extern_variable_def(CodeStream &os, const string &type, const string &name)
{
    // In windows making variables extern isn't enough to export then as DLL symbols - you need to add __declspec(dllexport)
#ifdef _WIN32
    const std::string varExportPrefix = GENN_PREFERENCES::buildSharedLibrary ? "__declspec(dllexport) extern" : "extern";
#else
    const std::string varExportPrefix = "extern";
#endif

    os << varExportPrefix << " " << type << " " << name << ";" << std::endl;
#ifndef CPU_ONLY
    os << varExportPrefix << " " << type << " d_" << name << ";" << std::endl;
#endif
}

//--------------------------------------------------------------------------
//! \brief This function generates host allocation code
//--------------------------------------------------------------------------

void allocate_host_variable(CodeStream &os, const string &type, const string &name, bool zeroCopy, const string &size)
{
#ifndef CPU_ONLY
    const char *flags = zeroCopy ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
    os << "    cudaHostAlloc(&" << name << ", " << size << " * sizeof(" << type << "), " << flags << ");" << std::endl;
#else
    USE(zeroCopy);

    os << "    " << name << " = new " << type << "[" << size << "];" << std::endl;
#endif
}

void allocate_host_variable(CodeStream &os, const string &type, const string &name, bool zeroCopy, size_t size)
{
    allocate_host_variable(os, type, name, zeroCopy, to_string(size));
}

void allocate_device_variable(CodeStream &os, const string &type, const string &name, bool zeroCopy, const string &size)
{
#ifndef CPU_ONLY
    // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
    if(zeroCopy) {
        os << "    deviceZeroCopy(" << name << ", &d_" << name << ", dd_" << name << ");" << std::endl;
    }
    else {
        os << "    deviceMemAllocate(&d_" << name << ", dd_" << name << ", " << size << " * sizeof(" << type << "));" << std::endl;
    }
#else
    USE(os);
    USE(type);
    USE(name);
    USE(zeroCopy);
    USE(size);
#endif
}

void allocate_device_variable(CodeStream &os, const string &type, const string &name, bool zeroCopy, size_t size)
{
    allocate_device_variable(os, type, name, zeroCopy, to_string(size));
}

//--------------------------------------------------------------------------
//! \brief This function generates host and device allocation with standard names (name, d_name, dd_name) and estimates size based on size known at generate-time
//--------------------------------------------------------------------------
unsigned int allocate_variable(CodeStream &os, const string &type, const string &name, bool zeroCopy, size_t size)
{
    // Allocate host and device variables
    allocate_host_variable(os, type, name, zeroCopy, size);
    allocate_device_variable(os, type, name, zeroCopy, size);

    // Return size estimate
    return size * theSize(type);
}

void allocate_variable(CodeStream &os, const string &type, const string &name, bool zeroCopy, const string &size)
{
    // Allocate host and device variables
    allocate_host_variable(os, type, name, zeroCopy, size);
    allocate_device_variable(os, type, name, zeroCopy, size);
}

void free_host_variable(CodeStream &os, const string &name)
{
#ifndef CPU_ONLY
    os << "    CHECK_CUDA_ERRORS(cudaFreeHost(" << name << "));" << std::endl;
#else
    os << "    delete[] " << name << ";" << std::endl;
#endif
}

void free_device_variable(CodeStream &os, const string &name, bool zeroCopy)
{
#ifndef CPU_ONLY
    // If this variable wasn't allocated in zero-copy mode, free it
    if(!zeroCopy) {
        os << "    CHECK_CUDA_ERRORS(cudaFree(d_" << name << "));" << std::endl;
    }
#else
    USE(os);
    USE(name);
    USE(zeroCopy);
#endif
}

//--------------------------------------------------------------------------
//! \brief This function generates code to free host and device allocations with standard names (name, d_name, dd_name)
//--------------------------------------------------------------------------
void free_variable(CodeStream &os, const string &name, bool zeroCopy)
{
    free_host_variable(os, name);
    free_device_variable(os, name, zeroCopy);
}
}

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
void genDefinitions(const NNmodel &model, //!< Model description
               const string &path) //!< Path for code generationn
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
    string definitionsName= path + "/" + model.getName() + "_CODE/definitions.h";
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
    os << "#include \"sparseUtils.h\"" << std::endl << std::endl;
    os << "#include \"sparseProjection.h\"" << std::endl;
    // **YUCK** because code is, by default, not build with C++11 the <cstdint> header is not present
    os << "#include <stdint.h>" << std::endl;
    if (model.isRNGRequired()) {
        os << "#include <random>" << std::endl;
#ifndef CPU_ONLY
        os << "#include <curand_kernel.h>" << std::endl;
#endif
    }
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
        if (!model.getSynapseGroups().empty()) {
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
    }
    os << std::endl;
    if(model.isRNGRequired()) {
        os << "extern std::mt19937 rng;" << std::endl;
    }
    os << std::endl;

    //---------------------------------
    // HOST AND DEVICE NEURON VARIABLES

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// neuron variables" << std::endl;
    os << std::endl;

    for(const auto &n : model.getNeuronGroups()) {
        extern_variable_def(os, "unsigned int *", "glbSpkCnt"+n.first);
        extern_variable_def(os, "unsigned int *", "glbSpk"+n.first);
        if (n.second.isSpikeEventRequired()) {
            extern_variable_def(os, "unsigned int *", "glbSpkCntEvnt"+n.first);
            extern_variable_def(os, "unsigned int *", "glbSpkEvnt"+n.first);
        }
        if (n.second.isDelayRequired()) {
            os << varExportPrefix << " unsigned int spkQuePtr" << n.first << ";" << std::endl;
        }
        if (n.second.isSpikeTimeRequired()) {
            extern_variable_def(os, model.getPrecision()+" *", "sT"+n.first);
        }
#ifndef CPU_ONLY
        if(n.second.isSimRNGRequired()) {
            os << "extern curandState *d_rng" << n.first << ";" << std::endl;
        }
#endif
        auto neuronModel = n.second.getNeuronModel();
        for(auto const &v : neuronModel->getVars()) {
            extern_variable_def(os, v.second +" *", v.first + n.first);
        }
        for(auto const &v : neuronModel->getExtraGlobalParams()) {
            extern_variable_def(os, v.second, v.first + n.first);
        }
    }
    os << std::endl;
    for(auto &n : model.getNeuronGroups()) {
        os << "#define glbSpkShift" << n.first;
        if (n.second.isDelayRequired()) {
            os << " spkQuePtr" << n.first << "*" << n.second.getNumNeurons();
        }
        else {
            os << " 0";
        }
        os << std::endl;
    }

    for(const auto &n : model.getNeuronGroups()) {
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

    for(const auto &s : model.getSynapseGroups()) {
        extern_variable_def(os, model.getPrecision()+" *", "inSyn" + s.first);
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            extern_variable_def(os, "uint32_t *", "gp" + s.first);
        }
        else if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << varExportPrefix << " SparseProjection C" << s.first << ";" << std::endl;
        }

        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) { // not needed for GLOBALG
            for(const auto &v : s.second.getWUModel()->getVars()) {
                extern_variable_def(os, v.second + " *", v.first + s.first);
            }
            for(const auto &v : s.second.getPSModel()->getVars()) {
                extern_variable_def(os, v.second + " *", v.first + s.first);
            }
        }

        for(auto const &p : s.second.getWUModel()->getExtraGlobalParams()) {
            extern_variable_def(os, p.second, p.first + s.first);
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
    os << "// copying things to device" << std::endl;
    os << std::endl;
    for(const auto &n : model.getNeuronGroups()) {
        os << funcExportPrefix << "void push" << n.first << "StateToDevice();" << std::endl;
        os << funcExportPrefix << "void push" << n.first << "SpikesToDevice();" << std::endl;
        os << funcExportPrefix << "void push" << n.first << "SpikeEventsToDevice();" << std::endl;
        os << funcExportPrefix << "void push" << n.first << "CurrentSpikesToDevice();" << std::endl;
        os << funcExportPrefix << "void push" << n.first << "CurrentSpikeEventsToDevice();" << std::endl;
    }
    for(const auto &s : model.getSynapseGroups()) {
        os << "#define push" << s.first << "ToDevice push" << s.first << "StateToDevice" << std::endl;
        os << funcExportPrefix << "void push" << s.first << "StateToDevice();" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things from device" << std::endl;
    os << std::endl;
    for(const auto &n : model.getNeuronGroups()) {
        os << funcExportPrefix << "void pull" << n.first << "StateFromDevice();" << std::endl;
        os << funcExportPrefix << "void pull" << n.first << "SpikesFromDevice();" << std::endl;
        os << funcExportPrefix << "void pull" << n.first << "SpikeEventsFromDevice();" << std::endl;
        os << funcExportPrefix << "void pull" << n.first << "CurrentSpikesFromDevice();" << std::endl;
        os << funcExportPrefix << "void pull" << n.first << "CurrentSpikeEventsFromDevice();" << std::endl;
    }
    for(const auto &s : model.getSynapseGroups()) {
        os << "#define pull" << s.first << "FromDevice pull" << s.first << "StateFromDevice" << std::endl;
        os << funcExportPrefix << "void pull" << s.first << "StateFromDevice();" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying values to device" << std::endl;
    os << std::endl;
    os << funcExportPrefix << "void copyStateToDevice();" << std::endl;
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

    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
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


#ifndef CPU_ONLY
    os << funcExportPrefix << "void initializeAllSparseArrays();" << std::endl;
    os << std::endl;
#endif

    os << "// --------------3200----------------------------------------------------------" << std::endl;
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
    os << "void stepTimeGPU(T arg1, ...)" << CodeStream::OB(101);
    os << "gennError(\"Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.\");" << std::endl;
    os << CodeStream::CB(101);
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper function for allocating memory blocks on the GPU device" << std::endl;
    os << std::endl;
    os << "template<class T>" << std::endl;
    os << "void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)" << std::endl;
    os << "{" << std::endl;
    os << "    void *devptr;" << std::endl;
    os << "    CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));" << std::endl;
    os << "    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));" << std::endl;
    os << "    CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));" << std::endl;
    os << "}" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper function for getting the device pointer corresponding to a zero-copied host pointer and assigning it to a symbol" << std::endl;
    os << std::endl;
    os << "template<class T>" << std::endl;
    os << "void deviceZeroCopy(T hostPtr, const T *devPtr, const T &devSymbol)" << std::endl;
    os << "{" << std::endl;
    os << "    CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void **)devPtr, (void*)hostPtr, 0));" << std::endl;
    os << "    void *devSymbolPtr;" << std::endl;
    os << "    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devSymbolPtr, devSymbol));" << std::endl;
    os << "    CHECK_CUDA_ERRORS(cudaMemcpy(devSymbolPtr, devPtr, sizeof(void*), cudaMemcpyHostToDevice));" << std::endl;
    os << "}" << std::endl;
    os << std::endl;
#endif

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Throw an error for \"old style\" time stepping calls (using CPU)" << std::endl;
    os << std::endl;
    os << "template <class T>" << std::endl;
    os << "void stepTimeCPU(T arg1, ...)" << CodeStream::OB(101);
    os << "gennError(\"Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.\");" << std::endl;
    os << CodeStream::CB(101);
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

    string supportCodeName= path + "/" + model.getName() + "_CODE/support_code.h";
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
    for(const auto &n : model.getNeuronGroups()) {
        if (!n.second.getNeuronModel()->getSupportCode().empty()) {
            os << "namespace " << n.first << "_neuron" << CodeStream::OB(11) << std::endl;
            os << ensureFtype(n.second.getNeuronModel()->getSupportCode(), model.getPrecision()) << std::endl;
            os << CodeStream::CB(11) << " // end of support code namespace " << n.first << std::endl;
        }
    }
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        if (!wu->getSimSupportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_simCode " << CodeStream::OB(11) << std::endl;
            os << ensureFtype(wu->getSimSupportCode(), model.getPrecision()) << std::endl;
            os << CodeStream::CB(11) << " // end of support code namespace " << s.first << "_weightupdate_simCode " << std::endl;
        }
        if (!wu->getLearnPostSupportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_simLearnPost " << CodeStream::OB(11) << std::endl;
            os << ensureFtype(wu->getLearnPostSupportCode(), model.getPrecision()) << std::endl;
            os << CodeStream::CB(11) << " // end of support code namespace " << s.first << "_weightupdate_simLearnPost " << std::endl;
        }
        if (!wu->getSynapseDynamicsSuppportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_synapseDynamics " << CodeStream::OB(11) << std::endl;
            os << ensureFtype(wu->getSynapseDynamicsSuppportCode(), model.getPrecision()) << std::endl;
            os << CodeStream::CB(11) << " // end of support code namespace " << s.first << "_weightupdate_synapseDynamics " << std::endl;
        }
        if (!psm->getSupportCode().empty()) {
            os << "namespace " << s.first << "_postsyn " << CodeStream::OB(11) << std::endl;
            os << ensureFtype(psm->getSupportCode(), model.getPrecision()) << std::endl;
            os << CodeStream::CB(11) << " // end of support code namespace " << s.first << "_postsyn " << std::endl;
        }

    }
    os << "#endif" << std::endl;
    fs.close();
}

void genRunner(const NNmodel &model, //!< Model description
               const string &path) //!< Path for code generationn

{
    //cout << "entering genRunner" << std::endl;
    string runnerName= path + "/" + model.getName() + "_CODE/runner.cc";
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
        if (!model.getSynapseGroups().empty()) {
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
    } 
    os << std::endl;
    if(model.isRNGRequired()) {
        os << "std::mt19937 rng;" << std::endl;
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
    for(const auto &n : model.getNeuronGroups()) {
        variable_def(os, "unsigned int *", "glbSpkCnt"+n.first);
        variable_def(os, "unsigned int *", "glbSpk"+n.first);
        if (n.second.isSpikeEventRequired()) {
            variable_def(os, "unsigned int *", "glbSpkCntEvnt"+n.first);
            variable_def(os, "unsigned int *", "glbSpkEvnt"+n.first);
        }
        if (n.second.isDelayRequired()) {
            os << "unsigned int spkQuePtr" << n.first << ";" << std::endl;
#ifndef CPU_ONLY
            os << "__device__ volatile unsigned int dd_spkQuePtr" << n.first << ";" << std::endl;
#endif
        }
        if (n.second.isSpikeTimeRequired()) {
            variable_def(os, model.getPrecision()+" *", "sT"+n.first);
        }
#ifndef CPU_ONLY
        if(n.second.isSimRNGRequired()) {
            os << "curandState *d_rng" << n.first << ";" << std::endl;
            os << "__device__ curandState *dd_rng" << n.first << ";" << std::endl;
        }
#endif

        auto neuronModel = n.second.getNeuronModel();
        for(auto const &v : neuronModel->getVars()) {
            variable_def(os, v.second + " *", v.first + n.first);
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

   for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        variable_def(os, model.getPrecision()+" *", "inSyn"+s.first);
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            variable_def(os, "uint32_t *", "gp"+s.first);
        }
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
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
        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) { // not needed for GLOBALG, INDIVIDUALID
            for(const auto &v : wu->getVars()) {
                variable_def(os, v.second + " *", v.first + s.first);
            }
            for(const auto &v : psm->getVars()) {
                variable_def(os, v.second+" *", v.first + s.first);
            }
        }

        for(const auto &v : wu->getExtraGlobalParams()) {
            os << v.second << " " <<  v.first<< s.first << ";" << std::endl;
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

    os << "void convertProbabilityToRandomNumberThreshold(" << model.getPrecision() << " *p_pattern, " << model.getRNType() << " *pattern, int N)" << std::endl;
    os << "{" << std::endl;
    os << "    " << model.getPrecision() << " fac= pow(2.0, (double) sizeof(" << model.getRNType() << ")*8-16);" << std::endl;
    os << "    for (int i= 0; i < N; i++) {" << std::endl;
    //os << "        assert(p_pattern[i] <= 1.0);" << std::endl;
    os << "        pattern[i]= (" << model.getRNType() << ") (p_pattern[i]*fac);" << std::endl;
    os << "    }" << std::endl;
    os << "}" << std::endl << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\brief Function to convert a firing rate (in kHz) " << std::endl;
    os << "to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given rate." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    os << "void convertRateToRandomNumberThreshold(" << model.getPrecision() << " *rateKHz_pattern, " << model.getRNType() << " *pattern, int N)" << std::endl;
    os << "{" << std::endl;
    os << "    " << model.getPrecision() << " fac= pow(2.0, (double) sizeof(" << model.getRNType() << ")*8-16)*DT;" << std::endl;
    os << "    for (int i= 0; i < N; i++) {" << std::endl;
    //os << "        assert(rateKHz_pattern[i] <= 1.0);" << std::endl;
    os << "        pattern[i]= (" << model.getRNType() << ") (rateKHz_pattern[i]*fac);" << std::endl;
    os << "    }" << std::endl;
    os << "}" << std::endl << std::endl;

    // include simulation kernels
#ifndef CPU_ONLY
    os << "#include \"runnerGPU.cc\"" << std::endl << std::endl;
#endif
    os << "#include \"init.cc\"" << std::endl;
    os << "#include \"neuronFnct.cc\"" << std::endl;
    if (!model.getSynapseGroups().empty()) {
        os << "#include \"synapseFnct.cc\"" << std::endl;
    }


    // ---------------------------------------------------------------------
    // Function for setting the CUDA device and the host's global variables.
    // Also estimates memory usage on device ...
  
    os << "void allocateMem()" << std::endl;
    os << "{" << std::endl;
#ifndef CPU_ONLY
    os << "    CHECK_CUDA_ERRORS(cudaSetDevice(" << theDevice << "));" << std::endl;

    // If the model requires zero-copy
    if(model.zeroCopyInUse())
    {
        // If device doesn't support mapping host memory error
        if(!deviceProp[theDevice].canMapHostMemory) {
            gennError("Device does not support mapping CPU host memory!");
        }

        // set appropriate device flags
        os << "    CHECK_CUDA_ERRORS(cudaSetDeviceFlags(cudaDeviceMapHost));" << std::endl;
    }
#endif
    //cout << "model.neuronGroupN " << model.neuronGrpN << std::endl;
    //os << "    " << model.getPrecision() << " free_m, total_m;" << std::endl;
    //os << "    cudaMemGetInfo((size_t*) &free_m, (size_t*) &total_m);" << std::endl;

    if (model.isTimingEnabled()) {
#ifndef CPU_ONLY
        os << "    cudaEventCreate(&neuronStart);" << std::endl;
        os << "    cudaEventCreate(&neuronStop);" << std::endl;
#endif
        os << "    neuron_tme= 0.0;" << std::endl;
        if (!model.getSynapseGroups().empty()) {
#ifndef CPU_ONLY
            os << "    cudaEventCreate(&synapseStart);" << std::endl;
            os << "    cudaEventCreate(&synapseStop);" << std::endl;
#endif
            os << "    synapse_tme= 0.0;" << std::endl;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
#ifndef CPU_ONLY
            os << "    cudaEventCreate(&learningStart);" << std::endl;
            os << "    cudaEventCreate(&learningStop);" << std::endl;
#endif
            os << "    learning_tme= 0.0;" << std::endl;
        }
        if (!model.getSynapseDynamicsGroups().empty()) {
#ifndef CPU_ONLY
            os << "    cudaEventCreate(&synDynStart);" << std::endl;
            os << "    cudaEventCreate(&synDynStop);" << std::endl;
#endif
            os << "    synDyn_tme= 0.0;" << std::endl;
        }
    }

    // ALLOCATE NEURON VARIABLES
    unsigned int mem = 0;
    for(const auto &n : model.getNeuronGroups()) {
        // Allocate population spike count
        mem += allocate_variable(os, "unsigned int", "glbSpkCnt" + n.first, n.second.isSpikeZeroCopyEnabled(),
                                 n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1);

        // Allocate population spike output buffer
        mem += allocate_variable(os, "unsigned int", "glbSpk" + n.first, n.second.isSpikeZeroCopyEnabled(),
                                 n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());


        if (n.second.isSpikeEventRequired()) {
            // Allocate population spike-like event counters
            mem += allocate_variable(os, "unsigned int", "glbSpkCntEvnt" + n.first, n.second.isSpikeEventZeroCopyEnabled(),
                                     n.second.getNumDelaySlots());

            // Allocate population spike-like event output buffer
            mem += allocate_variable(os, "unsigned int", "glbSpkEvnt" + n.first, n.second.isSpikeEventZeroCopyEnabled(),
                                     n.second.getNumNeurons() * n.second.getNumDelaySlots());
        }

        // Allocate buffer to hold last spike times if required
        if (n.second.isSpikeTimeRequired()) {
            mem += allocate_variable(os, model.getPrecision(), "sT" + n.first, n.second.isSpikeTimeZeroCopyEnabled(),
                                     n.second.getNumNeurons() * n.second.getNumDelaySlots());
        }

#ifndef CPU_ONLY
        if(n.second.isSimRNGRequired()) {
            allocate_device_variable(os, "curandState", "rng" + n.first, false,
                                     n.second.getNumNeurons());
        }
#endif  // CPU_ONLY

        // Allocate memory for neuron model's state variables
        for(const auto &v : n.second.getNeuronModel()->getVars()) {
            mem += allocate_variable(os, v.second, v.first + n.first, n.second.isVarZeroCopyEnabled(v.first),
                                     n.second.isVarQueueRequired(v.first) ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());
        }
        os << std::endl;
    }

    // ALLOCATE SYNAPSE VARIABLES
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        // Allocate buffer to hold input coming from this synapse population
        mem += allocate_variable(os, model.getPrecision(), "inSyn" + s.first, false,
                                 s.second.getTrgNeuronGroup()->getNumNeurons());

        // If connectivity is defined using a bitmask, allocate memory for bitmask
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t gpSize = (s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumNeurons()) / 32 + 1;
            mem += allocate_variable(os, "uint32_t", "gp" + s.first, false,
                                     gpSize);
        }
        // Otherwise, if matrix connectivity is defined using a dense matrix, allocate user-defined weight model variables
        // **NOTE** if matrix is sparse, allocate later in the allocatesparsearrays function when we know the size of the network
        else if ((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
            const size_t size = s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumNeurons();

            for(const auto &v : wu->getVars()) {
                mem += allocate_variable(os, v.second, v.first + s.first, s.second.isWUVarZeroCopyEnabled(v.first),
                                         size);
            }
        }

        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) { // not needed for GLOBALG
            const size_t size = s.second.getTrgNeuronGroup()->getNumNeurons();

            for(const auto &v : psm->getVars()) {
                mem += allocate_variable(os, v.second, v.first + s.first, s.second.isPSVarZeroCopyEnabled(v.first),
                                         size);
            }
        }
        os << std::endl;
    }
    os << "}" << std::endl << std::endl;

    // ------------------------------------------------------------------------
    // allocating conductance arrays for sparse matrices

    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << "void allocate" << s.first << "(unsigned int connN)" << "{" << std::endl;
            os << "// Allocate host side variables" << std::endl;
            os << "  C" << s.first << ".connN= connN;" << std::endl;

            // Allocate indices pointing to synapses in each presynaptic neuron's sparse matrix row
            allocate_host_variable(os, "unsigned int", "C" + s.first + ".indInG", false,
                                   s.second.getSrcNeuronGroup()->getNumNeurons() + 1);

            // Allocate the postsynaptic neuron indices that make up sparse matrix
            allocate_host_variable(os, "unsigned int", "C" + s.first + ".ind", false,
                                   "connN");

            if (model.isSynapseGroupDynamicsRequired(s.first)) {
                allocate_host_variable(os, "unsigned int", "C" + s.first + ".preInd", false,
                                       "connN");
            } else {
                os << "  C" << s.first << ".preInd= NULL;" << std::endl;
            }
            if (model.isSynapseGroupPostLearningRequired(s.first)) {
                // Allocate indices pointing to synapses in each postsynaptic neuron's sparse matrix column
                allocate_host_variable(os, "unsigned int", "C" + s.first + ".revIndInG", false,
                                       s.second.getTrgNeuronGroup()->getNumNeurons() + 1);

                // Allocate presynaptic neuron indices that make up postsynaptically indexed sparse matrix
                allocate_host_variable(os, "unsigned int", "C" + s.first + ".revInd", false,
                                       "connN");

                // Allocate array mapping from postsynaptically to presynaptically indexed sparse matrix
                allocate_host_variable(os, "unsigned int", "C" + s.first + ".remap", false,
                                       "connN");
            } else {
                os << "  C" << s.first << ".revIndInG= NULL;" << std::endl;
                os << "  C" << s.first << ".revInd= NULL;" << std::endl;
                os << "  C" << s.first << ".remap= NULL;" << std::endl;
            }

            const string numConnections = "C" + s.first + ".connN";

            allocate_device_variable(os, "unsigned int", "indInG" + s.first, false,
                                     s.second.getSrcNeuronGroup()->getNumNeurons() + 1);

            allocate_device_variable(os, "unsigned int", "ind" + s.first, false,
                                     numConnections);

            if (model.isSynapseGroupDynamicsRequired(s.first)) {
                allocate_device_variable(os, "unsigned int", "preInd" + s.first, false,
                                         numConnections);
            }
            if (model.isSynapseGroupPostLearningRequired(s.first)) {
                allocate_device_variable(os, "unsigned int", "revIndInG" + s.first, false,
                                         s.second.getTrgNeuronGroup()->getNumNeurons() + 1);
                allocate_device_variable(os, "unsigned int", "revInd" + s.first, false,
                                         numConnections);
                allocate_device_variable(os, "unsigned int", "remap" + s.first, false,
                                         numConnections);
            }

            // Allocate synapse variables
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : s.second.getWUModel()->getVars()) {
                    allocate_variable(os, v.second, v.first + s.first, s.second.isWUVarZeroCopyEnabled(v.first), numConnections);
                }
            }

            os << "}" << std::endl;
            os << std::endl;
            //setup up helper fn for this (specific) popn to generate sparse from dense
            os << "void createSparseConnectivityFromDense" << s.first << "(int preN,int postN, " << model.getPrecision() << " *denseMatrix)" << "{" << std::endl;
            os << "    gennError(\"The function createSparseConnectivityFromDense" << s.first << "() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \\n Please use your own logic and use the general tools allocate" << s.first << "(), countEntriesAbove(), and setSparseConnectivityFromDense().\");" << std::endl;
            os << "}" << std::endl;
            os << std::endl;
        }
    }

    // ------------------------------------------------------------------------
    // freeing global memory structures

    os << "void freeMem()" << std::endl;
    os << "{" << std::endl;

    // FREE NEURON VARIABLES
    for(const auto &n : model.getNeuronGroups()) {
        // Free spike buffer
        free_variable(os, "glbSpkCnt" + n.first, n.second.isSpikeZeroCopyEnabled());
        free_variable(os, "glbSpk" + n.first, n.second.isSpikeZeroCopyEnabled());

        // Free spike-like event buffer if allocated
        if (n.second.isSpikeEventRequired()) {
            free_variable(os, "glbSpkCntEvnt" + n.first, n.second.isSpikeEventZeroCopyEnabled());
            free_variable(os, "glbSpkEvnt" + n.first, n.second.isSpikeEventZeroCopyEnabled());
        }

        // Free last spike time buffer if allocated
        if (n.second.isSpikeTimeRequired()) {
            free_variable(os, "sT" + n.first, n.second.isSpikeTimeZeroCopyEnabled());
        }

#ifndef CPU_ONLY
        if(n.second.isSimRNGRequired()) {
            free_device_variable(os, "rng" + n.first, false);
        }
#endif
        // Free neuron state variables
        for (auto const &v : n.second.getNeuronModel()->getVars()) {
            free_variable(os, v.first + n.first,
                          n.second.isVarZeroCopyEnabled(v.first));
        }
    }

    // FREE SYNAPSE VARIABLES
    for(const auto &s : model.getSynapseGroups()) {
        free_variable(os, "inSyn" + s.first, false);

        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << "    C" << s.first << ".connN= 0;" << std::endl;

            free_host_variable(os, "C" + s.first + ".indInG");
            free_device_variable(os, "indInG" + s.first, false);

            free_host_variable(os, "C" + s.first + ".ind");
            free_device_variable(os, "ind" + s.first, false);

            if (model.isSynapseGroupPostLearningRequired(s.first)) {
                free_host_variable(os, "C" + s.first + ".revIndInG");
                free_device_variable(os, "revIndInG" + s.first, false);

                free_host_variable(os, "C" + s.first + ".revInd");
                free_device_variable(os, "revInd" + s.first, false);

                free_host_variable(os, "C" + s.first + ".remap");
                free_device_variable(os, "remap" + s.first, false);
            }

            if (model.isSynapseGroupDynamicsRequired(s.first)) {
                free_host_variable(os, "C" + s.first + ".preInd");
                free_device_variable(os, "preInd" + s.first, false);
            }
        }
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            free_variable(os, "gp" + s.first, false);
        }
        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            for(const auto &v : s.second.getWUModel()->getVars()) {
                free_variable(os, v.first + s.first, s.second.isWUVarZeroCopyEnabled(v.first));
            }
            for(const auto &v : s.second.getPSModel()->getVars()) {
                free_variable(os, v.first + s.first, s.second.isPSVarZeroCopyEnabled(v.first));
            }
        }
    }
    os << "}" << std::endl << std::endl;


    // ------------------------------------------------------------------------
    //! \brief Method for cleaning up and resetting device while quitting GeNN

    os << "void exitGeNN(){" << std::endl;
    os << "  freeMem();" << std::endl;
#ifndef CPU_ONLY
    os << "  cudaDeviceReset();" << std::endl;
#endif
    os << "}" << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// the actual time stepping procedure (using CPU)" << std::endl;
    os << "void stepTimeCPU()" << std::endl;
    os << "{" << std::endl;
    if (!model.getSynapseGroups().empty()) {
        if (!model.getSynapseDynamicsGroups().empty()) {
            if (model.isTimingEnabled()) os << "        synDyn_timer.startTimer();" << std::endl;
            os << "        calcSynapseDynamicsCPU(t);" << std::endl;
            if (model.isTimingEnabled()) {
                os << "        synDyn_timer.stopTimer();" << std::endl;
                os << "        synDyn_tme+= synDyn_timer.getElapsedTime();" << std::endl;
            }
        }
        if (model.isTimingEnabled()) os << "        synapse_timer.startTimer();" << std::endl;
        os << "        calcSynapsesCPU(t);" << std::endl;
        if (model.isTimingEnabled()) {
            os << "        synapse_timer.stopTimer();" << std::endl;
            os << "        synapse_tme+= synapse_timer.getElapsedTime();"<< std::endl;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
            if (model.isTimingEnabled()) os << "        learning_timer.startTimer();" << std::endl;
            os << "        learnSynapsesPostHost(t);" << std::endl;
            if (model.isTimingEnabled()) {
                os << "        learning_timer.stopTimer();" << std::endl;
                os << "        learning_tme+= learning_timer.getElapsedTime();" << std::endl;
            }
        }
    }
    if (model.isTimingEnabled()) os << "    neuron_timer.startTimer();" << std::endl;
    os << "    calcNeuronsCPU(t);" << std::endl;
    if (model.isTimingEnabled()) {
        os << "    neuron_timer.stopTimer();" << std::endl;
        os << "    neuron_tme+= neuron_timer.getElapsedTime();" << std::endl;
    }
    os << "iT++;" << std::endl;
    os << "t= iT*DT;" << std::endl;
    os << "}" << std::endl;
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
                  const string &path //!< Path for code generation
    )
{
//    cout << "entering GenRunnerGPU" << std::endl;
    string name= path + "/" + model.getName() + "_CODE/runnerGPU.cc";
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
        os << "__device__ double atomicAddSW(double* address, double val)" << std::endl;
        os << "{" << std::endl;
        os << "    unsigned long long int* address_as_ull =" << std::endl;
        os << "                                          (unsigned long long int*)address;" << std::endl;
        os << "    unsigned long long int old = *address_as_ull, assumed;" << std::endl;
        os << "    do {" << std::endl;
        os << "        assumed = old;" << std::endl;
        os << "        old = atomicCAS(address_as_ull, assumed, " << std::endl;
        os << "                        __double_as_longlong(val + " << std::endl;
        os << "                        __longlong_as_double(assumed)));" << std::endl;
        os << "    } while (assumed != old);" << std::endl;
        os << "    return __longlong_as_double(old);" << std::endl;
        os << "}" << std::endl;
        //os << "#endif"<< std::endl;
        os << std::endl;
    }

    if (deviceProp[theDevice].major < 2) {
        os << "// software version of atomic add for single precision float" << std::endl;
        os << "__device__ float atomicAddSW(float* address, float val)" << std::endl;
        os << "{" << std::endl;
        os << "    int* address_as_ull =" << std::endl;
        os << "                                          (int*)address;" << std::endl;
        os << "    int old = *address_as_ull, assumed;" << std::endl;
        os << "    do {" << std::endl;
        os << "        assumed = old;" << std::endl;
        os << "        old = atomicCAS(address_as_ull, assumed, " << std::endl;
        os << "                        __float_as_int(val + " << std::endl;
        os << "                        __int_as_float(assumed)));" << std::endl;
        os << "    } while (assumed != old);" << std::endl;
        os << "    return __int_as_float(old);" << std::endl;
        os << "}" << std::endl;
        os << std::endl;
    }

    if (model.isRNGRequired()) {
        os << "__device__ float exponentialDistFloat(curandState *rng) {" << std::endl;
        os << "    float a = 0.0f;" << std::endl;
        os << "    while (true) {" << std::endl;
        os << "        float u = curand_uniform(rng);" << std::endl;
        os << "        const float u0 = u;" << std::endl;
        os << "        while (true) {" << std::endl;
        os << "            float uStar = curand_uniform(rng);" << std::endl;
        os << "            if (u < uStar) {" << std::endl;
        os << "                return  a + u0;" << std::endl;
        os << "            }" << std::endl;
        os << "            u = curand_uniform(rng);" << std::endl;
        os << "            if (u >= uStar) {" << std::endl;
        os << "                break;" << std::endl;
        os << "            }" << std::endl;
        os << "        }" << std::endl;
        os << "        a += 1.0f;" << std::endl;
        os << "    }" << std::endl;
        os << "}" << std::endl;
        os << std::endl;
        os << "__device__ double exponentialDistDouble(curandState *rng) {" << std::endl;
        os << "    double a = 0.0f;" << std::endl;
        os << "    while (true) {" << std::endl;
        os << "        double u = curand_uniform_double(rng);" << std::endl;
        os << "        const double u0 = u;" << std::endl;
        os << "        while (true) {" << std::endl;
        os << "            double uStar = curand_uniform_double(rng);" << std::endl;
        os << "            if (u < uStar) {" << std::endl;
        os << "                return  a + u0;" << std::endl;
        os << "            }" << std::endl;
        os << "            u = curand_uniform_double(rng);" << std::endl;
        os << "            if (u >= uStar) {" << std::endl;
        os << "                break;" << std::endl;
        os << "            }" << std::endl;
        os << "        }" << std::endl;
        os << "        a += 1.0;" << std::endl;
        os << "    }" << std::endl;
        os << "}" << std::endl;
        os << std::endl;
    }

    os << "#include \"neuronKrnl.cc\"" << std::endl;
    if (!model.getSynapseGroups().empty()) {
        os << "#include \"synapseKrnl.cc\"" << std::endl;
    }

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things to device" << std::endl << std::endl;

    for(const auto &n : model.getNeuronGroups()) {
        // neuron state variables
        os << "void push" << n.first << "StateToDevice()" << std::endl;
        os << CodeStream::OB(1050);

        for(const auto &v : n.second.getNeuronModel()->getVars()) {
            // only copy non-zero-copied, non-pointers. Pointers don't transport between GPU and CPU
            if (v.second.find("*") == string::npos && !n.second.isVarZeroCopyEnabled(v.first)) {
                const size_t size = n.second.isVarQueueRequired(v.first)
                    ? n.second.getNumNeurons() * n.second.getNumDelaySlots()
                    : n.second.getNumNeurons();
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << n.first;
                os << ", " << v.first << n.first;
                os << ", " << size << " * sizeof(" << v.second << "), cudaMemcpyHostToDevice));" << std::endl;
            }
        }

        os << CodeStream::CB(1050);
        os << std::endl;

        // neuron spike variables
        os << "void push" << n.first << "SpikesToDevice()" << std::endl;
        os << CodeStream::OB(1060);

        if(!n.second.isSpikeZeroCopyEnabled()) {
            const size_t glbSpkCntSize = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCnt" << n.first;
            os << ", glbSpkCnt" << n.first;
            os << ", " << glbSpkCntSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;

            const size_t glbSpkSize = n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons();
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpk" << n.first;
            os << ", glbSpk" << n.first;
            os << ", " << glbSpkSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
        }
        
        if (n.second.isSpikeEventRequired()) {
          os << "push" << n.first << "SpikeEventsToDevice();" << std::endl;
        }

        if (n.second.isSpikeTimeRequired() && !n.second.isSpikeTimeZeroCopyEnabled()) {
            size_t size = n.second.getNumNeurons() * n.second.getNumDelaySlots();
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_sT" << n.first;
            os << ", sT" << n.first;
            os << ", " << size << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
        }

        os << CodeStream::CB(1060);
        os << std::endl;

        // neuron spike variables
        os << "void push" << n.first << "SpikeEventsToDevice()" << std::endl;
        os << CodeStream::OB(1060);

        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
            const size_t glbSpkCntEventSize = n.second.getNumDelaySlots();
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvnt" << n.first;
            os << ", glbSpkCntEvnt" << n.first;
            os << ", " << glbSpkCntEventSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;

            const size_t glbSpkEventSize = n.second.getNumNeurons() * n.second.getNumDelaySlots();
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvnt" << n.first;
            os << ", glbSpkEvnt" << n.first;
            os << ", " << glbSpkEventSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
        }

        os << CodeStream::CB(1060);
        os << std::endl;

        // current neuron spike variables
        os << "void push" << n.first << "CurrentSpikesToDevice()" << std::endl;
        os << CodeStream::OB(1061);

        if(!n.second.isSpikeZeroCopyEnabled()) {
            if (n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCnt" << n.first;
                os << "+spkQuePtr" << n.first << ", glbSpkCnt" << n.first;
                os << "+spkQuePtr" << n.first;
                os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpk" << n.first;
                os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
                os << ", glbSpk" << n.first;
                os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
                os << ", " << "glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            }
            else {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCnt" << n.first;
                os << ", glbSpkCnt" << n.first;
                os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpk" << n.first;
                os << ", glbSpk" << n.first;
                os << ", " << "glbSpkCnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            }
        }
        os << CodeStream::CB(1061);
        os << std::endl;

        // current neuron spike event variables
        os << "void push" << n.first << "CurrentSpikeEventsToDevice()" << std::endl;
        os << CodeStream::OB(1062);

        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
          if (n.second.isDelayRequired()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvnt" << n.first;
            os << "+spkQuePtr" << n.first << ", glbSpkCntEvnt" << n.first;
            os << "+spkQuePtr" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvnt" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", glbSpkEvnt" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", " << "glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
          }
          else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvnt" << n.first;
            os << ", glbSpkCntEvnt" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvnt" << n.first;
            os << ", glbSpkEvnt" << n.first;
            os << ", " << "glbSpkCntEvnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << std::endl;
          }
        }

        os << CodeStream::CB(1062);
        os << std::endl;
    }
    // synapse variables
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        os << "void push" << s.first << "StateToDevice()" << std::endl;
        os << CodeStream::OB(1100);

        const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
        const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();
        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) { // INDIVIDUALG
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                os << "size_t size = " << numSrcNeurons * numTrgNeurons << ";" << std::endl;
            }
            else {
                os << "size_t size = C" << s.first << ".connN;" << std::endl;
            }

            for(const auto &v : wu->getVars()) {
                 // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                if (v.second.find("*") == string::npos && !s.second.isWUVarZeroCopyEnabled(v.first)) {
                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << s.first;
                    os << ", " << v.first << s.first;
                    os << ", size * sizeof(" << v.second << "), cudaMemcpyHostToDevice));" << std::endl;
                }
            }

            for(const auto &v : psm->getVars()) {
                // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                if (v.second.find("*") == string::npos && !s.second.isPSVarZeroCopyEnabled(v.first)) {
                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << s.first;
                    os << ", " << v.first << s.first;
                    os << ", " << numTrgNeurons << " * sizeof(" << v.second << "), cudaMemcpyHostToDevice));" << std::endl;
                }
            }
        }
        else if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t size = (numSrcNeurons * numTrgNeurons) / 32 + 1;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << s.first;
            os << ", gp" << s.first;
            os << ", " << size << " * sizeof(uint32_t), cudaMemcpyHostToDevice));" << std::endl;
        }

        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyn" << s.first;
        os << ", inSyn" << s.first;
        os << ", " << numTrgNeurons << " * sizeof(" << model.getPrecision() << "), cudaMemcpyHostToDevice));" << std::endl;

        os << CodeStream::CB(1100);
        os << std::endl;
    }


    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying things from device" << std::endl << std::endl;

    for(const auto &n : model.getNeuronGroups()) {
        // neuron state variables
        os << "void pull" << n.first << "StateFromDevice()" << std::endl;
        os << CodeStream::OB(1050);
        
        for(const auto &v : n.second.getNeuronModel()->getVars()) {
            // only copy non-zero-copied, non-pointers. Pointers don't transport between GPU and CPU
            if (v.second.find("*") == string::npos && !n.second.isVarZeroCopyEnabled(v.first)) {
                const size_t size = n.second.isVarQueueRequired(v.first)
                    ? n.second.getNumNeurons() * n.second.getNumDelaySlots()
                    : n.second.getNumNeurons();

                os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << v.first << n.first;
                os << ", d_" << v.first << n.first;
                os << ", " << size << " * sizeof(" << v.second << "), cudaMemcpyDeviceToHost));" << std::endl;
            }
        }

        os << CodeStream::CB(1050);
        os << std::endl;

        // spike event variables
        os << "void pull" << n.first << "SpikeEventsFromDevice()" << std::endl;
        os << CodeStream::OB(1061);

        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
          const size_t glbSpkCntEvntSize = n.second.getNumDelaySlots();
          os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
          os << ", d_glbSpkCntEvnt" << n.first;
          os << ", " << glbSpkCntEvntSize << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;

          const size_t glbSpkEvntSize = n.second.getNumNeurons() * n.second.getNumDelaySlots();
          os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvnt" << n.first;
          os << ", d_glbSpkEvnt" << n.first;
          os << ", " << glbSpkEvntSize << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
        }

        os << CodeStream::CB(1061);
        os << std::endl;

        // neuron spike variables (including spike events)
        os << "void pull" << n.first << "SpikesFromDevice()" << std::endl;
        os << CodeStream::OB(1060);

        if(!n.second.isSpikeZeroCopyEnabled()) {
            size_t glbSpkCntSize = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
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
        os << CodeStream::CB(1060);
        os << std::endl;

        // neuron spike times
        os << "void pull" << n.first << "SpikeTimesFromDevice()" << std::endl;
        os << CodeStream::OB(10601);
        os << "//Assumes that spike numbers are already copied back from the device" << std::endl;
        if (n.second.isSpikeTimeRequired() && !n.second.isSpikeTimeZeroCopyEnabled()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(sT" << n.first;
            os << ", d_sT" << n.first;
            os << ", " << "glbSpkCnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
        }

        os << CodeStream::CB(10601);
        os << std::endl;

        os << "void pull" << n.first << "CurrentSpikesFromDevice()" << std::endl;
        os << CodeStream::OB(1061);

        if(!n.second.isSpikeZeroCopyEnabled()) {
            if ((n.second.isTrueSpikeRequired()) && n.second.isDelayRequired()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << n.first;
            os << "+spkQuePtr" << n.first << ", d_glbSpkCnt" << n.first;
            os << "+spkQuePtr" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", d_glbSpk" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", " << "glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }
            else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << n.first;
            os << ", d_glbSpkCnt" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << n.first;
            os << ", d_glbSpk" << n.first;
            os << ", " << "glbSpkCnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            }
        }

        os << CodeStream::CB(1061);
        os << std::endl;

        os << "void pull" << n.first << "CurrentSpikeEventsFromDevice()" << std::endl;
        os << CodeStream::OB(1062);

        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
          if (n.second.isDelayRequired()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
            os << "+spkQuePtr" << n.first << ", d_glbSpkCntEvnt" << n.first;
            os << "+spkQuePtr" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvnt" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", d_glbSpkEvnt" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", " << "glbSpkCntEvnt" << n.first << "[spkQuePtr" << n.first << "] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
          }
          else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
            os << ", d_glbSpkCntEvnt" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvnt" << n.first;
            os << ", d_glbSpkEvnt" << n.first;
            os << ", " << "glbSpkCntEvnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
          }
        }

        os << CodeStream::CB(1062);
        os << std::endl;
    }

    // synapse variables
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
        const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();

        os << "void pull" << s.first << "StateFromDevice()" << std::endl;
        os << CodeStream::OB(1100);

        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) { // INDIVIDUALG
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                os << "size_t size = " << numSrcNeurons * numTrgNeurons << ";" << std::endl;
            }
            else {
                os << "size_t size = C" << s.first << ".connN;" << std::endl;
            }

            for(const auto &v : wu->getVars()) {
                // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                if (v.second.find("*") == string::npos && !s.second.isWUVarZeroCopyEnabled(v.first)) {
                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << v.first << s.first;
                    os << ", d_"  << v.first << s.first;
                    os << ", size * sizeof(" << v.second << "), cudaMemcpyDeviceToHost));" << std::endl;
                }
            }

            for(const auto &v : psm->getVars()) {
                // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                if (v.second.find("*") == string::npos && !s.second.isPSVarZeroCopyEnabled(v.first)) {
                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << v.first << s.first;
                    os << ", d_"  << v.first << s.first;
                    os << ", " << numTrgNeurons << " * sizeof(" << v.second << "), cudaMemcpyDeviceToHost));" << std::endl;
                }
            }
        }
        else if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            size_t size = (numSrcNeurons * numTrgNeurons) / 32 + 1;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(gp" << s.first;
            os << ", d_gp" << s.first;
            os << ", " << size << " * sizeof(uint32_t), cudaMemcpyDeviceToHost));" << std::endl;
        }

        os << "CHECK_CUDA_ERRORS(cudaMemcpy(inSyn" << s.first;
        os << ", d_inSyn" << s.first;
        os << ", " << numTrgNeurons << " * sizeof(" << model.getPrecision() << "), cudaMemcpyDeviceToHost));" << std::endl;

        os << CodeStream::CB(1100);
        os << std::endl;
    }


    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying values to device" << std::endl;
    
    os << "void copyStateToDevice()" << std::endl;
    os << CodeStream::OB(1110);

    for(const auto &n : model.getNeuronGroups()) {
        os << "push" << n.first << "StateToDevice();" << std::endl;
        os << "push" << n.first << "SpikesToDevice();" << std::endl;
    }

    for(const auto &s : model.getSynapseGroups()) {
        os << "push" << s.first << "StateToDevice();" << std::endl;
    }

    os << CodeStream::CB(1110);
    os << std::endl;
    
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes to device" << std::endl;
    
    os << "void copySpikesToDevice()" << std::endl;
    os << CodeStream::OB(1111);
    for(const auto &n : model.getNeuronGroups()) {
      os << "push" << n.first << "SpikesToDevice();" << std::endl;
    }
    os << CodeStream::CB(1111);
   
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikes to device" << std::endl;
    
    os << "void copyCurrentSpikesToDevice()" << std::endl;
    os << CodeStream::OB(1112);
    for(const auto &n : model.getNeuronGroups()) {
      os << "push" << n.first << "CurrentSpikesToDevice();" << std::endl;
    }
    os << CodeStream::CB(1112);
   
    
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spike events to device" << std::endl;
    
    os << "void copySpikeEventsToDevice()" << std::endl;
    os << CodeStream::OB(1113);
    for(const auto &n : model.getNeuronGroups()) {
      os << "push" << n.first << "SpikeEventsToDevice();" << std::endl;
    }
    os << CodeStream::CB(1113);
   
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikes to device" << std::endl;
    
    os << "void copyCurrentSpikeEventsToDevice()" << std::endl;
    os << CodeStream::OB(1114);
    for(const auto &n : model.getNeuronGroups()) {
      os << "push" << n.first << "CurrentSpikeEventsToDevice();" << std::endl;
    }
    os << CodeStream::CB(1114);

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying values from device" << std::endl;
    
    os << "void copyStateFromDevice()" << std::endl;
    os << CodeStream::OB(1120);
    
    for(const auto &n : model.getNeuronGroups()) {
        os << "pull" << n.first << "StateFromDevice();" << std::endl;
        os << "pull" << n.first << "SpikesFromDevice();" << std::endl;
    }

    for(const auto &s : model.getSynapseGroups()) {
        os << "pull" << s.first << "StateFromDevice();" << std::endl;
    }
    
    os << CodeStream::CB(1120);
    os << std::endl;


    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spikes from device" << std::endl;
    
    os << "void copySpikesFromDevice()" << std::endl;
    os << CodeStream::OB(1121) << std::endl;

    for(const auto &n : model.getNeuronGroups()) {
      os << "pull" << n.first << "SpikesFromDevice();" << std::endl;
    }
    os << CodeStream::CB(1121) << std::endl;
    os << std::endl;
    
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikes from device" << std::endl;
    
    os << "void copyCurrentSpikesFromDevice()" << std::endl;
    os << CodeStream::OB(1122) << std::endl;

    for(const auto &n : model.getNeuronGroups()) {
      os << "pull" << n.first << "CurrentSpikesFromDevice();" << std::endl;
    }
    os << CodeStream::CB(1122) << std::endl;
    os << std::endl;
    

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying spike numbers from device (note, only use when only interested"<< std::endl;
    os << "// in spike numbers; copySpikesFromDevice() already includes this)" << std::endl;

    os << "void copySpikeNFromDevice()" << std::endl;
    os << CodeStream::OB(1123) << std::endl;

    for(const auto &n : model.getNeuronGroups()) {
        if(!n.second.isSpikeZeroCopyEnabled()) {
            size_t size = (n.second.isTrueSpikeRequired() && n.second.isDelayRequired())
                ? n.second.getNumDelaySlots() : 1;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << n.first;
            os << ", d_glbSpkCnt" << n.first << ", " << size << "* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
        }
    }

    os << CodeStream::CB(1123) << std::endl;
    os << std::endl;

    
    os << "// ------------------------------------------------------------------------"<< std::endl;
    os << "// global copying spikeEvents from device" << std::endl;
    
    os << "void copySpikeEventsFromDevice()" << std::endl;
    os << CodeStream::OB(1124) << std::endl;

    for(const auto &n : model.getNeuronGroups()) {
      os << "pull" << n.first << "SpikeEventsFromDevice();" << std::endl;
    }
    os << CodeStream::CB(1124) << std::endl;
    os << std::endl;
    
    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// copying current spikeEvents from device" << std::endl;
    
    os << "void copyCurrentSpikeEventsFromDevice()" << std::endl;
    os << CodeStream::OB(1125) << std::endl;

    for(const auto &n : model.getNeuronGroups()) {
      os << "pull" << n.first << "CurrentSpikeEventsFromDevice();" << std::endl;
    }
    os << CodeStream::CB(1125) << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// global copying spike event numbers from device (note, only use when only interested" << std::endl;
    os << "// in spike numbers; copySpikeEventsFromDevice() already includes this)" << std::endl;
    
    os << "void copySpikeEventNFromDevice()" << std::endl;
    os << CodeStream::OB(1126) << std::endl;

    for(const auto &n : model.getNeuronGroups()) {
        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
            const size_t size = n.second.isDelayRequired() ? n.second.getNumDelaySlots() : 1;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
            os << ", d_glbSpkCntEvnt" << n.first << ", " << size << "* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << std::endl;
        }
    }
    os << CodeStream::CB(1126) << std::endl;
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// the time stepping procedure (using GPU)" << std::endl;
    os << "void stepTimeGPU()" << std::endl;
    os << CodeStream::OB(1130) << std::endl;

    if (!model.getSynapseGroups().empty()) {
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
    if (!model.getSynapseGroups().empty()) {
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
    for(auto &n : model.getNeuronGroups()) {
        if (n.second.isDelayRequired()) {
            os << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
        }
    }
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
        if (!model.getSynapseGroups().empty()) {
            os << "cudaEventElapsedTime(&tmp, synapseStart, synapseStop);" << std::endl;
            os << "synapse_tme+= tmp/1000.0;" << std::endl;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
            os << "cudaEventElapsedTime(&tmp, learningStart, learningStop);" << std::endl;
            os << "learning_tme+= tmp/1000.0;" << std::endl;
        }
        if (!model.getSynapseDynamicsGroups().empty()) {
            os << "cudaEventElapsedTime(&tmp, synDynStart, synDynStop);" << std::endl;
            os << "lsynDyn_tme+= tmp/1000.0;" << std::endl;
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
    os << CodeStream::CB(1130) << std::endl;
    fs.close();
    //cout << "done with generating GPU runner" << std::endl;
}
#endif // CPU_ONLY


//----------------------------------------------------------------------------
/*!
  \brief A function that generates the Makefile for all generated GeNN code.
*/
//----------------------------------------------------------------------------

void genMakefile(const NNmodel &model, //!< Model description
                 const string &path    //!< Path for code generation
                 )
{
    string name = path + "/" + model.getName() + "_CODE/Makefile";
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
    cxxFlags += " -DCPU_ONLY";
    cxxFlags += " " + GENN_PREFERENCES::userCxxFlagsGNU;
    if (GENN_PREFERENCES::optimizeCode) {
        cxxFlags += " -O3 -ffast-math";
    }
    if (GENN_PREFERENCES::debugCode) {
        cxxFlags += " -O0 -g";
    }
    if (model.isRNGRequired()) {
        cxxFlags += " -std=c++11";
    }

    os << endl;
    os << "CXXFLAGS       :=" << cxxFlags << endl;
    os << endl;
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)/lib/include\"" << endl;
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

    nvccFlags += " -x cu -arch sm_";
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
    if (model.isRNGRequired()) {
        nvccFlags += " -std=c++11";
    }

    os << endl;
    os << "NVCC           :=\"" << NVCC << "\"" << endl;
    os << "NVCCFLAGS      :=" << nvccFlags << endl;
    os << endl;
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)/lib/include\"" << endl;
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
