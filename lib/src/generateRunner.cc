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
#include "CodeHelper.h"

#include <stdint.h>
#include <algorithm>
#include <cfloat>

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
//--------------------------------------------------------------------------
//! \brief This function generates host and device variable definitions, of the given type and name.
//--------------------------------------------------------------------------

void variable_def(ofstream &os, const string &type, const string &name)
{
    os << type << " " << name << ";" << ENDL;
#ifndef CPU_ONLY
    os << type << " d_" << name << ";" << ENDL;
    os << "__device__ " << type << " dd_" << name << ";" << ENDL;
#endif
}


//--------------------------------------------------------------------------
//! \brief This function generates host extern variable definitions, of the given type and name.
//--------------------------------------------------------------------------

void extern_variable_def(ofstream &os, const string &type, const string &name)
{
    os << "extern " << type << " " << name << ";" << ENDL;
#ifndef CPU_ONLY
    os << "extern " << type << " d_" << name << ";" << ENDL;
#endif
}


//--------------------------------------------------------------------------
//! \brief This function generates host allocation code
//--------------------------------------------------------------------------

void allocate_host_variable(ofstream &os, const string &type, const string &name, bool zeroCopy, const string &size)
{
#ifndef CPU_ONLY
    const char *flags = zeroCopy ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
    os << "    cudaHostAlloc(&" << name << ", " << size << " * sizeof(" << type << "), " << flags << ");" << ENDL;
#else
    USE(zeroCopy);

    os << "    " << name << " = new " << type << "[" << size << "];" << ENDL;
#endif
}

void allocate_host_variable(ofstream &os, const string &type, const string &name, bool zeroCopy, size_t size)
{
    allocate_host_variable(os, type, name, zeroCopy, to_string(size));
}

void allocate_device_variable(ofstream &os, const string &type, const string &name, bool zeroCopy, const string &size)
{
#ifndef CPU_ONLY
    // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
    if(zeroCopy) {
        os << "    deviceZeroCopy(" << name << ", &d_" << name << ", dd_" << name << ");" << ENDL;
    }
    else {
        os << "    deviceMemAllocate(&d_" << name << ", dd_" << name << ", " << size << " * sizeof(" << type << "));" << ENDL;
    }
#else
    USE(os);
    USE(type);
    USE(name);
    USE(zeroCopy);
    USE(size);
#endif
}

void allocate_device_variable(ofstream &os, const string &type, const string &name, bool zeroCopy, size_t size)
{
    allocate_device_variable(os, type, name, zeroCopy, to_string(size));
}

//--------------------------------------------------------------------------
//! \brief This function generates host and device allocation with standard names (name, d_name, dd_name) and estimates size based on size known at generate-time
//--------------------------------------------------------------------------
unsigned int allocate_variable(ofstream &os, const string &type, const string &name, bool zeroCopy, size_t size)
{
    // Allocate host and device variables
    allocate_host_variable(os, type, name, zeroCopy, size);
    allocate_device_variable(os, type, name, zeroCopy, size);

    // Return size estimate
    return size * theSize(type);
}

void allocate_variable(ofstream &os, const string &type, const string &name, bool zeroCopy, const string &size)
{
    // Allocate host and device variables
    allocate_host_variable(os, type, name, zeroCopy, size);
    allocate_device_variable(os, type, name, zeroCopy, size);
}

void free_host_variable(ofstream &os, const string &name)
{
#ifndef CPU_ONLY
    os << "    CHECK_CUDA_ERRORS(cudaFreeHost(" << name << "));" << ENDL;
#else
    os << "    delete[] " << name << ";" << ENDL;
#endif
}

void free_device_variable(ofstream &os, const string &name, bool zeroCopy)
{
#ifndef CPU_ONLY
    // If this variable wasn't allocated in zero-copy mode, free it
    if(!zeroCopy) {
        os << "    CHECK_CUDA_ERRORS(cudaFree(d_" << name << "));" << ENDL;
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
void free_variable(ofstream &os, const string &name, bool zeroCopy)
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

void genRunner(const NNmodel &model, //!< Model description
               const string &path //!< Path for code generationn
    )
{
    ofstream os;

    unsigned int mem = 0;
  
    string SCLR_MIN;
    string SCLR_MAX;
    if (model.ftype == "float") {
        SCLR_MIN= to_string(FLT_MIN)+"f";
        SCLR_MAX= to_string(FLT_MAX)+"f";
    }

    if (model.ftype == "double") {
        SCLR_MIN= to_string(DBL_MIN);
        SCLR_MAX= to_string(DBL_MAX);
    }

    //=======================
    // generate definitions.h
    //=======================

    // this file contains helpful macros and is separated out so that it can also be used by other code that is compiled separately
    string definitionsName= path + "/" + model.name + "_CODE/definitions.h";
    os.open(definitionsName.c_str());
    writeHeader(os);
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file definitions.h" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing useful Macros used for both GPU amd CPU versions." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;
    
    os << "#ifndef DEFINITIONS_H" << ENDL;
    os << "#define DEFINITIONS_H" << ENDL;
    os << ENDL;

    os << "#include \"utils.h\"" << ENDL;
    if (model.timing) os << "#include \"hr_time.h\"" << ENDL;
    os << "#include \"sparseUtils.h\"" << ENDL << ENDL;
    os << "#include \"sparseProjection.h\"" << ENDL;
    os << "#include <stdint.h>" << ENDL;
    os << ENDL;

#ifndef CPU_ONLY
    // write CUDA error handler macro
    os << "#ifndef CHECK_CUDA_ERRORS" << ENDL;
    os << "#define CHECK_CUDA_ERRORS(call) {\\" << ENDL;
    os << "    cudaError_t error = call;\\" << ENDL;
    os << "    if (error != cudaSuccess) {\\" << ENDL;
    os << "        fprintf(stderr, \"%s: %i: cuda error %i: %s\\n\", __FILE__, __LINE__, (int) error, cudaGetErrorString(error));\\" << ENDL;
    os << "        exit(EXIT_FAILURE);\\" << ENDL;
    os << "    }\\" << ENDL;
    os << "}" << ENDL;
    os << "#endif" << ENDL;
    os << ENDL;
#else
    // define CUDA device and function type qualifiers
    os << "#define __device__" << ENDL;
    os << "#define __global__" << ENDL;
    os << "#define __host__" << ENDL;
    os << "#define __constant__" << ENDL;
    os << "#define __shared__" << ENDL;
#endif // CPU_ONLY

    // write DT macro
    os << "#undef DT" << ENDL;
    if (model.ftype == "float") {
        os << "#define DT " << to_string(model.dt) << "f" << ENDL;
    } else {
        os << "#define DT " << to_string(model.dt) << ENDL;
    }

    // write MYRAND macro
    os << "#ifndef MYRAND" << ENDL;
    os << "#define MYRAND(Y,X) Y = Y * 1103515245 + 12345; X = (Y >> 16);" << ENDL;
    os << "#endif" << ENDL;
    os << "#ifndef MYRAND_MAX" << ENDL;
    os << "#define MYRAND_MAX 0x0000FFFFFFFFFFFFLL" << ENDL;
    os << "#endif" << ENDL;
    os << ENDL;

    os << "#ifndef scalar" << ENDL;
    os << "typedef " << model.ftype << " scalar;" << ENDL;
    os << "#endif" << ENDL;
    os << "#ifndef SCALAR_MIN" << ENDL;
    os << "#define SCALAR_MIN " << SCLR_MIN << ENDL;
    os << "#endif" << ENDL;
    os << "#ifndef SCALAR_MAX" << ENDL;
    os << "#define SCALAR_MAX " << SCLR_MAX << ENDL;
    os << "#endif" << ENDL;
    os << ENDL;


    //-----------------
    // GLOBAL VARIABLES

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global variables" << ENDL;
    os << ENDL;

    os << "extern unsigned long long iT;" << ENDL;
    os << "extern " << model.ftype << " t;" << ENDL;
    if (model.timing) {
#ifndef CPU_ONLY
        os << "extern cudaEvent_t neuronStart, neuronStop;" << ENDL;
#endif
        os << "extern double neuron_tme;" << ENDL;
        os << "extern CStopWatch neuron_timer;" << ENDL;
        if (!model.getSynapseGroups().empty()) {
#ifndef CPU_ONLY
            os << "extern cudaEvent_t synapseStart, synapseStop;" << ENDL;
#endif
            os << "extern double synapse_tme;" << ENDL;
            os << "extern CStopWatch synapse_timer;" << ENDL;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
#ifndef CPU_ONLY
            os << "extern cudaEvent_t learningStart, learningStop;" << ENDL;
#endif
            os << "extern double learning_tme;" << ENDL;
            os << "extern CStopWatch learning_timer;" << ENDL;
        }
        if (!model.getSynapseDynamicsGroups().empty()) {
#ifndef CPU_ONLY
            os << "extern cudaEvent_t synDynStart, synDynStop;" << ENDL;
#endif
            os << "extern double synDyn_tme;" << ENDL;
            os << "extern CStopWatch synDyn_timer;" << ENDL;
        }
    } 
    os << ENDL;


    //---------------------------------
    // HOST AND DEVICE NEURON VARIABLES

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// neuron variables" << ENDL;
    os << ENDL;

    for(const auto &n : model.getNeuronGroups()) {
        extern_variable_def(os, "unsigned int *", "glbSpkCnt"+n.first);
        extern_variable_def(os, "unsigned int *", "glbSpk"+n.first);
        if (n.second.isSpikeEventRequired()) {
            extern_variable_def(os, "unsigned int *", "glbSpkCntEvnt"+n.first);
            extern_variable_def(os, "unsigned int *", "glbSpkEvnt"+n.first);
        }
        if (n.second.isDelayRequired()) {
            os << "extern unsigned int spkQuePtr" << n.first << ";" << ENDL;
        }
        if (n.second.isSpikeTimeRequired()) {
            extern_variable_def(os, model.ftype+" *", "sT"+n.first);
        }

        auto neuronModel = n.second.getNeuronModel();
        for(auto const &v : neuronModel->getVars()) {
            extern_variable_def(os, v.second +" *", v.first + n.first);
        }
        for(auto const &v : neuronModel->getExtraGlobalParams()) {
            extern_variable_def(os, v.second, v.first + n.first);
        }
    }
    os << ENDL;
    for(auto &n : model.getNeuronGroups()) {
        os << "#define glbSpkShift" << n.first;
        if (n.second.isDelayRequired()) {
            os << " spkQuePtr" << n.first << "*" << n.second.getNumNeurons();
        }
        else {
            os << " 0";
        }
        os << ENDL;
    }

    for(const auto &n : model.getNeuronGroups()) {
        // convenience macros for accessing spike count
        os << "#define spikeCount_" << n.first << " glbSpkCnt" << n.first;
        if (n.second.isDelayRequired() && n.second.isTrueSpikeRequired()) {
            os << "[spkQuePtr" << n.first << "]" << ENDL;
        }
        else {
            os << "[0]" << ENDL;
        }
        // convenience macro for accessing spikes
        os << "#define spike_" << n.first;
        if (n.second.isDelayRequired() && n.second.isTrueSpikeRequired()) {
            os << " (glbSpk" << n.first << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << "))" << ENDL;
        }
        else {
            os << " glbSpk" << n.first << ENDL;
        }
        if (n.second.isSpikeEventRequired()) {
            // convenience macros for accessing spike count
            os << "#define spikeEventCount_" << n.first << " glbSpkCntEvnt" << n.first;
            if (n.second.isDelayRequired()) {
                os << "[spkQuePtr" << n.first << "]" << ENDL;
            }
            else {
                os << "[0]" << ENDL;
            }
            // convenience macro for accessing spikes
            os << "#define spikeEvent_" << n.first;
            if (n.second.isDelayRequired()) {
                os << " (glbSpkEvnt" << n.first << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << "))" << ENDL;
            }
            else {
                os << " glbSpkEvnt" << n.first << ENDL;
            }
        }
    }
    os << ENDL;


    //----------------------------------
    // HOST AND DEVICE SYNAPSE VARIABLES

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// synapse variables" << ENDL;
    os << ENDL;

    for(const auto &s : model.getSynapseGroups()) {
        extern_variable_def(os, model.ftype+" *", "inSyn" + s.first);
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            extern_variable_def(os, "uint32_t *", "gp" + s.first);
        }
        else if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << "extern SparseProjection C" << s.first << ";" << ENDL;
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
    os << ENDL;

    os << "#define Conductance SparseProjection" << ENDL;
    os << "/*struct Conductance is deprecated. \n\
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. \n\
  Please consider updating your user code by renaming Conductance as SparseProjection \n\
  and making g member a synapse variable.*/" << ENDL;
    os << ENDL;


    //--------------------------
    // HOST AND DEVICE FUNCTIONS

#ifndef CPU_ONLY
    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// Helper function for allocating memory blocks on the GPU device" << ENDL;
    os << ENDL;
    os << "template<class T>" << ENDL;
    os << "void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)" << ENDL;
    os << "{" << ENDL;
    os << "    void *devptr;" << ENDL;
    os << "    CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));" << ENDL;
    os << "    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));" << ENDL;
    os << "    CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));" << ENDL;
    os << "}" << ENDL;
    os << ENDL;
    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// Helper function for getting the device pointer corresponding to a zero-copied host pointer and assigning it to a symbol" << ENDL;
    os << ENDL;
    os << "template<class T>" << ENDL;
    os << "void deviceZeroCopy(T hostPtr, const T *devPtr, const T &devSymbol)" << ENDL;
    os << "{" << ENDL;
    os << "    CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void **)devPtr, (void*)hostPtr, 0));" << ENDL;
    os << "    void *devSymbolPtr;" << ENDL;
    os << "    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devSymbolPtr, devSymbol));" << ENDL;
    os << "    CHECK_CUDA_ERRORS(cudaMemcpy(devSymbolPtr, devPtr, sizeof(void*), cudaMemcpyHostToDevice));" << ENDL;
    os << "}" << ENDL;
    os << ENDL;
#endif

#ifndef CPU_ONLY
    // generate headers for the communication utility functions such as 
    // pullXXXStateFromDevice() etc. This is useful for the brian2genn
    // interface where we do more proper compile/link and do not want
    // to include runnerGPU.cc into all relevant code_objects (e.g.
    // spike and state monitors

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying things to device" << ENDL;
    os << ENDL;
    for(const auto &n : model.getNeuronGroups()) {
        os << "void push" << n.first << "StateToDevice();" << ENDL;
        os << "void push" << n.first << "SpikesToDevice();" << ENDL;
        os << "void push" << n.first << "SpikeEventsToDevice();" << ENDL;
        os << "void push" << n.first << "CurrentSpikesToDevice();" << ENDL;
        os << "void push" << n.first << "CurrentSpikeEventsToDevice();" << ENDL;
    }
    for(const auto &s : model.getSynapseGroups()) {
        os << "#define push" << s.first << "ToDevice push" << s.first << "StateToDevice" << ENDL;
        os << "void push" << s.first << "StateToDevice();" << ENDL;
    }
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying things from device" << ENDL;
    os << ENDL;
    for(const auto &n : model.getNeuronGroups()) {
         os << "void pull" << n.first << "StateFromDevice();" << ENDL;
        os << "void pull" << n.first << "SpikesFromDevice();" << ENDL;
        os << "void pull" << n.first << "SpikeEventsFromDevice();" << ENDL;
        os << "void pull" << n.first << "CurrentSpikesFromDevice();" << ENDL;
        os << "void pull" << n.first << "CurrentSpikeEventsFromDevice();" << ENDL;
    }
    for(const auto &s : model.getSynapseGroups()) {
        os << "#define pull" << s.first << "FromDevice pull" << s.first << "StateFromDevice" << ENDL;
        os << "void pull" << s.first << "StateFromDevice();" << ENDL;
    }
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying values to device" << ENDL;
    os << ENDL;
    os << "void copyStateToDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying spikes to device" << ENDL;
    os << ENDL;
    os << "void copySpikesToDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying current spikes to device" << ENDL;
    os << ENDL;
    os << "void copyCurrentSpikesToDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying spike events to device" << ENDL;    
    os << ENDL;
    os << "void copySpikeEventsToDevice();" << ENDL;    
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying current spikes to device" << ENDL;
    os << ENDL;
    os << "void copyCurrentSpikeEventsToDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying values from device" << ENDL;
    os << ENDL;
    os << "void copyStateFromDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying spikes from device" << ENDL;
    os << ENDL;
    os << "void copySpikesFromDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying current spikes from device" << ENDL;
    os << ENDL;
    os << "void copyCurrentSpikesFromDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying spike numbers from device (note, only use when only interested"<< ENDL;
    os << "// in spike numbers; copySpikesFromDevice() already includes this)" << ENDL;
    os << ENDL;
    os << "void copySpikeNFromDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------"<< ENDL;
    os << "// global copying spikeEvents from device" << ENDL;
    os << ENDL;
    os << "void copySpikeEventsFromDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying current spikeEvents from device" << ENDL;
    os << ENDL;
    os << "void copyCurrentSpikeEventsFromDevice();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying spike event numbers from device (note, only use when only interested" << ENDL;
    os << "// in spike numbers; copySpikeEventsFromDevice() already includes this)" << ENDL;
    os << ENDL;
    os << "void copySpikeEventNFromDevice();" << ENDL;
    os << ENDL;
#endif

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// Function for setting the CUDA device and the host's global variables." << ENDL;
    os << "// Also estimates memory usage on device." << ENDL;
    os << ENDL;
    os << "void allocateMem();" << ENDL;
    os << ENDL;

    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << "void allocate" << s.first << "(unsigned int connN);" << ENDL;
            os << ENDL;
        }
    }

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// Function to (re)set all model variables to their compile-time, homogeneous initial" << ENDL;
    os << "// values. Note that this typically includes synaptic weight values. The function" << ENDL;
    os << "// (re)sets host side variables and copies them to the GPU device." << ENDL;
    os << ENDL;
    os << "void initialize();" << ENDL;
    os << ENDL;


#ifndef CPU_ONLY
    os << "void initializeAllSparseArrays();" << ENDL;
    os << ENDL;
#endif

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// initialization of variables, e.g. reverse sparse arrays etc." << ENDL;
    os << "// that the user would not want to worry about" << ENDL;
    os << ENDL;
    os << "void init" << model.name << "();" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// Function to free all global memory structures." << ENDL;
    os << ENDL;
    os << "void freeMem();" << ENDL;
    os << ENDL;

    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "// Function to convert a firing probability (per time step) to an integer of type uint64_t" << ENDL;
    os << "// that can be used as a threshold for the GeNN random number generator to generate events with the given probability." << ENDL;
    os << ENDL;
    os << "void convertProbabilityToRandomNumberThreshold(" << model.ftype << " *p_pattern, " << model.RNtype << " *pattern, int N);" << ENDL;
    os << ENDL;

    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "// Function to convert a firing rate (in kHz) to an integer of type uint64_t that can be used" << ENDL;
    os << "// as a threshold for the GeNN random number generator to generate events with the given rate." << ENDL;
    os << ENDL;
    os << "void convertRateToRandomNumberThreshold(" << model.ftype << " *rateKHz_pattern, " << model.RNtype << " *pattern, int N);" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// Throw an error for \"old style\" time stepping calls (using CPU)" << ENDL;
    os << ENDL;
    os << "template <class T>" << ENDL;
    os << "void stepTimeCPU(T arg1, ...)" << OB(101);
    os << "gennError(\"Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.\");" << ENDL; 
    os << CB(101);
    os<< ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// the actual time stepping procedure (using CPU)" << ENDL;
    os << ENDL;
    os << "void stepTimeCPU();" << ENDL;
    os << ENDL;

#ifndef CPU_ONLY
    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// Throw an error for \"old style\" time stepping calls (using GPU)" << ENDL;
    os << ENDL;
    os << "template <class T>" << ENDL;
    os << "void stepTimeGPU(T arg1, ...)" << OB(101);
    os << "gennError(\"Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.\");" << ENDL;
    os << CB(101);
    os<< ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// the actual time stepping procedure (using GPU)" << ENDL;
    os << ENDL;
    os << "void stepTimeGPU();" << ENDL;
    os << ENDL;
#endif

    os << "#endif" << ENDL;
    os.close();


    //========================
    // generate support_code.h
    //========================

    string supportCodeName= path + "/" + model.name + "_CODE/support_code.h";
    os.open(supportCodeName.c_str());
    writeHeader(os);
    os << ENDL;
    
       // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file support_code.h" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing support code provided by the user and used for both GPU amd CPU versions." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;
    
    os << "#ifndef SUPPORT_CODE_H" << ENDL;
    os << "#define SUPPORT_CODE_H" << ENDL;
    // write the support codes
    os << "// support code for neuron and synapse models" << ENDL;
    for(const auto &n : model.getNeuronGroups()) {
        if (!n.second.getNeuronModel()->getSupportCode().empty()) {
            os << "namespace " << n.first << "_neuron" << OB(11) << ENDL;
            os << ensureFtype(n.second.getNeuronModel()->getSupportCode(), model.ftype) << ENDL;
            os << CB(11) << " // end of support code namespace " << n.first << ENDL;
        }
    }
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        if (!wu->getSimSupportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_simCode " << OB(11) << ENDL;
            os << ensureFtype(wu->getSimSupportCode(), model.ftype) << ENDL;
            os << CB(11) << " // end of support code namespace " << s.first << "_weightupdate_simCode " << ENDL;
        }
        if (!wu->getLearnPostSupportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_simLearnPost " << OB(11) << ENDL;
            os << ensureFtype(wu->getLearnPostSupportCode(), model.ftype) << ENDL;
            os << CB(11) << " // end of support code namespace " << s.first << "_weightupdate_simLearnPost " << ENDL;
        }
        if (!wu->getSynapseDynamicsSuppportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_synapseDynamics " << OB(11) << ENDL;
            os << ensureFtype(wu->getSynapseDynamicsSuppportCode(), model.ftype) << ENDL;
            os << CB(11) << " // end of support code namespace " << s.first << "_weightupdate_synapseDynamics " << ENDL;
        }
        if (!psm->getSupportCode().empty()) {
            os << "namespace " << s.first << "_postsyn " << OB(11) << ENDL;
            os << ensureFtype(psm->getSupportCode(), model.ftype) << ENDL;
            os << CB(11) << " // end of support code namespace " << s.first << "_postsyn " << ENDL;
        }

    }
    os << "#endif" << ENDL;
    os.close();
    

    //cout << "entering genRunner" << ENDL;
    string runnerName= path + "/" + model.name + "_CODE/runner.cc";
    os.open(runnerName.c_str());
    writeHeader(os);
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file runner.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing general control code." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << ENDL;

    os << "#define RUNNER_CC_COMPILE" << ENDL;
    os << ENDL;
    os << "#include \"definitions.h\"" << ENDL;
    os << "#include <cstdlib>" << ENDL;
    os << "#include <cstdio>" << ENDL;
    os << "#include <cmath>" << ENDL;
    os << "#include <ctime>" << ENDL;
    os << "#include <cassert>" << ENDL;
    os << "#include <stdint.h>" << ENDL;
    os << ENDL;


    //-----------------
    // GLOBAL VARIABLES

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global variables" << ENDL;
    os << ENDL;

    os << "unsigned long long iT= 0;" << ENDL;
    os << model.ftype << " t;" << ENDL;
    if (model.timing) {
#ifndef CPU_ONLY
        os << "cudaEvent_t neuronStart, neuronStop;" << ENDL;
#endif
        os << "double neuron_tme;" << ENDL;
        os << "CStopWatch neuron_timer;" << ENDL;
        if (!model.getSynapseGroups().empty()) {
#ifndef CPU_ONLY
            os << "cudaEvent_t synapseStart, synapseStop;" << ENDL;
#endif
            os << "double synapse_tme;" << ENDL;
            os << "CStopWatch synapse_timer;" << ENDL;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
#ifndef CPU_ONLY
            os << "cudaEvent_t learningStart, learningStop;" << ENDL;
#endif
            os << "double learning_tme;" << ENDL;
            os << "CStopWatch learning_timer;" << ENDL;
        }
        if (!model.getSynapseDynamicsGroups().empty()) {
#ifndef CPU_ONLY
            os << "cudaEvent_t synDynStart, synDynStop;" << ENDL;
#endif
            os << "double synDyn_tme;" << ENDL;
            os << "CStopWatch synDyn_timer;" << ENDL;
        }
    } 
    os << ENDL;


    //---------------------------------
    // HOST AND DEVICE NEURON VARIABLES

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// neuron variables" << ENDL;
    os << ENDL;

#ifndef CPU_ONLY
    os << "__device__ volatile unsigned int d_done;" << ENDL;
#endif
    for(const auto &n : model.getNeuronGroups()) {
        variable_def(os, "unsigned int *", "glbSpkCnt"+n.first);
        variable_def(os, "unsigned int *", "glbSpk"+n.first);
        if (n.second.isSpikeEventRequired()) {
            variable_def(os, "unsigned int *", "glbSpkCntEvnt"+n.first);
            variable_def(os, "unsigned int *", "glbSpkEvnt"+n.first);
        }
        if (n.second.isDelayRequired()) {
            os << "unsigned int spkQuePtr" << n.first << ";" << ENDL;
#ifndef CPU_ONLY
            os << "__device__ volatile unsigned int dd_spkQuePtr" << n.first << ";" << ENDL;
#endif
        }
        if (n.second.isSpikeTimeRequired()) {
            variable_def(os, model.ftype+" *", "sT"+n.first);
        }

        auto neuronModel = n.second.getNeuronModel();
        for(auto const &v : neuronModel->getVars()) {
            variable_def(os, v.second + " *", v.first + n.first);
        }
        for(auto const &v : neuronModel->getExtraGlobalParams()) {
            os << v.second << " " <<  v.first << n.first << ";" << ENDL;
        }
    }
    os << ENDL;


    //----------------------------------
    // HOST AND DEVICE SYNAPSE VARIABLES

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// synapse variables" << ENDL;
    os << ENDL;

   for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        variable_def(os, model.ftype+" *", "inSyn"+s.first);
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            variable_def(os, "uint32_t *", "gp"+s.first);
        }
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << "SparseProjection C" << s.first << ";" << ENDL;
#ifndef CPU_ONLY
            os << "unsigned int *d_indInG" << s.first << ";" << ENDL;
            os << "__device__ unsigned int *dd_indInG" << s.first << ";" << ENDL;
            os << "unsigned int *d_ind" << s.first << ";" << ENDL;
            os << "__device__ unsigned int *dd_ind" << s.first << ";" << ENDL;
            if (model.isSynapseGroupDynamicsRequired(s.first)) {
                os << "unsigned int *d_preInd" << s.first << ";" << ENDL;
                os << "__device__ unsigned int *dd_preInd" << s.first << ";" << ENDL;
            }
            if (model.isSynapseGroupPostLearningRequired(s.first)) {
                // TODO: make conditional on post-spike driven learning actually taking place
                os << "unsigned int *d_revIndInG" << s.first << ";" << ENDL;
                os << "__device__ unsigned int *dd_revIndInG" << s.first << ";" << ENDL;
                os << "unsigned int *d_revInd" << s.first << ";" << ENDL;
                os << "__device__ unsigned int *dd_revInd" << s.first << ";" << ENDL;
                os << "unsigned int *d_remap" << s.first << ";" << ENDL;
                os << "__device__ unsigned int *dd_remap" << s.first << ";" << ENDL;
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
            os << v.second << " " <<  v.first<< s.first << ";" << ENDL;
        }
    }
    os << ENDL;


    //--------------------------
    // HOST AND DEVICE FUNCTIONS

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\brief Function to convert a firing probability (per time step) " << ENDL;
    os << "to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given probability." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    os << "void convertProbabilityToRandomNumberThreshold(" << model.ftype << " *p_pattern, " << model.RNtype << " *pattern, int N)" << ENDL;
    os << "{" << ENDL;
    os << "    " << model.ftype << " fac= pow(2.0, (double) sizeof(" << model.RNtype << ")*8-16);" << ENDL;
    os << "    for (int i= 0; i < N; i++) {" << ENDL;
    //os << "        assert(p_pattern[i] <= 1.0);" << ENDL;
    os << "        pattern[i]= (" << model.RNtype << ") (p_pattern[i]*fac);" << ENDL;
    os << "    }" << ENDL;
    os << "}" << ENDL << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\brief Function to convert a firing rate (in kHz) " << ENDL;
    os << "to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given rate." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    os << "void convertRateToRandomNumberThreshold(" << model.ftype << " *rateKHz_pattern, " << model.RNtype << " *pattern, int N)" << ENDL;
    os << "{" << ENDL;
    os << "    " << model.ftype << " fac= pow(2.0, (double) sizeof(" << model.RNtype << ")*8-16)*DT;" << ENDL;
    os << "    for (int i= 0; i < N; i++) {" << ENDL;
    //os << "        assert(rateKHz_pattern[i] <= 1.0);" << ENDL;
    os << "        pattern[i]= (" << model.RNtype << ") (rateKHz_pattern[i]*fac);" << ENDL;
    os << "    }" << ENDL;
    os << "}" << ENDL << ENDL;

    // include simulation kernels
#ifndef CPU_ONLY
    os << "#include \"runnerGPU.cc\"" << ENDL << ENDL;
#endif
    os << "#include \"neuronFnct.cc\"" << ENDL;
    if (!model.getSynapseGroups().empty()) {
        os << "#include \"synapseFnct.cc\"" << ENDL;
    }


    // ---------------------------------------------------------------------
    // Function for setting the CUDA device and the host's global variables.
    // Also estimates memory usage on device ...
  
    os << "void allocateMem()" << ENDL;
    os << "{" << ENDL;
#ifndef CPU_ONLY
    os << "    CHECK_CUDA_ERRORS(cudaSetDevice(" << theDevice << "));" << ENDL;

    // If the model requires zero-copy
    if(model.zeroCopyInUse())
    {
        // If device doesn't support mapping host memory error
        if(!deviceProp[theDevice].canMapHostMemory) {
            gennError("Device does not support mapping CPU host memory!");
        }

        // set appropriate device flags
        os << "    CHECK_CUDA_ERRORS(cudaSetDeviceFlags(cudaDeviceMapHost));" << ENDL;
    }
#endif
    //cout << "model.neuronGroupN " << model.neuronGrpN << ENDL;
    //os << "    " << model.ftype << " free_m, total_m;" << ENDL;
    //os << "    cudaMemGetInfo((size_t*) &free_m, (size_t*) &total_m);" << ENDL;

    if (model.timing) {
#ifndef CPU_ONLY
        os << "    cudaEventCreate(&neuronStart);" << ENDL;
        os << "    cudaEventCreate(&neuronStop);" << ENDL;
#endif
        os << "    neuron_tme= 0.0;" << ENDL;
        if (!model.getSynapseGroups().empty()) {
#ifndef CPU_ONLY
            os << "    cudaEventCreate(&synapseStart);" << ENDL;
            os << "    cudaEventCreate(&synapseStop);" << ENDL;
#endif
            os << "    synapse_tme= 0.0;" << ENDL;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
#ifndef CPU_ONLY
            os << "    cudaEventCreate(&learningStart);" << ENDL;
            os << "    cudaEventCreate(&learningStop);" << ENDL;
#endif
            os << "    learning_tme= 0.0;" << ENDL;
        }
        if (!model.getSynapseDynamicsGroups().empty()) {
#ifndef CPU_ONLY
            os << "    cudaEventCreate(&synDynStart);" << ENDL;
            os << "    cudaEventCreate(&synDynStop);" << ENDL;
#endif
            os << "    synDyn_tme= 0.0;" << ENDL;
        }
    }

    // ALLOCATE NEURON VARIABLES
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
            mem += allocate_variable(os, model.ftype, "sT" + n.first, n.second.isSpikeTimeZeroCopyEnabled(),
                                     n.second.getNumNeurons() * n.second.getNumDelaySlots());
        }

        // Allocate memory for neuron model's state variables
        for(const auto &v : n.second.getNeuronModel()->getVars()) {
            mem += allocate_variable(os, v.second, v.first + n.first, n.second.isVarZeroCopyEnabled(v.first),
                                     n.second.isVarQueueRequired(v.first) ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());
        }
        os << ENDL;
    }

    // ALLOCATE SYNAPSE VARIABLES
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        // Allocate buffer to hold input coming from this synapse population
        mem += allocate_variable(os, model.ftype, "inSyn" + s.first, false,
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
        os << ENDL;
    }
    os << "}" << ENDL << ENDL;


    // ------------------------------------------------------------------------
    // initializing variables
    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\brief Function to (re)set all model variables to their compile-time, homogeneous initial values." << ENDL;
    os << " Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    os << "void initialize()" << ENDL;
    os << "{" << ENDL;

    // Extra braces around Windows for loops to fix https://support.microsoft.com/en-us/kb/315481
#ifdef _WIN32
    string oB = "{", cB = "}";
#else
    string oB = "", cB = "";
#endif // _WIN32

    if (model.seed == 0) {
        os << "    srand((unsigned int) time(NULL));" << ENDL;
    }
    else {
        os << "    srand((unsigned int) " << model.seed << ");" << ENDL;
    }
    os << ENDL;

    // INITIALISE NEURON VARIABLES
    os << "    // neuron variables" << ENDL;
    for(const auto &n : model.getNeuronGroups()) {
        if (n.second.isDelayRequired()) {
            os << "    spkQuePtr" << n.first << " = 0;" << ENDL;
#ifndef CPU_ONLY
            os << "CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_spkQuePtr" << n.first;
            os << ", &spkQuePtr" << n.first;
            os << ", sizeof(unsigned int), 0, cudaMemcpyHostToDevice));" << ENDL;
#endif
        }

        if (n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++) {" << ENDL;
            os << "        glbSpkCnt" << n.first << "[i] = 0;" << ENDL;
            os << "    }" << cB << ENDL;
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++) {" << ENDL;
            os << "        glbSpk" << n.first << "[i] = 0;" << ENDL;
            os << "    }" << cB << ENDL;
        }
        else {
            os << "    glbSpkCnt" << n.first << "[0] = 0;" << ENDL;
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++) {" << ENDL;
            os << "        glbSpk" << n.first << "[i] = 0;" << ENDL;
            os << "    }" << cB << ENDL;
        }

        if (n.second.isSpikeEventRequired() && n.second.isDelayRequired()) {
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++) {" << ENDL;
            os << "        glbSpkCntEvnt" << n.first << "[i] = 0;" << ENDL;
            os << "    }" << cB << ENDL;
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++) {" << ENDL;
            os << "        glbSpkEvnt" << n.first << "[i] = 0;" << ENDL;
            os << "    }" << cB << ENDL;
        }
        else if (n.second.isSpikeEventRequired()) {
            os << "    glbSpkCntEvnt" << n.first << "[0] = 0;" << ENDL;
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++) {" << ENDL;
            os << "        glbSpkEvnt" << n.first << "[i] = 0;" << ENDL;
            os << "    }" << cB << ENDL;
        }

        if (n.second.isSpikeTimeRequired()) {
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++) {" << ENDL;
            os << "        sT" <<  n.first << "[i] = -10.0;" << ENDL;
            os << "    }" << cB << ENDL;
        }
        
        auto neuronModelVars = n.second.getNeuronModel()->getVars();
        for (size_t j = 0; j < neuronModelVars.size(); j++) {
            if (n.second.isVarQueueRequired(neuronModelVars[j].first)) {
                os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++) {" << ENDL;
            }
            else {
                os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++) {" << ENDL;
            }
            if (neuronModelVars[j].second == model.ftype) {
                os << "        " << neuronModelVars[j].first << n.first << "[i] = " << model.scalarExpr(n.second.getInitVals()[j]) << ";" << ENDL;
            }
            else {
                os << "        " << neuronModelVars[j].first << n.first << "[i] = " << n.second.getInitVals()[j] << ";" << ENDL;
            }
            os << "    }" << cB << ENDL;
        }

        if (n.second.getNeuronModel()->isPoisson()) {
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++) {" << ENDL;
            os << "        seed" << n.first << "[i] = rand();" << ENDL;
            os << "    }" << cB << ENDL;
        }

        /*if ((model.neuronType[i] == IZHIKEVICH) && (model.dt != 1.0)) {
            os << "    fprintf(stderr,\"WARNING: You use a time step different than 1 ms. Izhikevich model behaviour may not be robust.\\n\"); " << ENDL;
        }*/
    }
    os << ENDL;

    // INITIALISE SYNAPSE VARIABLES
    os << "    // synapse variables" << ENDL;
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
        const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();

        os << "    " << oB << "for (int i = 0; i < " << numTrgNeurons << "; i++) {" << ENDL;
        os << "        inSyn" << s.first << "[i] = " << model.scalarExpr(0.0) << ";" << ENDL;
        os << "    }" << cB << ENDL;

        if ((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
            auto wuVars = wu->getVars();
            for (size_t k= 0, l= wuVars.size(); k < l; k++) {
                os << "    " << oB << "for (int i = 0; i < " << numSrcNeurons * numTrgNeurons << "; i++) {" << ENDL;
                if (wuVars[k].second == model.ftype) {
                    os << "        " << wuVars[k].first << s.first << "[i] = " << model.scalarExpr(s.second.getWUInitVals()[k]) << ";" << ENDL;
                }
                else {
                    os << "        " << wuVars[k].first << s.first << "[i] = " << s.second.getWUInitVals()[k] << ";" << ENDL;
                }
        
                os << "    }" << cB << ENDL;
            }
        }

        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            auto psmVars = psm->getVars();
            for (size_t k= 0, l= psmVars.size(); k < l; k++) {
                os << "    " << oB << "for (int i = 0; i < " << numTrgNeurons << "; i++) {" << ENDL;
                if (psmVars[k].second == model.ftype) {
                    os << "        " << psmVars[k].first << s.first << "[i] = " << model.scalarExpr(s.second.getPSInitVals()[k]) << ";" << ENDL;
                }
                else {
                    os << "        " << psmVars[k].first << s.first << "[i] = " << s.second.getPSInitVals()[k] << ";" << ENDL;
                }
                os << "    }" << cB << ENDL;
            }
        }
    }
    os << ENDL << ENDL;
#ifndef CPU_ONLY
    os << "    copyStateToDevice();" << ENDL << ENDL;
    os << "    //initializeAllSparseArrays(); //I comment this out instead of removing to keep in mind that sparse arrays need to be initialised manually by hand later" << ENDL;
#endif
    os << "}" << ENDL << ENDL;


    // ------------------------------------------------------------------------
    // allocating conductance arrays for sparse matrices

    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << "void allocate" << s.first << "(unsigned int connN)" << "{" << ENDL;
            os << "// Allocate host side variables" << ENDL;
            os << "  C" << s.first << ".connN= connN;" << ENDL;

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
                os << "  C" << s.first << ".preInd= NULL;" << ENDL;
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
                os << "  C" << s.first << ".revIndInG= NULL;" << ENDL;
                os << "  C" << s.first << ".revInd= NULL;" << ENDL;
                os << "  C" << s.first << ".remap= NULL;" << ENDL;
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

            os << "}" << ENDL;
            os << ENDL;
            //setup up helper fn for this (specific) popn to generate sparse from dense
            os << "void createSparseConnectivityFromDense" << s.first << "(int preN,int postN, " << model.ftype << " *denseMatrix)" << "{" << ENDL;
            os << "    gennError(\"The function createSparseConnectivityFromDense" << s.first << "() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \\n Please use your own logic and use the general tools allocate" << s.first << "(), countEntriesAbove(), and setSparseConnectivityFromDense().\");" << ENDL;
            os << "}" << ENDL;
            os << ENDL;
        }
    }

    // ------------------------------------------------------------------------
    // initializing sparse arrays

#ifndef CPU_ONLY
    os << "void initializeAllSparseArrays() {" << ENDL;
    if(any_of(begin(model.getSynapseGroups()), end(model.getSynapseGroups()),
        [](const std::pair<string, SynapseGroup> &s)
        {
            return (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE);

        }))
    {
        os << "size_t size;" << ENDL;
    }

    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE){
            os << "size = C" << s.first << ".connN;" << ENDL;
            os << "  initializeSparseArray(C" << s.first << ",";
            os << " d_ind" << s.first << ",";
            os << " d_indInG" << s.first << ",";
            os << s.second.getSrcNeuronGroup()->getNumNeurons() <<");" << ENDL;
            if (model.isSynapseGroupDynamicsRequired(s.first)) {
                os << "  initializeSparseArrayPreInd(C" << s.first << ",";
                os << " d_preInd" << s.first << ");" << ENDL;
            }
            if (model.isSynapseGroupPostLearningRequired(s.first)) {
                os << "  initializeSparseArrayRev(C" << s.first << ",";
                os << "  d_revInd" << s.first << ",";
                os << "  d_revIndInG" << s.first << ",";
                os << "  d_remap" << s.first << ",";
                os << s.second.getTrgNeuronGroup()->getNumNeurons() <<");" << ENDL;
            }
           
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : s.second.getWUModel()->getVars()) {
                    if(!s.second.isWUVarZeroCopyEnabled(v.first)) {
                        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << s.first << ", "  << v.first << s.first << ", sizeof(" << v.second << ") * size , cudaMemcpyHostToDevice));" << ENDL;
                    }
                }
            }
        }
    }
    os << "}" << ENDL; 
    os << ENDL;
#endif

    // ------------------------------------------------------------------------
    // initialization of variables, e.g. reverse sparse arrays etc. 
    // that the user would not want to worry about
    
    os << "void init" << model.name << "()" << ENDL;
    os << OB(1130) << ENDL;
    bool anySparse = false;
    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            anySparse = true;
            if (model.isSynapseGroupDynamicsRequired(s.first)) {
                os << "createPreIndices(" << s.second.getSrcNeuronGroup()->getNumNeurons() << ", " << s.second.getTrgNeuronGroup()->getNumNeurons() << ", &C" << s.first << ");" << ENDL;
            }
            if (model.isSynapseGroupPostLearningRequired(s.first)) {
                os << "createPosttoPreArray(" << s.second.getSrcNeuronGroup()->getNumNeurons() << ", " << s.second.getTrgNeuronGroup()->getNumNeurons() << ", &C" << s.first << ");" << ENDL;
            }
        }
    }

    if (anySparse) {
#ifndef CPU_ONLY
        os << "initializeAllSparseArrays();" << ENDL;
#endif
    }

    os << CB(1130) << ENDL;

    // ------------------------------------------------------------------------
    // freeing global memory structures

    os << "void freeMem()" << ENDL;
    os << "{" << ENDL;

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
            os << "    C" << s.first << ".connN= 0;" << ENDL;

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
    os << "}" << ENDL << ENDL;


    // ------------------------------------------------------------------------
    //! \brief Method for cleaning up and resetting device while quitting GeNN

    os << "void exitGeNN(){" << ENDL;  
    os << "  freeMem();" << ENDL;
#ifndef CPU_ONLY
    os << "  cudaDeviceReset();" << ENDL;
#endif
    os << "}" << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// the actual time stepping procedure (using CPU)" << ENDL;
    os << "void stepTimeCPU()" << ENDL;
    os << "{" << ENDL;
    if (!model.getSynapseGroups().empty()) {
        if (!model.getSynapseDynamicsGroups().empty()) {
            if (model.timing) os << "        synDyn_timer.startTimer();" << ENDL;
            os << "        calcSynapseDynamicsCPU(t);" << ENDL;
            if (model.timing) {
                os << "        synDyn_timer.stopTimer();" << ENDL;
                os << "        synDyn_tme+= synDyn_timer.getElapsedTime();" << ENDL;
            }
        }
        if (model.timing) os << "        synapse_timer.startTimer();" << ENDL;
        os << "        calcSynapsesCPU(t);" << ENDL;
        if (model.timing) {
            os << "        synapse_timer.stopTimer();" << ENDL;
            os << "        synapse_tme+= synapse_timer.getElapsedTime();"<< ENDL;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
            if (model.timing) os << "        learning_timer.startTimer();" << ENDL;
            os << "        learnSynapsesPostHost(t);" << ENDL;
            if (model.timing) {
                os << "        learning_timer.stopTimer();" << ENDL;
                os << "        learning_tme+= learning_timer.getElapsedTime();" << ENDL;
            }
        }
    }
    if (model.timing) os << "    neuron_timer.startTimer();" << ENDL;
    os << "    calcNeuronsCPU(t);" << ENDL;
    if (model.timing) {
        os << "    neuron_timer.stopTimer();" << ENDL;
        os << "    neuron_tme+= neuron_timer.getElapsedTime();" << ENDL;
    }
    os << "iT++;" << ENDL;
    os << "t= iT*DT;" << ENDL;
    os << "}" << ENDL;
    os.close();


    // ------------------------------------------------------------------------
    // finish up

#ifndef CPU_ONLY
    cout << "Global memory required for core model: " << mem/1e6 << " MB. " << ENDL;
    cout << deviceProp[theDevice].totalGlobalMem << " for device " << theDevice << ENDL;  
  
    if (0.5 * deviceProp[theDevice].totalGlobalMem < mem) {
        cout << "memory required for core model (" << mem/1e6;
        cout << "MB) is more than 50% of global memory on the chosen device";
        cout << "(" << deviceProp[theDevice].totalGlobalMem/1e6 << "MB)." << ENDL;
        cout << "Experience shows that this is UNLIKELY TO WORK ... " << ENDL;
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
    ofstream os;

//    cout << "entering GenRunnerGPU" << ENDL;
    string name= path + "/" + model.name + "_CODE/runnerGPU.cc";
    os.open(name.c_str());
    writeHeader(os);

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file runnerGPU.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the host side code for a GPU simulator version." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;
    os << ENDL;
    int version;
    cudaRuntimeGetVersion(&version); 
    if ((deviceProp[theDevice].major < 6) || (version < 8000)){
        //os << "#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600" << ENDL;
        //os << "#else"<< ENDL;
        //os << "#if __CUDA_ARCH__ < 600" << ENDL;
        os << "// software version of atomic add for double precision" << ENDL;
        os << "__device__ double atomicAddSW(double* address, double val)" << ENDL;
        os << "{" << ENDL;
        os << "    unsigned long long int* address_as_ull =" << ENDL;
        os << "                                          (unsigned long long int*)address;" << ENDL;
        os << "    unsigned long long int old = *address_as_ull, assumed;" << ENDL;
        os << "    do {" << ENDL;
        os << "        assumed = old;" << ENDL;
        os << "        old = atomicCAS(address_as_ull, assumed, " << ENDL;
        os << "                        __double_as_longlong(val + " << ENDL;
        os << "                        __longlong_as_double(assumed)));" << ENDL;
        os << "    } while (assumed != old);" << ENDL;
        os << "    return __longlong_as_double(old);" << ENDL;
        os << "}" << ENDL;
        //os << "#endif"<< ENDL;
        os << ENDL;
    }

    if (deviceProp[theDevice].major < 2) {
        os << "// software version of atomic add for single precision float" << ENDL;
        os << "__device__ float atomicAddSW(float* address, float val)" << ENDL;
        os << "{" << ENDL;
        os << "    int* address_as_ull =" << ENDL;
        os << "                                          (int*)address;" << ENDL;
        os << "    int old = *address_as_ull, assumed;" << ENDL;
        os << "    do {" << ENDL;
        os << "        assumed = old;" << ENDL;
        os << "        old = atomicCAS(address_as_ull, assumed, " << ENDL;
        os << "                        __float_as_int(val + " << ENDL;
        os << "                        __int_as_float(assumed)));" << ENDL;
        os << "    } while (assumed != old);" << ENDL;
        os << "    return __int_as_float(old);" << ENDL;
        os << "}" << ENDL;
        os << ENDL;
    }

    os << "#include \"neuronKrnl.cc\"" << ENDL;
    if (!model.getSynapseGroups().empty()) {
        os << "#include \"synapseKrnl.cc\"" << ENDL;
    }

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying things to device" << ENDL << ENDL;

    for(const auto &n : model.getNeuronGroups()) {
        // neuron state variables
        os << "void push" << n.first << "StateToDevice()" << ENDL;
        os << OB(1050);

        for(const auto &v : n.second.getNeuronModel()->getVars()) {
            // only copy non-zero-copied, non-pointers. Pointers don't transport between GPU and CPU
            if (v.second.find("*") == string::npos && !n.second.isVarZeroCopyEnabled(v.first)) {
                const size_t size = n.second.isVarQueueRequired(v.first)
                    ? n.second.getNumNeurons() * n.second.getNumDelaySlots()
                    : n.second.getNumNeurons();
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << n.first;
                os << ", " << v.first << n.first;
                os << ", " << size << " * sizeof(" << v.second << "), cudaMemcpyHostToDevice));" << ENDL;
            }
        }

        os << CB(1050);
        os << ENDL;

        // neuron spike variables
        os << "void push" << n.first << "SpikesToDevice()" << ENDL;
        os << OB(1060);

        if(!n.second.isSpikeZeroCopyEnabled()) {
            const size_t glbSpkCntSize = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCnt" << n.first;
            os << ", glbSpkCnt" << n.first;
            os << ", " << glbSpkCntSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;

            const size_t glbSpkSize = n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons();
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpk" << n.first;
            os << ", glbSpk" << n.first;
            os << ", " << glbSpkSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
        }
        
        if (n.second.isSpikeEventRequired()) {
          os << "push" << n.first << "SpikeEventsToDevice();" << ENDL;
        }

        if (n.second.isSpikeTimeRequired() && !n.second.isSpikeTimeZeroCopyEnabled()) {
            size_t size = n.second.getNumNeurons() * n.second.getNumDelaySlots();
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_sT" << n.first;
            os << ", sT" << n.first;
            os << ", " << size << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
        }

        os << CB(1060);
        os << ENDL;

        // neuron spike variables
        os << "void push" << n.first << "SpikeEventsToDevice()" << ENDL;
        os << OB(1060);

        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
            const size_t glbSpkCntEventSize = n.second.getNumDelaySlots();
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvnt" << n.first;
            os << ", glbSpkCntEvnt" << n.first;
            os << ", " << glbSpkCntEventSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;

            const size_t glbSpkEventSize = n.second.getNumNeurons() * n.second.getNumDelaySlots();
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvnt" << n.first;
            os << ", glbSpkEvnt" << n.first;
            os << ", " << glbSpkEventSize << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
        }

        os << CB(1060);
        os << ENDL;

        // current neuron spike variables
        os << "void push" << n.first << "CurrentSpikesToDevice()" << ENDL;
        os << OB(1061);

        if(!n.second.isSpikeZeroCopyEnabled()) {
            if (n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCnt" << n.first;
                os << "+spkQuePtr" << n.first << ", glbSpkCnt" << n.first;
                os << "+spkQuePtr" << n.first;
                os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpk" << n.first;
                os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
                os << ", glbSpk" << n.first;
                os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
                os << ", " << "glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
            }
            else {
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCnt" << n.first;
                os << ", glbSpkCnt" << n.first;
                os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
                os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpk" << n.first;
                os << ", glbSpk" << n.first;
                os << ", " << "glbSpkCnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
            }
        }
        os << CB(1061);
        os << ENDL;

        // current neuron spike event variables
        os << "void push" << n.first << "CurrentSpikeEventsToDevice()" << ENDL;
        os << OB(1062);

        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
          if (n.second.isDelayRequired()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvnt" << n.first;
            os << "+spkQuePtr" << n.first << ", glbSpkCntEvnt" << n.first;
            os << "+spkQuePtr" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvnt" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", glbSpkEvnt" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", " << "glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
          }
          else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvnt" << n.first;
            os << ", glbSpkCntEvnt" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvnt" << n.first;
            os << ", glbSpkEvnt" << n.first;
            os << ", " << "glbSpkCntEvnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));" << ENDL;
          }
        }

        os << CB(1062);
        os << ENDL;
    }
    // synapse variables
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        os << "void push" << s.first << "StateToDevice()" << ENDL;
        os << OB(1100);

        const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
        const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();
        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) { // INDIVIDUALG
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                os << "size_t size = " << numSrcNeurons * numTrgNeurons << ";" << ENDL;
            }
            else {
                os << "size_t size = C" << s.first << ".connN;" << ENDL;
            }

            for(const auto &v : wu->getVars()) {
                 // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                if (v.second.find("*") == string::npos && !s.second.isWUVarZeroCopyEnabled(v.first)) {
                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << s.first;
                    os << ", " << v.first << s.first;
                    os << ", size * sizeof(" << v.second << "), cudaMemcpyHostToDevice));" << ENDL;
                }
            }

            for(const auto &v : psm->getVars()) {
                // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                if (v.second.find("*") == string::npos && !s.second.isPSVarZeroCopyEnabled(v.first)) {
                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << s.first;
                    os << ", " << v.first << s.first;
                    os << ", " << numTrgNeurons << " * sizeof(" << v.second << "), cudaMemcpyHostToDevice));" << ENDL;
                }
            }
        }
        else if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t size = (numSrcNeurons * numTrgNeurons) / 32 + 1;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << s.first;
            os << ", gp" << s.first;
            os << ", " << size << " * sizeof(uint32_t), cudaMemcpyHostToDevice));" << ENDL;
        }

        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyn" << s.first;
        os << ", inSyn" << s.first;
        os << ", " << numTrgNeurons << " * sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << ENDL;

        os << CB(1100);
        os << ENDL;
    }


    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying things from device" << ENDL << ENDL;

    for(const auto &n : model.getNeuronGroups()) {
        // neuron state variables
        os << "void pull" << n.first << "StateFromDevice()" << ENDL;
        os << OB(1050);
        
        for(const auto &v : n.second.getNeuronModel()->getVars()) {
            // only copy non-zero-copied, non-pointers. Pointers don't transport between GPU and CPU
            if (v.second.find("*") == string::npos && !n.second.isVarZeroCopyEnabled(v.first)) {
                const size_t size = n.second.isVarQueueRequired(v.first)
                    ? n.second.getNumNeurons() * n.second.getNumDelaySlots()
                    : n.second.getNumNeurons();

                os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << v.first << n.first;
                os << ", d_" << v.first << n.first;
                os << ", " << size << " * sizeof(" << v.second << "), cudaMemcpyDeviceToHost));" << ENDL;
            }
        }

        os << CB(1050);
        os << ENDL;

        // spike event variables
        os << "void pull" << n.first << "SpikeEventsFromDevice()" << ENDL;
        os << OB(1061);

        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
          const size_t glbSpkCntEvntSize = n.second.getNumDelaySlots();
          os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
          os << ", d_glbSpkCntEvnt" << n.first;
          os << ", " << glbSpkCntEvntSize << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;

          const size_t glbSpkEvntSize = n.second.getNumNeurons() * n.second.getNumDelaySlots();
          os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvnt" << n.first;
          os << ", d_glbSpkEvnt" << n.first;
          os << ", " << glbSpkEvntSize << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
        }

        os << CB(1061);
        os << ENDL;

        // neuron spike variables (including spike events)
        os << "void pull" << n.first << "SpikesFromDevice()" << ENDL;
        os << OB(1060);

        if(!n.second.isSpikeZeroCopyEnabled()) {
            size_t glbSpkCntSize = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << n.first;
            os << ", d_glbSpkCnt" << n.first;
            os << ", " << glbSpkCntSize << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << n.first;
            os << ", d_glbSpk" << n.first;
            os << ", " << "glbSpkCnt" << n.first << " [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
        }

        if (n.second.isSpikeEventRequired()) {
          os << "pull" << n.first << "SpikeEventsFromDevice();" << ENDL;
        }
        os << CB(1060);
        os << ENDL;

        // neuron spike times
        os << "void pull" << n.first << "SpikeTimesFromDevice()" << ENDL;
        os << OB(10601);
        os << "//Assumes that spike numbers are already copied back from the device" << ENDL;
        if (n.second.isSpikeTimeRequired() && !n.second.isSpikeTimeZeroCopyEnabled()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(sT" << n.first;
            os << ", d_sT" << n.first;
            os << ", " << "glbSpkCnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
        }

        os << CB(10601);
        os << ENDL;

        os << "void pull" << n.first << "CurrentSpikesFromDevice()" << ENDL;
        os << OB(1061);

        if(!n.second.isSpikeZeroCopyEnabled()) {
            if ((n.second.isTrueSpikeRequired()) && n.second.isDelayRequired()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << n.first;
            os << "+spkQuePtr" << n.first << ", d_glbSpkCnt" << n.first;
            os << "+spkQuePtr" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", d_glbSpk" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", " << "glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
            }
            else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << n.first;
            os << ", d_glbSpkCnt" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << n.first;
            os << ", d_glbSpk" << n.first;
            os << ", " << "glbSpkCnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
            }
        }

        os << CB(1061);
        os << ENDL;

        os << "void pull" << n.first << "CurrentSpikeEventsFromDevice()" << ENDL;
        os << OB(1062);

        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
          if (n.second.isDelayRequired()) {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
            os << "+spkQuePtr" << n.first << ", d_glbSpkCntEvnt" << n.first;
            os << "+spkQuePtr" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvnt" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", d_glbSpkEvnt" << n.first;
            os << "+(spkQuePtr" << n.first << "*" << n.second.getNumNeurons() << ")";
            os << ", " << "glbSpkCntEvnt" << n.first << "[spkQuePtr" << n.first << "] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
          }
          else {
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
            os << ", d_glbSpkCntEvnt" << n.first;
            os << ", sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvnt" << n.first;
            os << ", d_glbSpkEvnt" << n.first;
            os << ", " << "glbSpkCntEvnt" << n.first << "[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
          }
        }

        os << CB(1062);
        os << ENDL;
    }

    // synapse variables
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
        const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();

        os << "void pull" << s.first << "StateFromDevice()" << ENDL;
        os << OB(1100);

        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) { // INDIVIDUALG
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) {
                os << "size_t size = " << numSrcNeurons * numTrgNeurons << ";" << ENDL;
            }
            else {
                os << "size_t size = C" << s.first << ".connN;" << ENDL;
            }

            for(const auto &v : wu->getVars()) {
                // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                if (v.second.find("*") == string::npos && !s.second.isWUVarZeroCopyEnabled(v.first)) {
                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << v.first << s.first;
                    os << ", d_"  << v.first << s.first;
                    os << ", size * sizeof(" << v.second << "), cudaMemcpyDeviceToHost));" << ENDL;
                }
            }

            for(const auto &v : psm->getVars()) {
                // only copy non-pointers and non-zero-copied. Pointers don't transport between GPU and CPU
                if (v.second.find("*") == string::npos && !s.second.isPSVarZeroCopyEnabled(v.first)) {
                    os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << v.first << s.first;
                    os << ", d_"  << v.first << s.first;
                    os << ", " << numTrgNeurons << " * sizeof(" << v.second << "), cudaMemcpyDeviceToHost));" << ENDL;
                }
            }
        }
        else if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            size_t size = (numSrcNeurons * numTrgNeurons) / 32 + 1;
            os << "CHECK_CUDA_ERRORS(cudaMemcpy(gp" << s.first;
            os << ", d_gp" << s.first;
            os << ", " << size << " * sizeof(uint32_t), cudaMemcpyDeviceToHost));" << ENDL;
        }

        os << "CHECK_CUDA_ERRORS(cudaMemcpy(inSyn" << s.first;
        os << ", d_inSyn" << s.first;
        os << ", " << numTrgNeurons << " * sizeof(" << model.ftype << "), cudaMemcpyDeviceToHost));" << ENDL;

        os << CB(1100);
        os << ENDL;
    }


    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying values to device" << ENDL;
    
    os << "void copyStateToDevice()" << ENDL;
    os << OB(1110);

    for(const auto &n : model.getNeuronGroups()) {
        os << "push" << n.first << "StateToDevice();" << ENDL;
        os << "push" << n.first << "SpikesToDevice();" << ENDL;
    }

    for(const auto &s : model.getSynapseGroups()) {
        os << "push" << s.first << "StateToDevice();" << ENDL;
    }

    os << CB(1110);
    os << ENDL;
    
    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying spikes to device" << ENDL;
    
    os << "void copySpikesToDevice()" << ENDL;
    os << OB(1111);
    for(const auto &n : model.getNeuronGroups()) {
      os << "push" << n.first << "SpikesToDevice();" << ENDL;
    }
    os << CB(1111);
   
    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying current spikes to device" << ENDL;
    
    os << "void copyCurrentSpikesToDevice()" << ENDL;
    os << OB(1112);
    for(const auto &n : model.getNeuronGroups()) {
      os << "push" << n.first << "CurrentSpikesToDevice();" << ENDL;
    }
    os << CB(1112);
   
    
    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying spike events to device" << ENDL;
    
    os << "void copySpikeEventsToDevice()" << ENDL;
    os << OB(1113);
    for(const auto &n : model.getNeuronGroups()) {
      os << "push" << n.first << "SpikeEventsToDevice();" << ENDL;
    }
    os << CB(1113);
   
    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying current spikes to device" << ENDL;
    
    os << "void copyCurrentSpikeEventsToDevice()" << ENDL;
    os << OB(1114);
    for(const auto &n : model.getNeuronGroups()) {
      os << "push" << n.first << "CurrentSpikeEventsToDevice();" << ENDL;
    }
    os << CB(1114);

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying values from device" << ENDL;
    
    os << "void copyStateFromDevice()" << ENDL;
    os << OB(1120);
    
    for(const auto &n : model.getNeuronGroups()) {
        os << "pull" << n.first << "StateFromDevice();" << ENDL;
        os << "pull" << n.first << "SpikesFromDevice();" << ENDL;
    }

    for(const auto &s : model.getSynapseGroups()) {
        os << "pull" << s.first << "StateFromDevice();" << ENDL;
    }
    
    os << CB(1120);
    os << ENDL;


    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying spikes from device" << ENDL;
    
    os << "void copySpikesFromDevice()" << ENDL;
    os << OB(1121) << ENDL;

    for(const auto &n : model.getNeuronGroups()) {
      os << "pull" << n.first << "SpikesFromDevice();" << ENDL;
    }
    os << CB(1121) << ENDL;
    os << ENDL;
    
    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying current spikes from device" << ENDL;
    
    os << "void copyCurrentSpikesFromDevice()" << ENDL;
    os << OB(1122) << ENDL;

    for(const auto &n : model.getNeuronGroups()) {
      os << "pull" << n.first << "CurrentSpikesFromDevice();" << ENDL;
    }
    os << CB(1122) << ENDL;
    os << ENDL;
    

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying spike numbers from device (note, only use when only interested"<< ENDL;
    os << "// in spike numbers; copySpikesFromDevice() already includes this)" << ENDL;

    os << "void copySpikeNFromDevice()" << ENDL;
    os << OB(1123) << ENDL;

    for(const auto &n : model.getNeuronGroups()) {
        if(!n.second.isSpikeZeroCopyEnabled()) {
            size_t size = (n.second.isTrueSpikeRequired() && n.second.isDelayRequired())
                ? n.second.getNumDelaySlots() : 1;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << n.first;
            os << ", d_glbSpkCnt" << n.first << ", " << size << "* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
        }
    }

    os << CB(1123) << ENDL;
    os << ENDL;

    
    os << "// ------------------------------------------------------------------------"<< ENDL;
    os << "// global copying spikeEvents from device" << ENDL;
    
    os << "void copySpikeEventsFromDevice()" << ENDL;
    os << OB(1124) << ENDL;

    for(const auto &n : model.getNeuronGroups()) {
      os << "pull" << n.first << "SpikeEventsFromDevice();" << ENDL;
    }
    os << CB(1124) << ENDL;
    os << ENDL;
    
    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// copying current spikeEvents from device" << ENDL;
    
    os << "void copyCurrentSpikeEventsFromDevice()" << ENDL;
    os << OB(1125) << ENDL;

    for(const auto &n : model.getNeuronGroups()) {
      os << "pull" << n.first << "CurrentSpikeEventsFromDevice();" << ENDL;
    }
    os << CB(1125) << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// global copying spike event numbers from device (note, only use when only interested" << ENDL;
    os << "// in spike numbers; copySpikeEventsFromDevice() already includes this)" << ENDL;
    
    os << "void copySpikeEventNFromDevice()" << ENDL;
    os << OB(1126) << ENDL;

    for(const auto &n : model.getNeuronGroups()) {
        if (n.second.isSpikeEventRequired() && !n.second.isSpikeEventZeroCopyEnabled()) {
            const size_t size = n.second.isDelayRequired() ? n.second.getNumDelaySlots() : 1;

            os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << n.first;
            os << ", d_glbSpkCntEvnt" << n.first << ", " << size << "* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << ENDL;
        }
    }
    os << CB(1126) << ENDL;
    os << ENDL;

    os << "// ------------------------------------------------------------------------" << ENDL;
    os << "// the time stepping procedure (using GPU)" << ENDL;
    os << "void stepTimeGPU()" << ENDL;
    os << OB(1130) << ENDL;

    if (!model.getSynapseGroups().empty()) {
        unsigned int synapseGridSz = model.getSynapseKernelGridSize();
        os << "//model.padSumSynapseTrgN[model.synapseGrpN - 1] is " << synapseGridSz << ENDL;
        synapseGridSz = synapseGridSz / synapseBlkSz;
        os << "dim3 sThreads(" << synapseBlkSz << ", 1);" << ENDL;
        os << "dim3 sGrid(" << synapseGridSz << ", 1);" << ENDL;
        os << ENDL;
    }
    if (!model.getSynapsePostLearnGroups().empty()) {
        const unsigned int learnGridSz = ceil((float)model.getSynapsePostLearnGridSize() / learnBlkSz);
        os << "dim3 lThreads(" << learnBlkSz << ", 1);" << ENDL;
        os << "dim3 lGrid(" << learnGridSz << ", 1);" << ENDL;
        os << ENDL;
    }

    if (!model.getSynapseDynamicsGroups().empty()) {
        const unsigned int synDynGridSz = ceil((float)model.getSynapseDynamicsGridSize() / synDynBlkSz);
        os << "dim3 sDThreads(" << synDynBlkSz << ", 1);" << ENDL;
        os << "dim3 sDGrid(" << synDynGridSz << ", 1);" << ENDL;
        os << ENDL;
    }

    const unsigned int neuronGridSz = ceil((float) model.getNeuronGridSize() / neuronBlkSz);
    os << "dim3 nThreads(" << neuronBlkSz << ", 1);" << ENDL;
    if (neuronGridSz < (unsigned int)deviceProp[theDevice].maxGridSize[1]) {
        os << "dim3 nGrid(" << neuronGridSz << ", 1);" << ENDL;
    }
    else {
        int sqGridSize = ceil((float) sqrt((float) neuronGridSz));
        os << "dim3 nGrid(" << sqGridSize << ","<< sqGridSize <<");" << ENDL;
    }
    os << ENDL;
    if (!model.getSynapseGroups().empty()) {
        if (!model.getSynapseDynamicsGroups().empty()) {
            if (model.timing) {
                os << "cudaEventRecord(synDynStart);" << ENDL;
            }
            os << "calcSynapseDynamics <<< sDGrid, sDThreads >>> (";
            for(const auto &p : model.getSynapseDynamicsKernelParameters()) {
                os << p.first << ", ";
            }
            os << "t);" << ENDL;
            if (model.timing) {
                os << "cudaEventRecord(synDynStop);" << ENDL;
            }
        }
        if (model.timing) {
            os << "cudaEventRecord(synapseStart);" << ENDL;
        }
        os << "calcSynapses <<< sGrid, sThreads >>> (";
        for(const auto &p : model.getSynapseKernelParameters()) {
            os << p.first << ", ";
        }
        os << "t);" << ENDL;
        if (model.timing) {
            os << "cudaEventRecord(synapseStop);" << ENDL;
        }

        if (!model.getSynapsePostLearnGroups().empty()) {
            if (model.timing) {
                os << "cudaEventRecord(learningStart);" << ENDL;
            }
            os << "learnSynapsesPost <<< lGrid, lThreads >>> (";
            for(const auto &p : model.getSimLearnPostKernelParameters()) {
                os << p.first << ", ";
            }
            os << "t);" << ENDL;
            if (model.timing) {
                os << "cudaEventRecord(learningStop);" << ENDL;
            }
        }
    }    
    for(auto &n : model.getNeuronGroups()) {
        if (n.second.isDelayRequired()) {
            os << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << ENDL;
        }
    }
    if (model.timing) {
        os << "cudaEventRecord(neuronStart);" << ENDL;
    }

    os << "calcNeurons <<< nGrid, nThreads >>> (";
    for(const auto &p : model.getNeuronKernelParameters()) {
        os << p.first << ", ";
    }
    os << "t);" << ENDL;
    if (model.timing) {
        os << "cudaEventRecord(neuronStop);" << ENDL;
        os << "cudaEventSynchronize(neuronStop);" << ENDL;
        os << "float tmp;" << ENDL;
        if (!model.getSynapseGroups().empty()) {
            os << "cudaEventElapsedTime(&tmp, synapseStart, synapseStop);" << ENDL;
            os << "synapse_tme+= tmp/1000.0;" << ENDL;
        }
        if (!model.getSynapsePostLearnGroups().empty()) {
            os << "cudaEventElapsedTime(&tmp, learningStart, learningStop);" << ENDL;
            os << "learning_tme+= tmp/1000.0;" << ENDL;
        }
        if (!model.getSynapseDynamicsGroups().empty()) {
            os << "cudaEventElapsedTime(&tmp, synDynStart, synDynStop);" << ENDL;
            os << "lsynDyn_tme+= tmp/1000.0;" << ENDL;
        }
        os << "cudaEventElapsedTime(&tmp, neuronStart, neuronStop);" << ENDL;
        os << "neuron_tme+= tmp/1000.0;" << ENDL;
    }

    // Synchronise if zero-copy is in use
    if(model.zeroCopyInUse()) {
        os << "cudaDeviceSynchronize();" << ENDL;
    }

    os << "iT++;" << ENDL;
    os << "t= iT*DT;" << ENDL;
    os << CB(1130) << ENDL;
    os.close();
    //cout << "done with generating GPU runner" << ENDL;
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
    string name = path + "/" + model.name + "_CODE/Makefile";
    ofstream os;
    os.open(name.c_str());

#ifdef _WIN32

#ifdef CPU_ONLY
    string cxxFlags = "/c /DCPU_ONLY";
    cxxFlags += " " + GENN_PREFERENCES::userCxxFlagsWIN;
    if (GENN_PREFERENCES::optimizeCode) cxxFlags += " /O2";
    if (GENN_PREFERENCES::debugCode) cxxFlags += " /debug /Zi /Od";

    os << endl;
    os << "CXXFLAGS       =/nologo /EHsc " << cxxFlags << endl;
    os << endl;
    os << "INCLUDEFLAGS   =/I\"$(GENN_PATH)\\lib\\include\"" << endl;
    os << endl;
    os << "all: runner.obj" << endl;
    os << endl;
    os << "runner.obj: runner.cc" << endl;
    os << "\t$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) runner.cc" << endl;
    os << endl;
    os << "clean:" << endl;
    os << "\t-del runner.obj 2>nul" << endl;
#else
    string nvccFlags = "-c -x cu -arch sm_";
    nvccFlags += deviceProp[theDevice].major + deviceProp[theDevice].minor;
    nvccFlags += " " + GENN_PREFERENCES::userNvccFlags;
    if (GENN_PREFERENCES::optimizeCode) nvccFlags += " -O3 -use_fast_math";
    if (GENN_PREFERENCES::debugCode) nvccFlags += " -O0 -g -G";
    if (GENN_PREFERENCES::showPtxInfo) nvccFlags += " -Xptxas \"-v\"";

    os << endl;
    os << "NVCC           =\"" << NVCC << "\"" << endl;
    os << "NVCCFLAGS      =" << nvccFlags << endl;
    os << endl;
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)\\lib\\include\"" << endl;
    os << endl;
    os << "all: runner.obj" << endl;
    os << endl;
    os << "runner.obj: runner.cc" << endl;
    os << "\t$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc" << endl;
    os << endl;
    os << "clean:" << endl;
    os << "\t-del runner.obj 2>nul" << endl;
#endif

#else // UNIX

#ifdef CPU_ONLY
    string cxxFlags = "-c -DCPU_ONLY";
    cxxFlags += " " + GENN_PREFERENCES::userCxxFlagsGNU;
    if (GENN_PREFERENCES::optimizeCode) cxxFlags += " -O3 -ffast-math";
    if (GENN_PREFERENCES::debugCode) cxxFlags += " -O0 -g";

    os << endl;
    os << "CXXFLAGS       :=" << cxxFlags << endl;
    os << endl;
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)/lib/include\"" << endl;
    os << endl;
    os << "all: runner.o" << endl;
    os << endl;
    os << "runner.o: runner.cc" << endl;
    os << "\t$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) runner.cc" << endl;
    os << endl;
    os << "clean:" << endl;
    os << "\trm -f runner.o" << endl;
#else
    string nvccFlags = "-c -x cu -arch sm_";
    nvccFlags += to_string(deviceProp[theDevice].major) + to_string(deviceProp[theDevice].minor);
    nvccFlags += " " + GENN_PREFERENCES::userNvccFlags;
    if (GENN_PREFERENCES::optimizeCode) nvccFlags += " -O3 -use_fast_math -Xcompiler \"-ffast-math\"";
    if (GENN_PREFERENCES::debugCode) nvccFlags += " -O0 -g -G";
    if (GENN_PREFERENCES::showPtxInfo) nvccFlags += " -Xptxas \"-v\"";

    os << endl;
    os << "NVCC           :=\"" << NVCC << "\"" << endl;
    os << "NVCCFLAGS      :=" << nvccFlags << endl;
    os << endl;
    os << "INCLUDEFLAGS   =-I\"$(GENN_PATH)/lib/include\"" << endl;
    os << endl;
    os << "all: runner.o" << endl;
    os << endl;
    os << "runner.o: runner.cc" << endl;
    os << "\t$(NVCC) $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc" << endl;
    os << endl;
    os << "clean:" << endl;
    os << "\trm -f runner.o" << endl;
#endif

#endif

    os.close();
}
