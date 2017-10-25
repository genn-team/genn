/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model definitions.
  
--------------------------------------------------------------------------*/

#ifndef MODELSPEC_CC
#define MODELSPEC_CC

// GeNN includes
#include "codeGenUtils.h"
#include "global.h"
#include "modelSpec.h"
#include "standardSubstitutions.h"
#include "utils.h"

// Standard includes
#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>

unsigned int GeNNReady = 0;

// ------------------------------------------------------------------------
//! \brief Method for GeNN initialisation (by preparing standard models)
    
void initGeNN()
{
    prepareStandardModels();
    preparePostSynModels();
    prepareWeightUpdateModels();
    GeNNReady= 1;
}

// ------------------------------------------------------------------------
// NNmodel
// ------------------------------------------------------------------------
// class NNmodel for specifying a neuronal network model

NNmodel::NNmodel() 
{
    final= false;
    setDT(0.5);
    setPrecision(GENN_FLOAT);
    setTiming(false);
    RNtype= "uint64_t";
#ifndef CPU_ONLY
    setGPUDevice(AUTODEVICE);
#endif
    setSeed(0);
}

NNmodel::~NNmodel() 
{
}

void NNmodel::setName(const string &inname)
{
    if (final) {
        gennError("Trying to set the name of a finalized model.");
    }
    name= inname;
}

bool NNmodel::zeroCopyInUse() const
{
    // If any neuron groups use zero copy return true
    if(any_of(begin(m_NeuronGroups), end(m_NeuronGroups),
        [](const NeuronGroupValueType &n){ return n.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

    // If any synapse groups use zero copy return true
    if(any_of(begin(m_SynapseGroups), end(m_SynapseGroups),
        [](const SynapseGroupValueType &s){ return s.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

    return false;
}

bool NNmodel::isRNGRequired() const
{
    // If any neuron groups require an RNG return true
    if(any_of(begin(m_NeuronGroups), end(m_NeuronGroups),
        [](const NeuronGroupValueType &n){ return n.second.isRNGRequired(); }))
    {
        return true;
    }

    return false;
}

//--------------------------------------------------------------------------
/*! \brief This function is for setting which host and which device a neuron group will be simulated on
 */
//--------------------------------------------------------------------------

void NNmodel::setNeuronClusterIndex(const string &neuronGroup, /**< Name of the neuron population */
                                    int hostID, /**< ID of the host */
                                    int deviceID /**< ID of the device */)
{
    findNeuronGroup(neuronGroup)->setClusterIndex(hostID, deviceID);
}


//--------------------------------------------------------------------------
/*! \brief This function is for setting which host and which device a synapse group will be simulated on
 */
//--------------------------------------------------------------------------

void NNmodel::setSynapseClusterIndex(const string &synapseGroup, /**< Name of the synapse population */
                                     int hostID, /**< ID of the host */
                                     int deviceID /**< ID of the device */)
{
    findSynapseGroup(synapseGroup)->setClusterIndex(hostID, deviceID);
}

unsigned int NNmodel::getNeuronGridSize() const
{
    if(m_NeuronGroups.empty()) {
        return 0;
    }
    else {
        return m_NeuronGroups.rbegin()->second.getPaddedIDRange().second;
    }

}

unsigned int NNmodel::getNumNeurons() const
{
    if(m_NeuronGroups.empty()) {
        return 0;
    }
    else {
        return m_NeuronGroups.rbegin()->second.getIDRange().second;
    }
}

const NeuronGroup *NNmodel::findNeuronGroup(const std::string &name) const
{
    auto neuronGroup = m_NeuronGroups.find(name);

    if(neuronGroup == m_NeuronGroups.cend())
    {
        gennError("neuron group " + name + " not found, aborting ...");
        return NULL;
    }
    else
    {
        return &neuronGroup->second;
    }
}

NeuronGroup *NNmodel::findNeuronGroup(const std::string &name)
{
    auto neuronGroup = m_NeuronGroups.find(name);

    if(neuronGroup == m_NeuronGroups.cend())
    {
        gennError("neuron group " + name + " not found, aborting ...");
        return NULL;
    }
    else
    {
        return &neuronGroup->second;
    }
}

//--------------------------------------------------------------------------
/*! \overload

  This function adds a neuron population to a neuronal network models, assigning the name, the number of neurons in the group, the neuron type, parameters and initial values, the latter two defined as double *
 */
//--------------------------------------------------------------------------

NeuronGroup *NNmodel::addNeuronPopulation(
  const string &name, /**<  The name of the neuron population*/
  unsigned int nNo, /**<  Number of neurons in the population */
  unsigned int type, /**<  Type of the neurons, refers to either a standard type or user-defined type*/
  const double *p, /**< Parameters of this neuron type */
  const double *ini /**< Initial values for variables of this neuron type */)
{
  vector<double> vp;
  vector<double> vini;
  for (size_t i= 0; i < nModels[type].pNames.size(); i++) {
    vp.push_back(p[i]);
  }
  for (size_t i= 0; i < nModels[type].varNames.size(); i++) {
    vini.push_back(ini[i]);
  }
  return addNeuronPopulation(name, nNo, type, vp, vini);
}
  

//--------------------------------------------------------------------------
/*! \brief This function adds a neuron population to a neuronal network models, assigning the name, the number of neurons in the group, the neuron type, parameters and initial values. The latter two defined as STL vectors of double.
 */
//--------------------------------------------------------------------------

NeuronGroup *NNmodel::addNeuronPopulation(
  const string &name, /**<  The name of the neuron population*/
  unsigned int nNo, /**<  Number of neurons in the population */
  unsigned int type, /**<  Type of the neurons, refers to either a standard type or user-defined type*/
  const vector<double> &p, /**< Parameters of this neuron type */
  const vector<double> &ini /**< Initial values for variables of this neuron type */)
{
    if (!GeNNReady) {
        gennError("You need to call initGeNN first.");
    }
    if (final) {
        gennError("Trying to add a neuron population to a finalized model.");
    }
    if (p.size() != nModels[type].pNames.size()) {
        gennError("The number of parameter values for neuron group " + name + " does not match that of their neuron type, " + to_string(p.size()) + " != " + to_string(nModels[type].pNames.size()));
    }
    if (ini.size() != nModels[type].varNames.size()) {
        gennError("The number of variable initial values for neuron group " + name + " does not match that of their neuron type, " + to_string(ini.size()) + " != " + to_string(nModels[type].varNames.size()));
    }

    // Add neuron group
    auto result = m_NeuronGroups.insert(
        pair<string, NeuronGroup>(name, NeuronGroup(name, nNo, new NeuronModels::LegacyWrapper(type), p, ini)));

    if(!result.second)
    {
        gennError("Cannot add a neuron population with duplicate name:" + name);
        return NULL;
    }
    else
    {
        return &result.first->second;
    }
}


//--------------------------------------------------------------------------
/*! \brief This function defines the type of the explicit input to the neuron model. Current options are common constant input to all neurons, input  from a file and input defines as a rule.
*/ 
//--------------------------------------------------------------------------
void NNmodel::activateDirectInput(
  const string &, /**< Name of the neuron population */
  unsigned int /**< Type of input: 1 if common input, 2 if custom input from file, 3 if custom input as a rule*/)
{
    gennError("This function has been deprecated since GeNN 2.2. Use neuron variables, extraGlobalNeuronKernelParameters, or parameters instead.");
}

unsigned int NNmodel::getSynapseKernelGridSize() const
{
    if(m_SynapseGroups.empty()) {
        return 0;
    }
    else {
        return  m_SynapseGroups.rbegin()->second.getPaddedKernelIDRange().second;
    }

}

unsigned int NNmodel::getSynapsePostLearnGridSize() const
{
    if(m_SynapsePostLearnGroups.empty()) {
        return 0;
    }
    else {
        return m_SynapsePostLearnGroups.rbegin()->second.second;
    }
}

unsigned int NNmodel::getSynapseDynamicsGridSize() const
{
    if(m_SynapseDynamicsGroups.empty()) {
        return 0;
    }
    else {
        return m_SynapseDynamicsGroups.rbegin()->second.second;
    }
}

const SynapseGroup *NNmodel::findSynapseGroup(const std::string &name) const
{
    auto synapseGroup = m_SynapseGroups.find(name);

    if(synapseGroup == m_SynapseGroups.cend())
    {
        gennError("synapse group " + name + " not found, aborting ...");
        return NULL;
    }
    else
    {
        return &synapseGroup->second;
    }
}

SynapseGroup *NNmodel::findSynapseGroup(const std::string &name)
{
    auto synapseGroup = m_SynapseGroups.find(name);

    if(synapseGroup == m_SynapseGroups.cend())
    {
        gennError("synapse group " + name + " not found, aborting ...");
        return NULL;
    }
    else
    {
        return &synapseGroup->second;
    }
}

bool NNmodel::isSynapseGroupDynamicsRequired(const std::string &name) const
{
    return (m_SynapseDynamicsGroups.find(name) != end(m_SynapseDynamicsGroups));
}

bool NNmodel::isSynapseGroupPostLearningRequired(const std::string &name) const
{
    return (m_SynapsePostLearnGroups.find(name) != end(m_SynapsePostLearnGroups));
}

//--------------------------------------------------------------------------
/*! \overload

  This deprecated function is provided for compatibility with the previous release of GeNN.
 * Default values are provide for new parameters, it is strongly recommended these be selected explicity via the new version othe function
 */
//--------------------------------------------------------------------------

SynapseGroup *NNmodel::addSynapsePopulation(
  const string &, /**<  The name of the synapse population*/
  unsigned int, /**< The type of synapse to be added (i.e. learning mode) */
  SynapseConnType, /**< The type of synaptic connectivity*/
  SynapseGType, /**< The way how the synaptic conductivity g will be defined*/
  const string &, /**< Name of the (existing!) pre-synaptic neuron population*/
  const string &, /**< Name of the (existing!) post-synaptic neuron population*/
  const double*) /**< A C-type array of doubles that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses.*/
{
  gennError("This version of addSynapsePopulation() has been deprecated since GeNN 2.2. Please use the newer addSynapsePopulation functions instead.");
  return NULL;
}


//--------------------------------------------------------------------------
/*! \brief Overloaded old version (deprecated)
*/
//--------------------------------------------------------------------------

SynapseGroup *NNmodel::addSynapsePopulation(
  const string &name, /**<  The name of the synapse population*/
  unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
  SynapseConnType conntype, /**< The type of synaptic connectivity*/
  SynapseGType gtype, /**< The way how the synaptic conductivity g will be defined*/
  unsigned int delaySteps, /**< Number of delay slots*/
  unsigned int postsyn, /**< Postsynaptic integration method*/
  const string &src, /**< Name of the (existing!) pre-synaptic neuron population*/
  const string &trg, /**< Name of the (existing!) post-synaptic neuron population*/
  const double *p, /**< A C-type array of doubles that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses.*/
  const double* PSVini, /**< A C-type array of doubles that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const double *ps /**< A C-type array of doubles that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses.*/)
{
    cerr << "!!!!!!GeNN WARNING: This function has been deprecated since GeNN 2.2, and will be removed in a future release. You use the overloaded method which passes a null pointer for the initial values of weight update variables. If you use a method that uses synapse variables, please add a pointer to this vector in the function call, like:\n          addSynapsePopulation(name, syntype, conntype, gtype, NO_DELAY, EXPDECAY, src, target, double * SYNVARINI, params, postSynV,postExpSynapsePopn);" << endl;
    const double *iniv = NULL;
    return addSynapsePopulation(name, syntype, conntype, gtype, delaySteps, postsyn, src, trg, iniv, p, PSVini, ps);
}


//--------------------------------------------------------------------------
/*! \brief This function adds a synapse population to a neuronal network model, assigning the name, the synapse type, the connectivity type, the type of conductance specification, the source and destination neuron populations, and the synaptic parameters.
 */
//--------------------------------------------------------------------------

SynapseGroup *NNmodel::addSynapsePopulation(
  const string &name, /**<  The name of the synapse population*/
  unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
  SynapseConnType conntype, /**< The type of synaptic connectivity*/
  SynapseGType gtype, /**< The way how the synaptic conductivity g will be defined*/
  unsigned int delaySteps, /**< Number of delay slots*/
  unsigned int postsyn, /**< Postsynaptic integration method*/
  const string &src, /**< Name of the (existing!) pre-synaptic neuron population*/
  const string &trg, /**< Name of the (existing!) post-synaptic neuron population*/
  const double* synini, /**< A C-type array of doubles that contains the initial values for synapse variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const double *p, /**< A C-type array of doubles that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses.*/
  const double* PSVini, /**< A C-type array of doubles that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const double *ps /**< A C-type array of doubles that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses.*/)
{
  vector<double> vsynini;
  for (size_t j= 0; j < weightUpdateModels[syntype].varNames.size(); j++) {
    vsynini.push_back(synini[j]);
  }
  vector<double> vp;
  for (size_t j= 0; j < weightUpdateModels[syntype].pNames.size(); j++) {
    vp.push_back(p[j]);
  }
  vector<double> vpsini;
  for (size_t j= 0; j < postSynModels[postsyn].varNames.size(); j++) {
    vpsini.push_back(PSVini[j]);
  }
  vector<double> vps;
  for (size_t j= 0; j <  postSynModels[postsyn].pNames.size(); j++) {
    vps.push_back(ps[j]);
  }
  return addSynapsePopulation(name, syntype, conntype, gtype, delaySteps, postsyn, src, trg, vsynini, vp, vpsini, vps);
}


//--------------------------------------------------------------------------
/*! \brief This function adds a synapse population to a neuronal network model, assigning the name, the synapse type, the connectivity type, the type of conductance specification, the source and destination neuron populations, and the synaptic parameters.
 */
//--------------------------------------------------------------------------

SynapseGroup *NNmodel::addSynapsePopulation(
  const string &name, /**<  The name of the synapse population*/
  unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
  SynapseConnType conntype, /**< The type of synaptic connectivity*/
  SynapseGType gtype, /**< The way how the synaptic conductivity g will be defined*/
  unsigned int delaySteps, /**< Number of delay slots*/
  unsigned int postsyn, /**< Postsynaptic integration method*/
  const string &src, /**< Name of the (existing!) pre-synaptic neuron population*/
  const string &trg, /**< Name of the (existing!) post-synaptic neuron population*/
  const vector<double> &synini, /**< A C-type array of doubles that contains the initial values for synapse variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const vector<double> &p, /**< A C-type array of doubles that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses.*/
  const vector<double> &PSVini, /**< A C-type array of doubles that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const vector<double> &ps /**< A C-type array of doubles that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses.*/)
{
    if (!GeNNReady) {
        gennError("You need to call initGeNN first.");
    }
    if (final) {
        gennError("Trying to add a synapse population to a finalized model.");
    }
    if (p.size() != weightUpdateModels[syntype].pNames.size()) {
        gennError("The number of presynaptic parameter values for synapse group " + name + " does not match that of their synapse type, " + to_string(p.size()) + " != " + to_string(weightUpdateModels[syntype].pNames.size()));
    }
    if (synini.size() != weightUpdateModels[syntype].varNames.size()) {
        gennError("The number of presynaptic variable initial values for synapse group " + name + " does not match that of their synapse type, " + to_string(synini.size()) + " != " + to_string(weightUpdateModels[syntype].varNames.size()));
    }
    if (ps.size() != postSynModels[postsyn].pNames.size()) {
        gennError("The number of presynaptic parameter values for synapse group " + name + " does not match that of their synapse type, " + to_string(ps.size()) + " != " + to_string(postSynModels[postsyn].pNames.size()));
    }
    if (PSVini.size() != postSynModels[postsyn].varNames.size()) {
        gennError("The number of presynaptic variable initial values for synapse group " + name + " does not match that of their synapse type, " + to_string(PSVini.size()) + " != " + to_string(postSynModels[postsyn].varNames.size()));
    }

    SynapseMatrixType mtype;
    if(conntype == SPARSE && gtype == GLOBALG)
    {
        mtype = SynapseMatrixType::SPARSE_GLOBALG;
    }
    else if(conntype == SPARSE && gtype == INDIVIDUALG)
    {
        mtype = SynapseMatrixType::SPARSE_INDIVIDUALG;
    }
    else if((conntype == DENSE || conntype == ALLTOALL) && gtype == INDIVIDUALG)
    {
        mtype = SynapseMatrixType::DENSE_INDIVIDUALG;
    }
    else if((conntype == DENSE || conntype == ALLTOALL) && gtype == GLOBALG)
    {
        mtype = SynapseMatrixType::DENSE_GLOBALG;
    }
    else if(gtype == INDIVIDUALID)
    {
        mtype = SynapseMatrixType::BITMASK_GLOBALG;
    }
    else
    {
        gennError("Combination of connection type " + to_string(conntype) + " and weight type " + to_string(gtype) + " not supported");
        return NULL;
    }

    auto srcNeuronGrp = findNeuronGroup(src);
    auto trgNeuronGrp = findNeuronGroup(trg);

    // Add synapse group
    auto result = m_SynapseGroups.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(name, mtype, delaySteps,
                              new WeightUpdateModels::LegacyWrapper(syntype), p, synini,
                              new PostsynapticModels::LegacyWrapper(postsyn), ps, PSVini,
                              srcNeuronGrp, trgNeuronGrp));

    if(!result.second)
    {
        gennError("Cannot add a synapse population with duplicate name:" + name);
        return NULL;
    }
    else
    {
        return &result.first->second;
    }
}


//--------------------------------------------------------------------------
/*! \brief This function defines the maximum number of connections for a neuron in the population
*/ 
//--------------------------------------------------------------------------

void NNmodel::setMaxConn(const string &sname, /**<  */
                         unsigned int maxConnP /**<  */)
{
    if (final) {
        gennError("Trying to set MaxConn in a finalized model.");
    }
    findSynapseGroup(sname)->setMaxConnections(maxConnP);

}


//--------------------------------------------------------------------------
/*! \brief This function defines the execution order of the synapses in the kernels
  (0 : execute for every postsynaptic neuron 1: execute for every presynaptic neuron)
 */ 
//--------------------------------------------------------------------------

void NNmodel::setSpanTypeToPre(const string &sname /**< name of the synapse group to which to apply the pre-synaptic span type */)
{
    if (final) {
        gennError("Trying to set spanType in a finalized model.");
    }
    findSynapseGroup(sname)->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);

}


//--------------------------------------------------------------------------
/*! \brief This functions sets the global value of the maximal synaptic conductance for a synapse population that was idfentified as conductance specifcation method "GLOBALG" 
 */
//--------------------------------------------------------------------------

void NNmodel::setSynapseG(const string &, /**<  */
                          double /**<  */)
{
    gennError("NOTE: This function has been deprecated as of GeNN 2.2. Please provide the correct initial values in \"addSynapsePopulation\" for all your variables and they will be the constant values in the GLOBALG mode.");
}


//--------------------------------------------------------------------------
/*! \brief This function sets a global input value to the specified neuron group.
 */
//--------------------------------------------------------------------------

void NNmodel::setConstInp(const string &, /**<  */
                          double /**<  */)
{
    gennError("This function has been deprecated as of GeNN 2.2. Use parameters in the neuron model instead.");
}


//--------------------------------------------------------------------------
/*! \brief This function sets the integration time step DT of the model
 */
//--------------------------------------------------------------------------

void NNmodel::setDT(double newDT /**<  */)
{
    if (final) {
        gennError("Trying to set DT in a finalized model.");
    }
    dt = newDT;
}


//--------------------------------------------------------------------------
/*! \brief This function sets the numerical precision of floating type variables. By default, it is GENN_GENN_FLOAT.
 */
//--------------------------------------------------------------------------

void NNmodel::setPrecision(FloatType floattype /**<  */)
{
    if (final) {
        gennError("Trying to set the precision of a finalized model.");
    }
    switch (floattype) {
    case GENN_FLOAT:
        ftype = "float";
        break;
    case GENN_DOUBLE:
        ftype = "double"; // not supported by compute capability < 1.3
        break;
    case GENN_LONG_DOUBLE:
        ftype = "long double"; // not supported by CUDA at the moment.
        break;
    default:
        gennError("Unrecognised floating-point type.");
    }
}


//--------------------------------------------------------------------------
/*! \brief This function sets a flag to determine whether timers and timing commands are to be included in generated code.
 */
//--------------------------------------------------------------------------

void NNmodel::setTiming(bool theTiming /**<  */)
{
    if (final) {
        gennError("Trying to set timing flag in a finalized model.");
    }
    timing= theTiming;
}


//--------------------------------------------------------------------------
/*! \brief This function sets the random seed. If the passed argument is > 0, automatic seeding is disabled. If the argument is 0, the underlying seed is obtained from the time() function.
 */
//--------------------------------------------------------------------------

void NNmodel::setSeed(unsigned int inseed /*!< the new seed  */)
{
    if (final) {
        gennError("Trying to set the random seed in a finalized model.");
    }
    seed= inseed;
}

//--------------------------------------------------------------------------
/*! \brief Sets the underlying type for random number generation (default: uint64_t)
 */
//--------------------------------------------------------------------------
void NNmodel::setRNType(const std::string &type)
{
    if (final) {
        gennError("Trying to set the random number type in a finalized model.");
    }
    RNtype= type;
}
#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief This function defines the way how the GPU is chosen. If "AUTODEVICE" (-1) is given as the argument, GeNN will use internal heuristics to choose the device. Otherwise the argument is the device number and the indicated device will be used.
*/ 
//--------------------------------------------------------------------------

void NNmodel::setGPUDevice(int device)
{
  int deviceCount;
  CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
  assert(device >= -1);
  assert(device < deviceCount);
  if (device == -1) GENN_PREFERENCES::autoChooseDevice= 1;
  else {
      GENN_PREFERENCES::autoChooseDevice= 0;
      GENN_PREFERENCES::defaultDevice= device;
  }
}
#endif


string NNmodel::scalarExpr(const double val) const
{
    string tmp;
    float fval= (float) val;
    if (ftype == "float") {
        tmp= to_string(fval) + "f";
    }
    if (ftype == "double") {
        tmp= to_string(val);
    }
    return tmp;
}

//--------------------------------------------------------------------------
/*! \brief Accumulate the sums and block-size-padded sums of all simulation groups.

  This method saves the neuron numbers of the populations rounded to the next multiple of the block size as well as the sums s(i) = sum_{1...i} n_i of the rounded population sizes. These are later used to determine the branching structure for the generated neuron kernel code. 
*/
//--------------------------------------------------------------------------

void NNmodel::setPopulationSums()
{
    // NEURON GROUPS
    unsigned int neuronIDStart = 0;
    unsigned int paddedNeuronIDStart = 0;
    for(auto &n : m_NeuronGroups) {
        n.second.calcSizes(neuronBlkSz, neuronIDStart, paddedNeuronIDStart);
    }

    // SYNAPSE groups
    unsigned int paddedSynapseKernelIDStart = 0;
    unsigned int paddedSynapseDynamicsIDStart = 0;
    unsigned int paddedSynapsePostLearnIDStart = 0;
    for(auto &s : m_SynapseGroups) {
        // Calculate synapse kernel sizes
        s.second.calcKernelSizes(synapseBlkSz, paddedSynapseKernelIDStart);

        if (!s.second.getWUModel()->getLearnPostCode().empty()) {
            const unsigned int startID = paddedSynapsePostLearnIDStart;
            paddedSynapsePostLearnIDStart += s.second.getPaddedPostLearnKernelSize(learnBlkSz);

            // Add this synapse group to map of synapse groups with postsynaptic learning
            // or update the existing entry with the new block sizes
            m_SynapsePostLearnGroups[s.first] = std::make_pair(startID, paddedSynapsePostLearnIDStart);
        }

         if (!s.second.getWUModel()->getSynapseDynamicsCode().empty()) {
            const unsigned int startID = paddedSynapseDynamicsIDStart;
            paddedSynapseDynamicsIDStart += s.second.getPaddedDynKernelSize(synDynBlkSz);

            // Add this synapse group to map of synapse groups with dynamics
            // or update the existing entry with the new block sizes
            m_SynapseDynamicsGroups[s.first] = std::make_pair(startID, paddedSynapseDynamicsIDStart);
         }
    }
}

void NNmodel::finalize()
{
    //initializing learning parameters to start
    if (final) {
        gennError("Your model has already been finalized");
    }
    final = true;

    // Loop through neuron populations and their outgoing synapse populations
    for(auto &n : m_NeuronGroups) {
        for(auto *sg : n.second.getOutSyn()) {
            const auto *wu = sg->getWUModel();

            if (!wu->getEventCode().empty()) {
                sg->setSpikeEventRequired(true);
                n.second.setSpikeEventRequired(true);
                assert(!wu->getEventThresholdConditionCode().empty());

                 // Create iteration context to iterate over derived and extra global parameters
                ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
                DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());

                // do an early replacement of parameters, derived parameters and extraglobalsynapse parameters
                string eCode = wu->getEventThresholdConditionCode();
                value_substitutions(eCode, wu->getParamNames(), sg->getWUParams());
                value_substitutions(eCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
                name_substitutions(eCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());

                // Add code and name of
                string supportCodeNamespaceName = wu->getSimSupportCode().empty() ?
                    "" : sg->getName() + "_weightupdate_simCode";

                // Add code and name of support code namespace to set
                n.second.addSpkEventCondition(eCode, supportCodeNamespaceName);

                // analyze which neuron variables need queues
                n.second.updateVarQueues(wu->getEventCode());
            }
        }
        if (n.second.getSpikeEventCondition().size() > 1) {
            for(auto *sg : n.second.getOutSyn()) {
                if (!sg->getWUModel()->getEventCode().empty()) {
                    sg->setEventThresholdReTestRequired(true);
                }
            }
        }
    }

    // NEURON GROUPS
    for(auto &n : m_NeuronGroups) {
        // Initialize derived parameters
        n.second.initDerivedParams(dt);

        // Make extra global parameter lists
        n.second.addExtraGlobalParams(neuronKernelParameters);
    }

    // SYNAPSE groups
    for(auto &s : m_SynapseGroups) {
        const auto *wu = s.second.getWUModel();

        // Initialize derived parameters
        s.second.initDerivedParams(dt);

        if (!wu->getSimCode().empty()) {
            s.second.setTrueSpikeRequired(true);
            s.second.getSrcNeuronGroup()->setTrueSpikeRequired(true);

            // analyze which neuron variables need queues
            s.second.getSrcNeuronGroup()->updateVarQueues(wu->getSimCode());
        }

        if (!wu->getLearnPostCode().empty()) {
            s.second.getSrcNeuronGroup()->updateVarQueues(wu->getLearnPostCode());
        }

        if (!wu->getSynapseDynamicsCode().empty()) {
            s.second.getSrcNeuronGroup()->updateVarQueues(wu->getSynapseDynamicsCode());
        }

        // Make extra global parameter lists
        s.second.addExtraGlobalNeuronParams(neuronKernelParameters);
        s.second.addExtraGlobalSynapseParams(synapseKernelParameters);
        s.second.addExtraGlobalPostLearnParams(simLearnPostKernelParameters);
        s.second.addExtraGlobalSynapseDynamicsParams(synapseDynamicsKernelParameters);
    }

    setPopulationSums();

#ifndef CPU_ONLY
    // figure out where to reset the spike counters
    if (m_SynapseGroups.empty()) { // no synapses -> reset in neuron kernel
        resetKernel= GENN_FLAGS::calcNeurons;
    }
    else { // there are synapses
        if (!m_SynapsePostLearnGroups.empty()) {
            resetKernel= GENN_FLAGS::learnSynapsesPost;
        }
        else {
            resetKernel= GENN_FLAGS::calcSynapses;
        }
    }
#endif
}



#endif // MODELSPEC_CC
