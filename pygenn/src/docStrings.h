/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_CurrentSource = R"doc()doc";

static const char *__doc_CurrentSource_2 = R"doc()doc";

static const char *__doc_CurrentSourceEGPAdapter = R"doc()doc";

static const char *__doc_CurrentSourceEGPAdapter_CurrentSourceEGPAdapter = R"doc()doc";

static const char *__doc_CurrentSourceEGPAdapter_getDefs = R"doc()doc";

static const char *__doc_CurrentSourceEGPAdapter_getLoc = R"doc()doc";

static const char *__doc_CurrentSourceEGPAdapter_m_CS = R"doc()doc";

static const char *__doc_CurrentSourceInternal = R"doc()doc";

static const char *__doc_CurrentSourceInternal_2 = R"doc()doc";

static const char *__doc_CurrentSourceInternal_3 = R"doc()doc";

static const char *__doc_CurrentSourceInternal_CurrentSourceInternal = R"doc()doc";

static const char *__doc_CurrentSourceModels_Base = R"doc(Base class for all current source models)doc";

static const char *__doc_CurrentSourceModels_Base_getHashDigest = R"doc(Update hash from model)doc";

static const char *__doc_CurrentSourceModels_Base_getInjectionCode = R"doc(Gets the code that defines current injected each timestep)doc";

static const char *__doc_CurrentSourceModels_Base_getNeuronVarRefs = R"doc(Gets names and types of model variable references)doc";

static const char *__doc_CurrentSourceModels_Base_getVar = R"doc(Find the named variable)doc";

static const char *__doc_CurrentSourceModels_Base_getVars = R"doc(Gets model variables)doc";

static const char *__doc_CurrentSourceModels_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_CurrentSourceModels_DC =
R"doc(DC source
It has a single parameter:
- ``amp``    - amplitude of the current [nA])doc";

static const char *__doc_CurrentSourceModels_DC_getInjectionCode = R"doc()doc";

static const char *__doc_CurrentSourceModels_DC_getInstance = R"doc()doc";

static const char *__doc_CurrentSourceModels_DC_getParams = R"doc()doc";

static const char *__doc_CurrentSourceModels_GaussianNoise =
R"doc(Noisy current source with noise drawn from normal distribution
It has 2 parameters:
- ``mean``   - mean of the normal distribution [nA]
- ``sd``     - standard deviation of the normal distribution [nA])doc";

static const char *__doc_CurrentSourceModels_GaussianNoise_getInjectionCode = R"doc()doc";

static const char *__doc_CurrentSourceModels_GaussianNoise_getInstance = R"doc()doc";

static const char *__doc_CurrentSourceModels_GaussianNoise_getParams = R"doc()doc";

static const char *__doc_CurrentSourceModels_PoissonExp =
R"doc(Current source for injecting a current equivalent to a population of
Poisson spike sources, one-to-one connected with exponential synapses
It has 3 parameters:
- ``weight`` - synaptic weight of the Poisson spikes [nA]
- ``tauSyn`` - decay time constant [ms]
- ``rate``   - mean firing rate [Hz])doc";

static const char *__doc_CurrentSourceModels_PoissonExp_getDerivedParams = R"doc()doc";

static const char *__doc_CurrentSourceModels_PoissonExp_getInjectionCode = R"doc()doc";

static const char *__doc_CurrentSourceModels_PoissonExp_getInstance = R"doc()doc";

static const char *__doc_CurrentSourceModels_PoissonExp_getParams = R"doc()doc";

static const char *__doc_CurrentSourceModels_PoissonExp_getVars = R"doc()doc";

static const char *__doc_CurrentSourceNeuronVarRefAdapter = R"doc()doc";

static const char *__doc_CurrentSourceNeuronVarRefAdapter_CurrentSourceNeuronVarRefAdapter = R"doc()doc";

static const char *__doc_CurrentSourceNeuronVarRefAdapter_getDefs = R"doc()doc";

static const char *__doc_CurrentSourceNeuronVarRefAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CurrentSourceNeuronVarRefAdapter_m_CS = R"doc()doc";

static const char *__doc_CurrentSourceVarAdapter = R"doc()doc";

static const char *__doc_CurrentSourceVarAdapter_CurrentSourceVarAdapter = R"doc()doc";

static const char *__doc_CurrentSourceVarAdapter_getDefs = R"doc()doc";

static const char *__doc_CurrentSourceVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CurrentSourceVarAdapter_getLoc = R"doc()doc";

static const char *__doc_CurrentSourceVarAdapter_getTarget = R"doc()doc";

static const char *__doc_CurrentSourceVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_CurrentSourceVarAdapter_isVarDelayed = R"doc()doc";

static const char *__doc_CurrentSourceVarAdapter_m_CS = R"doc()doc";

static const char *__doc_CurrentSource_CurrentSource = R"doc()doc";

static const char *__doc_CurrentSource_CurrentSource_2 = R"doc()doc";

static const char *__doc_CurrentSource_CurrentSource_3 = R"doc()doc";

static const char *__doc_CurrentSource_finalise = R"doc()doc";

static const char *__doc_CurrentSource_getDerivedParams = R"doc()doc";

static const char *__doc_CurrentSource_getExtraGlobalParamLocation = R"doc(Get location of neuron model extra global parameter by name)doc";

static const char *__doc_CurrentSource_getHashDigest =
R"doc(Updates hash with current source
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CurrentSource_getInitHashDigest =
R"doc(Updates hash with current source initialisation
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CurrentSource_getInjectionCodeTokens = R"doc()doc";

static const char *__doc_CurrentSource_getModel = R"doc(Gets the current source model used by this group)doc";

static const char *__doc_CurrentSource_getName = R"doc()doc";

static const char *__doc_CurrentSource_getNeuronVarReferences = R"doc()doc";

static const char *__doc_CurrentSource_getParams = R"doc()doc";

static const char *__doc_CurrentSource_getTargetVar =
R"doc(Get name of neuron input variable current source model will inject into
This will either be 'Isyn' or the name of one of the target neuron's additional input variables.)doc";

static const char *__doc_CurrentSource_getTrgNeuronGroup = R"doc()doc";

static const char *__doc_CurrentSource_getVarInitialisers = R"doc()doc";

static const char *__doc_CurrentSource_getVarLocation = R"doc(Get variable location for current source model state variable)doc";

static const char *__doc_CurrentSource_getVarLocationHashDigest = R"doc()doc";

static const char *__doc_CurrentSource_isParamDynamic = R"doc(Is parameter dynamic i.e. it can be changed at runtime)doc";

static const char *__doc_CurrentSource_isVarInitRequired = R"doc(Is var init code required for any variables in this current source?)doc";

static const char *__doc_CurrentSource_isZeroCopyEnabled = R"doc()doc";

static const char *__doc_CurrentSource_m_DerivedParams = R"doc()doc";

static const char *__doc_CurrentSource_m_DynamicParams = R"doc(Data structure tracking whether parameters are dynamic or not)doc";

static const char *__doc_CurrentSource_m_ExtraGlobalParamLocation = R"doc(Location of extra global parameters)doc";

static const char *__doc_CurrentSource_m_InjectionCodeTokens = R"doc(Tokens produced by scanner from injection code)doc";

static const char *__doc_CurrentSource_m_Model = R"doc()doc";

static const char *__doc_CurrentSource_m_Name = R"doc()doc";

static const char *__doc_CurrentSource_m_NeuronVarReferences = R"doc()doc";

static const char *__doc_CurrentSource_m_Params = R"doc()doc";

static const char *__doc_CurrentSource_m_TargetVar =
R"doc(Name of neuron input variable current source will inject into
This should either be 'Isyn' or the name of one of the target neuron's additional input variables.)doc";

static const char *__doc_CurrentSource_m_TrgNeuronGroup = R"doc()doc";

static const char *__doc_CurrentSource_m_VarInitialisers = R"doc()doc";

static const char *__doc_CurrentSource_m_VarLocation = R"doc(Location of individual state variables)doc";

static const char *__doc_CurrentSource_setExtraGlobalParamLocation =
R"doc(Set location of extra global parameter
This is ignored for simulations on hardware with a single memory space.)doc";

static const char *__doc_CurrentSource_setParamDynamic = R"doc(Set whether parameter is dynamic or not i.e. it can be changed at runtime)doc";

static const char *__doc_CurrentSource_setTargetVar =
R"doc(Set name of neuron input variable current source model will inject into
This should either be 'Isyn' or the name of one of the target neuron's additional input variables.)doc";

static const char *__doc_CurrentSource_setVarLocation = R"doc(Set location of current source state variable)doc";

static const char *__doc_CustomConnectivityUpdate = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_2 = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateEGPAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateEGPAdapter_CustomConnectivityUpdateEGPAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateEGPAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateEGPAdapter_getLoc = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateEGPAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateInternal = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateInternal_2 = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateInternal_3 = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateInternal_CustomConnectivityUpdateInternal = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateModels_Base = R"doc(Base class for all current source models)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getHashDigest = R"doc(Update hash from model)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getHostUpdateCode = R"doc(Gets the code that performs host update)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getPostVar = R"doc(Find the named postsynaptic variable)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getPostVarRefs = R"doc(Gets names and types (as strings) of postsynaptic variable references)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getPostVars =
R"doc(Gets names and types (as strings) of state variables that are common
across all synapses going to the same postsynaptic neuron)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getPreVar = R"doc(Find the named presynaptic variable)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getPreVarRefs = R"doc(Gets names and types (as strings) of presynaptic variable references)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getPreVars =
R"doc(Gets names and types (as strings) of state variables that are common
across all synapses coming from the same presynaptic neuron)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getRowUpdateCode = R"doc(Gets the code that performs a row-wise update)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getVar = R"doc(Find the named variable)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getVarRefs = R"doc(Gets names and types (as strings) of synapse variable references)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_getVars = R"doc(Gets model variables)doc";

static const char *__doc_CustomConnectivityUpdateModels_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_CustomConnectivityUpdatePostVarAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarAdapter_CustomConnectivityUpdatePostVarAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarAdapter_getLoc = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarAdapter_getTarget = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarAdapter_isVarDelayed = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarRefAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarRefAdapter_CustomConnectivityUpdatePostVarRefAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarRefAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarRefAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePostVarRefAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarAdapter_CustomConnectivityUpdatePreVarAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarAdapter_getLoc = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarAdapter_getTarget = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarAdapter_isVarDelayed = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarRefAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarRefAdapter_CustomConnectivityUpdatePreVarRefAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarRefAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarRefAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdatePreVarRefAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarAdapter_CustomConnectivityUpdateVarAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarAdapter_getLoc = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarAdapter_getTarget = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarRefAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarRefAdapter_CustomConnectivityUpdateVarRefAdapter = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarRefAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarRefAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdateVarRefAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_CustomConnectivityUpdate = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_CustomConnectivityUpdate_2 = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_CustomConnectivityUpdate_3 = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_finalise = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getDependentVariables =
R"doc(Get vector of group names and variables in synapse groups, custom updates and other
custom connectivity updates which are attached to the same sparse connectivity this
custom connectivty update will update and thus will need modifying when we add and remove synapses)doc";

static const char *__doc_CustomConnectivityUpdate_getDerivedParams = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getExtraGlobalParamLocation = R"doc(Get location of neuron model extra global parameter by name)doc";

static const char *__doc_CustomConnectivityUpdate_getHashDigest =
R"doc(Updates hash with custom update
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CustomConnectivityUpdate_getHostUpdateCodeTokens = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getInitHashDigest =
R"doc(Updates hash with custom update
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CustomConnectivityUpdate_getModel = R"doc(Gets the custom connectivity update model used by this group)doc";

static const char *__doc_CustomConnectivityUpdate_getName = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getParams = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getPostDelayNeuronGroup = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getPostVarInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getPostVarLocation = R"doc(Get variable location for postsynaptic state variable)doc";

static const char *__doc_CustomConnectivityUpdate_getPostVarReferences = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getPreDelayNeuronGroup = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getPreVarInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getPreVarLocation = R"doc(Get variable location for presynaptic state variable)doc";

static const char *__doc_CustomConnectivityUpdate_getPreVarReferences = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getRowUpdateCodeTokens = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getSynapseGroup = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getUpdateGroupName = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getVarInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getVarLocation = R"doc(Get variable location for synaptic state variable)doc";

static const char *__doc_CustomConnectivityUpdate_getVarLocationHashDigest = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getVarRefDelayGroup = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_getVarReferences = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_isParamDynamic = R"doc(Is parameter dynamic i.e. it can be changed at runtime)doc";

static const char *__doc_CustomConnectivityUpdate_isPostVarInitRequired = R"doc(Is var init code required for any postsynaptic variables in this custom connectivity update group?)doc";

static const char *__doc_CustomConnectivityUpdate_isPreVarInitRequired = R"doc(Is var init code required for any presynaptic variables in this custom connectivity update group?)doc";

static const char *__doc_CustomConnectivityUpdate_isVarInitRequired = R"doc(Is var init code required for any synaptic variables in this custom connectivity update group?)doc";

static const char *__doc_CustomConnectivityUpdate_isZeroCopyEnabled = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_DerivedParams = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_DynamicParams = R"doc(Data structure tracking whether parameters are dynamic or not)doc";

static const char *__doc_CustomConnectivityUpdate_m_ExtraGlobalParamLocation = R"doc(Location of extra global parameters)doc";

static const char *__doc_CustomConnectivityUpdate_m_HostUpdateCodeTokens = R"doc(Tokens produced by scanner from host update code)doc";

static const char *__doc_CustomConnectivityUpdate_m_Model = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_Name = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_Params = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_PostDelayNeuronGroup = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_PostVarInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_PostVarLocation = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_PostVarReferences = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_PreDelayNeuronGroup = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_PreVarInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_PreVarLocation = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_PreVarReferences = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_RowUpdateCodeTokens = R"doc(Tokens produced by scanner from row update code)doc";

static const char *__doc_CustomConnectivityUpdate_m_SynapseGroup = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_UpdateGroupName = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_VarInitialisers = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_m_VarLocation = R"doc(Location of individual state variables)doc";

static const char *__doc_CustomConnectivityUpdate_m_VarReferences = R"doc()doc";

static const char *__doc_CustomConnectivityUpdate_setExtraGlobalParamLocation =
R"doc(Set location of extra global parameter
This is ignored for simulations on hardware with a single memory space.)doc";

static const char *__doc_CustomConnectivityUpdate_setParamDynamic = R"doc(Set whether parameter is dynamic or not i.e. it can be changed at runtime)doc";

static const char *__doc_CustomConnectivityUpdate_setPostVarLocation =
R"doc(Set location of postsynaptic state variable
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_CustomConnectivityUpdate_setPreVarLocation =
R"doc(Set location of presynaptic state variable
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_CustomConnectivityUpdate_setVarLocation =
R"doc(Set location of synaptic state variable
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_CustomUpdate = R"doc()doc";

static const char *__doc_CustomUpdate_2 = R"doc()doc";

static const char *__doc_CustomUpdateBase = R"doc()doc";

static const char *__doc_CustomUpdateBase_CustomUpdateBase = R"doc()doc";

static const char *__doc_CustomUpdateBase_CustomUpdateBase_2 = R"doc()doc";

static const char *__doc_CustomUpdateBase_CustomUpdateBase_3 = R"doc()doc";

static const char *__doc_CustomUpdateBase_checkVarReferenceDims = R"doc(Helper function to check if variable reference types match those specified in model)doc";

static const char *__doc_CustomUpdateBase_finalise = R"doc()doc";

static const char *__doc_CustomUpdateBase_getDerivedParams = R"doc()doc";

static const char *__doc_CustomUpdateBase_getDims = R"doc(Get dimensions of this custom update)doc";

static const char *__doc_CustomUpdateBase_getEGPReferences = R"doc()doc";

static const char *__doc_CustomUpdateBase_getExtraGlobalParamLocation = R"doc(Get location of neuron model extra global parameter by name)doc";

static const char *__doc_CustomUpdateBase_getModel = R"doc(Gets the custom update model used by this group)doc";

static const char *__doc_CustomUpdateBase_getName = R"doc()doc";

static const char *__doc_CustomUpdateBase_getParams = R"doc()doc";

static const char *__doc_CustomUpdateBase_getReferencedCustomUpdates = R"doc()doc";

static const char *__doc_CustomUpdateBase_getUpdateCodeTokens = R"doc()doc";

static const char *__doc_CustomUpdateBase_getUpdateGroupName = R"doc()doc";

static const char *__doc_CustomUpdateBase_getVarInitialisers = R"doc()doc";

static const char *__doc_CustomUpdateBase_getVarLocation = R"doc(Get variable location for custom update model state variable)doc";

static const char *__doc_CustomUpdateBase_getVarLocationHashDigest = R"doc()doc";

static const char *__doc_CustomUpdateBase_isInitRNGRequired = R"doc(Does this current source group require an RNG for it's init code)doc";

static const char *__doc_CustomUpdateBase_isModelReduction = R"doc()doc";

static const char *__doc_CustomUpdateBase_isParamDynamic = R"doc(Is parameter dynamic i.e. it can be changed at runtime)doc";

static const char *__doc_CustomUpdateBase_isReduction = R"doc()doc";

static const char *__doc_CustomUpdateBase_isVarInitRequired = R"doc(Is var init code required for any variables in this custom update group's custom update model?)doc";

static const char *__doc_CustomUpdateBase_isZeroCopyEnabled = R"doc()doc";

static const char *__doc_CustomUpdateBase_m_DerivedParams = R"doc()doc";

static const char *__doc_CustomUpdateBase_m_Dims = R"doc(Dimensions of this custom update)doc";

static const char *__doc_CustomUpdateBase_m_DynamicParams = R"doc(Data structure tracking whether parameters are dynamic or not)doc";

static const char *__doc_CustomUpdateBase_m_EGPReferences = R"doc()doc";

static const char *__doc_CustomUpdateBase_m_ExtraGlobalParamLocation = R"doc(Location of extra global parameters)doc";

static const char *__doc_CustomUpdateBase_m_Model = R"doc()doc";

static const char *__doc_CustomUpdateBase_m_Name = R"doc()doc";

static const char *__doc_CustomUpdateBase_m_Params = R"doc()doc";

static const char *__doc_CustomUpdateBase_m_UpdateCodeTokens = R"doc(Tokens produced by scanner from update code)doc";

static const char *__doc_CustomUpdateBase_m_UpdateGroupName = R"doc()doc";

static const char *__doc_CustomUpdateBase_m_VarInitialisers = R"doc()doc";

static const char *__doc_CustomUpdateBase_m_VarLocation = R"doc(Location of individual state variables)doc";

static const char *__doc_CustomUpdateBase_setExtraGlobalParamLocation =
R"doc(Set location of extra global parameter
This is ignored for simulations on hardware with a single memory space.)doc";

static const char *__doc_CustomUpdateBase_setParamDynamic = R"doc(Set whether parameter is dynamic or not i.e. it can be changed at runtime)doc";

static const char *__doc_CustomUpdateBase_setVarLocation =
R"doc(Set location of state variable
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_CustomUpdateBase_updateHash =
R"doc(Updates hash with custom update
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CustomUpdateBase_updateInitHash =
R"doc(Updates hash with custom update
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CustomUpdateEGPAdapter = R"doc()doc";

static const char *__doc_CustomUpdateEGPAdapter_CustomUpdateEGPAdapter = R"doc()doc";

static const char *__doc_CustomUpdateEGPAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomUpdateEGPAdapter_getLoc = R"doc()doc";

static const char *__doc_CustomUpdateEGPAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomUpdateInternal = R"doc()doc";

static const char *__doc_CustomUpdateInternal_2 = R"doc()doc";

static const char *__doc_CustomUpdateInternal_CustomUpdateInternal = R"doc()doc";

static const char *__doc_CustomUpdateModels_Base = R"doc(Base class for all current source models)doc";

static const char *__doc_CustomUpdateModels_Base_getExtraGlobalParamRefs = R"doc(Gets names and types of model extra global parameter references)doc";

static const char *__doc_CustomUpdateModels_Base_getHashDigest = R"doc(Update hash from model)doc";

static const char *__doc_CustomUpdateModels_Base_getUpdateCode = R"doc(Gets the code that performs the custom update)doc";

static const char *__doc_CustomUpdateModels_Base_getVar = R"doc(Find the named variable)doc";

static const char *__doc_CustomUpdateModels_Base_getVarRefs = R"doc(Gets names and typesn of model variable references)doc";

static const char *__doc_CustomUpdateModels_Base_getVars = R"doc(Gets model variables)doc";

static const char *__doc_CustomUpdateModels_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_CustomUpdateModels_Base_validate_2 = R"doc()doc";

static const char *__doc_CustomUpdateModels_Transpose = R"doc(Minimal custom update model for calculating tranpose)doc";

static const char *__doc_CustomUpdateModels_Transpose_getInstance = R"doc()doc";

static const char *__doc_CustomUpdateModels_Transpose_getVarRefs = R"doc()doc";

static const char *__doc_CustomUpdateVarAccess =
R"doc(Supported combinations of access mode and dimension for custom update variables
The axes are defined 'subtractively' ie VarAccessDim::BATCH indicates that this axis should be removed)doc";

static const char *__doc_CustomUpdateVarAccess_READ_ONLY = R"doc()doc";

static const char *__doc_CustomUpdateVarAccess_READ_ONLY_SHARED = R"doc()doc";

static const char *__doc_CustomUpdateVarAccess_READ_ONLY_SHARED_NEURON = R"doc()doc";

static const char *__doc_CustomUpdateVarAccess_READ_WRITE = R"doc()doc";

static const char *__doc_CustomUpdateVarAccess_REDUCE_BATCH_MAX = R"doc()doc";

static const char *__doc_CustomUpdateVarAccess_REDUCE_BATCH_SUM = R"doc()doc";

static const char *__doc_CustomUpdateVarAccess_REDUCE_NEURON_MAX = R"doc()doc";

static const char *__doc_CustomUpdateVarAccess_REDUCE_NEURON_SUM = R"doc()doc";

static const char *__doc_CustomUpdateVarAdapter = R"doc()doc";

static const char *__doc_CustomUpdateVarAdapter_CustomUpdateVarAdapter = R"doc()doc";

static const char *__doc_CustomUpdateVarAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomUpdateVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CustomUpdateVarAdapter_getLoc = R"doc()doc";

static const char *__doc_CustomUpdateVarAdapter_getTarget = R"doc()doc";

static const char *__doc_CustomUpdateVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_CustomUpdateVarAdapter_isVarDelayed = R"doc()doc";

static const char *__doc_CustomUpdateVarAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomUpdateVarRefAdapter = R"doc()doc";

static const char *__doc_CustomUpdateVarRefAdapter_CustomUpdateVarRefAdapter = R"doc()doc";

static const char *__doc_CustomUpdateVarRefAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomUpdateVarRefAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CustomUpdateVarRefAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomUpdateWU = R"doc()doc";

static const char *__doc_CustomUpdateWU_2 = R"doc()doc";

static const char *__doc_CustomUpdateWUInternal = R"doc()doc";

static const char *__doc_CustomUpdateWUInternal_2 = R"doc()doc";

static const char *__doc_CustomUpdateWUInternal_3 = R"doc()doc";

static const char *__doc_CustomUpdateWUInternal_CustomUpdateWUInternal = R"doc()doc";

static const char *__doc_CustomUpdateWUVarRefAdapter = R"doc()doc";

static const char *__doc_CustomUpdateWUVarRefAdapter_CustomUpdateWUVarRefAdapter = R"doc()doc";

static const char *__doc_CustomUpdateWUVarRefAdapter_getDefs = R"doc()doc";

static const char *__doc_CustomUpdateWUVarRefAdapter_getInitialisers = R"doc()doc";

static const char *__doc_CustomUpdateWUVarRefAdapter_m_CU = R"doc()doc";

static const char *__doc_CustomUpdateWU_CustomUpdateWU = R"doc()doc";

static const char *__doc_CustomUpdateWU_finalise = R"doc()doc";

static const char *__doc_CustomUpdateWU_getHashDigest =
R"doc(Updates hash with custom update
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CustomUpdateWU_getInitHashDigest =
R"doc(Updates hash with custom update
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CustomUpdateWU_getKernelSize = R"doc()doc";

static const char *__doc_CustomUpdateWU_getReferencedCustomUpdates = R"doc(Get vector of other custom updates referenced by this custom update)doc";

static const char *__doc_CustomUpdateWU_getSynapseGroup = R"doc()doc";

static const char *__doc_CustomUpdateWU_getVarReferences = R"doc()doc";

static const char *__doc_CustomUpdateWU_isBatchReduction = R"doc()doc";

static const char *__doc_CustomUpdateWU_isTransposeOperation = R"doc()doc";

static const char *__doc_CustomUpdateWU_m_SynapseGroup = R"doc()doc";

static const char *__doc_CustomUpdateWU_m_VarReferences = R"doc()doc";

static const char *__doc_CustomUpdate_CustomUpdate = R"doc()doc";

static const char *__doc_CustomUpdate_finalise = R"doc()doc";

static const char *__doc_CustomUpdate_getDelayNeuronGroup = R"doc()doc";

static const char *__doc_CustomUpdate_getHashDigest =
R"doc(Updates hash with custom update
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CustomUpdate_getInitHashDigest =
R"doc(Updates hash with custom update
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_CustomUpdate_getNumNeurons = R"doc()doc";

static const char *__doc_CustomUpdate_getReferencedCustomUpdates = R"doc(Get vector of other custom updates referenced by this custom update)doc";

static const char *__doc_CustomUpdate_getVarReferences = R"doc()doc";

static const char *__doc_CustomUpdate_isBatchReduction = R"doc()doc";

static const char *__doc_CustomUpdate_isNeuronReduction = R"doc()doc";

static const char *__doc_CustomUpdate_m_DelayNeuronGroup = R"doc()doc";

static const char *__doc_CustomUpdate_m_NumNeurons = R"doc()doc";

static const char *__doc_CustomUpdate_m_VarReferences = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Base = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Base_getCalcKernelSizeFunc = R"doc(Get function to calculate kernel size required for this conenctor based on its parameters)doc";

static const char *__doc_InitSparseConnectivitySnippet_Base_getCalcMaxColLengthFunc = R"doc(Get function to calculate the maximum column length of this connector based on the parameters and the size of the pre and postsynaptic population)doc";

static const char *__doc_InitSparseConnectivitySnippet_Base_getCalcMaxRowLengthFunc = R"doc(Get function to calculate the maximum row length of this connector based on the parameters and the size of the pre and postsynaptic population)doc";

static const char *__doc_InitSparseConnectivitySnippet_Base_getColBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Base_getHashDigest = R"doc(Update hash from snippet)doc";

static const char *__doc_InitSparseConnectivitySnippet_Base_getHostInitCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Base_getRowBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_InitSparseConnectivitySnippet_Conv2D =
R"doc(Initialises convolutional connectivity
Row build state variables are used to convert presynaptic neuron index to rows, columns and channels and,
from these, to calculate the range of postsynaptic rows, columns and channels connections will be made within.
This sparse connectivity snippet does not support multiple threads per neuron)doc";

static const char *__doc_InitSparseConnectivitySnippet_Conv2D_getCalcKernelSizeFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Conv2D_getCalcMaxRowLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Conv2D_getInstance = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Conv2D_getParams = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Conv2D_getRowBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPostWithReplacement =
R"doc(Initialises connectivity with a fixed number of random synapses per row.
The postsynaptic targets of the synapses can be initialised in parallel by sampling from the discrete
uniform distribution. However, to sample connections in ascending order, we sample from the 1st order statistic
of the uniform distribution -- Beta[1, Npost] -- essentially the next smallest value. In this special case
this is equivalent to the exponential distribution which can be sampled in constant time using the inversion method.)doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPostWithReplacement_getCalcMaxColLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPostWithReplacement_getCalcMaxRowLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPostWithReplacement_getInstance = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPostWithReplacement_getParams = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPostWithReplacement_getRowBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPreWithReplacement =
R"doc(Initialises connectivity with a fixed number of random synapses per column.
No need for ordering here so fine to sample directly from uniform distribution)doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPreWithReplacement_getCalcMaxColLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPreWithReplacement_getCalcMaxRowLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPreWithReplacement_getColBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPreWithReplacement_getInstance = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberPreWithReplacement_getParams = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberTotalWithReplacement =
R"doc(Initialises connectivity with a total number of random synapses.
The first stage in using this connectivity is to determine how many of the total synapses end up in each row.
This can be determined by sampling from the multinomial distribution. However, this operation cannot be
efficiently parallelised so must be performed on the host and the result passed as an extra global parameter array.
Once the length of each row is determined, the postsynaptic targets of the synapses can be initialised in parallel
by sampling from the discrete uniform distribution. However, to sample connections in ascending order, we sample
from the 1st order statistic of the uniform distribution -- Beta[1, Npost] -- essentially the next smallest value.
In this special case this is equivalent to the exponential distribution which can be sampled in constant time using the inversion method.)doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberTotalWithReplacement_getCalcMaxColLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberTotalWithReplacement_getCalcMaxRowLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberTotalWithReplacement_getExtraGlobalParams = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberTotalWithReplacement_getHostInitCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberTotalWithReplacement_getInstance = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberTotalWithReplacement_getParams = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedNumberTotalWithReplacement_getRowBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbability =
R"doc(Initialises connectivity with a fixed probability of a synapse existing
between a pair of pre and postsynaptic neurons.
Whether a synapse exists between a pair of pre and a postsynaptic
neurons can be modelled using a Bernoulli distribution. While this COULD
be sampled directly by repeatedly drawing from the uniform distribution,
this is inefficient. Instead we sample from the geometric distribution
which describes "the probability distribution of the number of Bernoulli
trials needed to get one success" -- essentially the distribution of the
'gaps' between synapses. We do this using the "inversion method"
described by Devroye (1986) -- essentially inverting the CDF of the
equivalent continuous distribution (in this case the exponential distribution))doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbabilityBase =
R"doc(Base class for snippets which initialise connectivity with a fixed probability
of a synapse existing between a pair of pre and postsynaptic neurons.)doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbabilityBase_getCalcMaxColLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbabilityBase_getCalcMaxRowLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbabilityBase_getDerivedParams = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbabilityBase_getParams = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbabilityBase_getRowBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbabilityNoAutapse =
R"doc(Initialises connectivity with a fixed probability of a synapse existing
between a pair of pre and postsynaptic neurons. This version ensures there
are no autapses - connections between neurons with the same id
so should be used for recurrent connections.
Whether a synapse exists between a pair of pre and a postsynaptic
neurons can be modelled using a Bernoulli distribution. While this COULD
br sampling directly by repeatedly drawing from the uniform distribution,
this is innefficient. Instead we sample from the gemetric distribution
which describes "the probability distribution of the number of Bernoulli
trials needed to get one success" -- essentially the distribution of the
'gaps' between synapses. We do this using the "inversion method"
described by Devroye (1986) -- essentially inverting the CDF of the
equivalent continuous distribution (in this case the exponential distribution))doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbabilityNoAutapse_getInstance = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbabilityNoAutapse_getRowBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbability_getInstance = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_FixedProbability_getRowBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init_Init = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init_getColBuildCodeTokens = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init_getHostInitCodeTokens = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init_getRowBuildCodeTokens = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init_isHostRNGRequired = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init_isRNGRequired = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init_m_ColBuildCodeTokens = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init_m_HostInitCodeTokens = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Init_m_RowBuildCodeTokens = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_OneToOne = R"doc(Initialises connectivity to a 'one-to-one' diagonal matrix)doc";

static const char *__doc_InitSparseConnectivitySnippet_OneToOne_getCalcMaxColLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_OneToOne_getCalcMaxRowLengthFunc = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_OneToOne_getInstance = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_OneToOne_getRowBuildCode = R"doc()doc";

static const char *__doc_InitSparseConnectivitySnippet_Uninitialised = R"doc(Used to mark connectivity as uninitialised - no initialisation code will be run)doc";

static const char *__doc_InitSparseConnectivitySnippet_Uninitialised_getInstance = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_AvgPoolConv2D =
R"doc(Initialises convolutional connectivity preceded by averaging pooling
Row build state variables are used to convert presynaptic neuron index to rows, columns and channels and,
from these, to calculate the range of postsynaptic rows, columns and channels connections will be made within.)doc";

static const char *__doc_InitToeplitzConnectivitySnippet_AvgPoolConv2D_getCalcKernelSizeFunc = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_AvgPoolConv2D_getCalcMaxRowLengthFunc = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_AvgPoolConv2D_getDerivedParams = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_AvgPoolConv2D_getDiagonalBuildCode = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_AvgPoolConv2D_getInstance = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_AvgPoolConv2D_getParams = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Base = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Base_getCalcKernelSizeFunc = R"doc(Get function to calculate kernel size required for this conenctor based on its parameters)doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Base_getCalcMaxRowLengthFunc = R"doc(Get function to calculate the maximum row length of this connector based on the parameters and the size of the pre and postsynaptic population)doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Base_getDiagonalBuildCode = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Base_getHashDigest = R"doc(Update hash from snippet)doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Conv2D =
R"doc(Initialises convolutional connectivity
Row build state variables are used to convert presynaptic neuron index to rows, columns and channels and,
from these, to calculate the range of postsynaptic rows, columns and channels connections will be made within.)doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Conv2D_getCalcKernelSizeFunc = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Conv2D_getCalcMaxRowLengthFunc = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Conv2D_getDerivedParams = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Conv2D_getDiagonalBuildCode = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Conv2D_getInstance = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Conv2D_getParams = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Init = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Init_Init = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Init_getDiagonalBuildCodeTokens = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Init_isRNGRequired = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Init_m_DiagonalBuildCodeTokens = R"doc()doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Uninitialised = R"doc(Used to mark connectivity as uninitialised - no initialisation code will be run)doc";

static const char *__doc_InitToeplitzConnectivitySnippet_Uninitialised_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_Base = R"doc()doc";

static const char *__doc_InitVarSnippet_Base_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_Base_getHashDigest = R"doc(Update hash from snippet)doc";

static const char *__doc_InitVarSnippet_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_InitVarSnippet_Binomial =
R"doc(Initialises variable by sampling from the binomial distribution
This snippet takes 2 parameters:

- ``n`` - number of trials
- ``p`` - success probability for each trial)doc";

static const char *__doc_InitVarSnippet_Binomial_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_Binomial_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_Binomial_getParams = R"doc()doc";

static const char *__doc_InitVarSnippet_Constant =
R"doc(Initialises variable to a constant value
This snippet takes 1 parameter:

- ``value`` - The value to intialise the variable to

\note This snippet type is seldom used directly - InitVarSnippet::Init
has an implicit constructor that, internally, creates one of these snippets)doc";

static const char *__doc_InitVarSnippet_Constant_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_Constant_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_Constant_getParams = R"doc()doc";

static const char *__doc_InitVarSnippet_Exponential =
R"doc(Initialises variable by sampling from the exponential distribution
This snippet takes 1 parameter:

- ``lambda`` - mean event rate (events per unit time/distance))doc";

static const char *__doc_InitVarSnippet_Exponential_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_Exponential_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_Exponential_getParams = R"doc()doc";

static const char *__doc_InitVarSnippet_Gamma =
R"doc(Initialises variable by sampling from the gamma distribution
This snippet takes 2 parameters:

- ``a`` - distribution shape
- ``b`` - distribution scale)doc";

static const char *__doc_InitVarSnippet_Gamma_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_Gamma_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_Gamma_getParams = R"doc()doc";

static const char *__doc_InitVarSnippet_Init =
R"doc(Class used to bind together everything required to initialise a variable:
1. A pointer to a variable initialisation snippet
2. The parameters required to control the variable initialisation snippet)doc";

static const char *__doc_InitVarSnippet_Init_Init = R"doc()doc";

static const char *__doc_InitVarSnippet_Init_Init_2 = R"doc()doc";

static const char *__doc_InitVarSnippet_Init_getCodeTokens = R"doc()doc";

static const char *__doc_InitVarSnippet_Init_isKernelRequired = R"doc()doc";

static const char *__doc_InitVarSnippet_Init_isRNGRequired = R"doc()doc";

static const char *__doc_InitVarSnippet_Init_m_CodeTokens = R"doc()doc";

static const char *__doc_InitVarSnippet_Kernel = R"doc(Used to initialise synapse variables from a kernel)doc";

static const char *__doc_InitVarSnippet_Kernel_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_Kernel_getExtraGlobalParams = R"doc()doc";

static const char *__doc_InitVarSnippet_Kernel_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_Normal =
R"doc(Initialises variable by sampling from the normal distribution
This snippet takes 2 parameters:

- ``mean`` - The mean
- ``sd`` - The standard deviation)doc";

static const char *__doc_InitVarSnippet_NormalClipped =
R"doc(Initialises variable by sampling from the normal distribution,
Resamples value if out of range specified my min and max
This snippet takes 2 parameters:

- ``mean`` - The mean
- ``sd`` - ThGeNN::e standard deviation
- ``min`` - The minimum value
- ``max`` - The maximum value)doc";

static const char *__doc_InitVarSnippet_NormalClippedDelay =
R"doc(Initialises variable by sampling from the normal distribution,
Resamples value of out of range specified my min and max.
This snippet is intended for initializing (dendritic) delay parameters
where parameters are specified in ms but converted to timesteps.
This snippet takes 2 parameters:

- ``mean`` - The mean [ms]
- ``sd`` - The standard deviation [ms]
- ``min`` - The minimum value [ms]
- ``max`` - The maximum value [ms])doc";

static const char *__doc_InitVarSnippet_NormalClippedDelay_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_NormalClippedDelay_getDerivedParams = R"doc()doc";

static const char *__doc_InitVarSnippet_NormalClippedDelay_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_NormalClippedDelay_getParams = R"doc()doc";

static const char *__doc_InitVarSnippet_NormalClipped_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_NormalClipped_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_NormalClipped_getParams = R"doc()doc";

static const char *__doc_InitVarSnippet_Normal_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_Normal_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_Normal_getParams = R"doc()doc";

static const char *__doc_InitVarSnippet_Uniform =
R"doc(Initialises variable by sampling from the uniform distribution
This snippet takes 2 parameters:

- ``min`` - The minimum value
- ``max`` - The maximum value)doc";

static const char *__doc_InitVarSnippet_Uniform_getCode = R"doc()doc";

static const char *__doc_InitVarSnippet_Uniform_getInstance = R"doc()doc";

static const char *__doc_InitVarSnippet_Uniform_getParams = R"doc()doc";

static const char *__doc_InitVarSnippet_Uninitialised = R"doc(Used to mark variables as uninitialised - no initialisation code will be run)doc";

static const char *__doc_InitVarSnippet_Uninitialised_getInstance = R"doc()doc";

static const char *__doc_LocationContainer = R"doc()doc";

static const char *__doc_LocationContainer_LocationContainer = R"doc()doc";

static const char *__doc_LocationContainer_anyZeroCopy = R"doc()doc";

static const char *__doc_LocationContainer_get = R"doc()doc";

static const char *__doc_LocationContainer_m_DefaultLocation = R"doc()doc";

static const char *__doc_LocationContainer_m_Locations = R"doc()doc";

static const char *__doc_LocationContainer_set = R"doc()doc";

static const char *__doc_LocationContainer_updateHash = R"doc()doc";

static const char *__doc_Logging_Channel = R"doc()doc";

static const char *__doc_Logging_Channel_CHANNEL_BACKEND = R"doc()doc";

static const char *__doc_Logging_Channel_CHANNEL_CODE_GEN = R"doc()doc";

static const char *__doc_Logging_Channel_CHANNEL_GENN = R"doc()doc";

static const char *__doc_Logging_Channel_CHANNEL_MAX = R"doc()doc";

static const char *__doc_Logging_Channel_CHANNEL_RUNTIME = R"doc()doc";

static const char *__doc_Logging_Channel_CHANNEL_TRANSPILER = R"doc()doc";

static const char *__doc_Logging_init = R"doc()doc";

static const char *__doc_ModelSpec = R"doc(Object used for specifying a neuronal network model)doc";

static const char *__doc_ModelSpecInternal = R"doc()doc";

static const char *__doc_ModelSpec_ModelSpec = R"doc()doc";

static const char *__doc_ModelSpec_ModelSpec_2 = R"doc()doc";

static const char *__doc_ModelSpec_addCurrentSource =
R"doc(Adds a new current source to the model using a current source model managed by the user


$Parameter ``currentSourceName``:

 string containing unique name of current source.


$Parameter ``model``:

 current source model to use for current source.


$Parameter ``neuronGroup``:

 pointer to target neuron group


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 state variable initialiser snippets and parameters wrapped in VarValues object.


$Returns:

pointer to newly created CurrentSource)doc";

static const char *__doc_ModelSpec_addCurrentSource_2 =
R"doc(Adds a new current source to the model using a singleton current source model created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros


$Template parameter ``CurrentSourceModel``:

 type of neuron model (derived from CurrentSourceModel::Base).


$Parameter ``currentSourceName``:

 string containing unique name of current source.


$Parameter ``neuronGroup``:

 pointer to target neuron group


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 state variable initialiser snippets and parameters wrapped in VarValues object.


$Returns:

pointer to newly created CurrentSource)doc";

static const char *__doc_ModelSpec_addCustomConnectivityUpdate =
R"doc(Adds a new custom connectivity update attached to synapse group and potentially with synaptic, presynaptic and
postsynaptic state variables and variable references using a custom connectivity update model managed by the user


$Template parameter ``CustomConnectivityUpdateModel``:

 type of custom connectivity update model (derived from CustomConnectivityUpdateModels::Base).


$Parameter ``name``:

 string containing unique name of custom update


$Parameter ``updateGroupName``:

 string containing name of group to add this custom update to


$Parameter ``synapseGroup``:

 pointer to the synapse group whose connectivity this group will update


$Parameter ``model``:

 custom update model to use for custom update.


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 synaptic state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``preVarInitialisers``:

 presynaptic state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``postVarInitialisers``:

 postsynaptic state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``varReferences``:

 variable references wrapped in WUVarReferences object.


$Parameter ``varReferences``:

 variable references wrapped in VarReferences object.


$Parameter ``varReferences``:

 variable references wrapped in VarReferences object.


$Returns:

pointer to newly created CustomConnectivityUpdate)doc";

static const char *__doc_ModelSpec_addCustomConnectivityUpdate_2 =
R"doc(Adds a new custom connectivity update attached to synapse group and potentially with synaptic, presynaptic and
postsynaptic state variables and variable references using a singleton custom connectivity update model created
using standard DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL and IMPLEMENT_MODEL macros


$Template parameter ``CustomConnectivityUpdateModel``:

 type of custom connectivity update model (derived from CustomConnectivityUpdateModels::Base).


$Parameter ``name``:

 string containing unique name of custom update


$Parameter ``updateGroupName``:

 string containing name of group to add this custom update to


$Parameter ``synapseGroup``:

 pointer to the synapse group whose connectivity this group will update


$Parameter ``model``:

 custom update model to use for custom update.


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 synaptic state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``preVarInitialisers``:

 presynaptic state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``postVarInitialisers``:

 postsynaptic state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``varReferences``:

 variable references wrapped in WUVarReferences object.


$Parameter ``varReferences``:

 variable references wrapped in VarReferences object.


$Parameter ``varReferences``:

 variable references wrapped in VarReferences object.


$Returns:

pointer to newly created CustomConnectivityUpdate)doc";

static const char *__doc_ModelSpec_addCustomUpdate =
R"doc(Adds a new custom update with references to the model using a custom update model managed by the user


$Parameter ``name``:

 string containing unique name of custom update


$Parameter ``updateGroupName``:

 string containing name of group to add this custom update to


$Parameter ``model``:

 custom update model to use for custom update.


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``varReferences``:

 variable references wrapped in VarReferences object.


$Returns:

pointer to newly created CustomUpdateBase)doc";

static const char *__doc_ModelSpec_addCustomUpdate_2 =
R"doc(Adds a new custom update with references to weight update model variable to the
model using a custom update model managed by the user


$Parameter ``name``:

 string containing unique name of custom update


$Parameter ``updateGroupName``:

 string containing name of group to add this custom update to


$Parameter ``model``:

 custom update model to use for custom update.


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``varReferences``:

 variable references wrapped in VarReferences object.


$Returns:

pointer to newly created CustomUpdateBase)doc";

static const char *__doc_ModelSpec_addCustomUpdate_3 =
R"doc(Adds a new custom update to the model using a singleton custom update model
created using standard DECLARE_CUSTOM_UPDATE_MODEL and IMPLEMENT_MODEL macros


$Template parameter ``CustomUpdateModel``:

 type of custom update model (derived from CustomUpdateModels::Base).


$Parameter ``name``:

 string containing unique name of custom update


$Parameter ``updateGroupName``:

 string containing name of group to add this custom update to


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``varInitialisers``:

 variable references wrapped in VarReferences object.


$Returns:

pointer to newly created CustomUpdateBase)doc";

static const char *__doc_ModelSpec_addCustomUpdate_4 =
R"doc(Adds a new custom update with references to weight update model variables to the model using a singleton
custom update model created using standard DECLARE_CUSTOM_UPDATE_MODEL and IMPLEMENT_MODEL macros


$Template parameter ``CustomUpdateModel``:

 type of neuron model (derived from CustomUpdateModels::Base).


$Parameter ``name``:

 string containing unique name of custom update


$Parameter ``updateGroupName``:

 string containing name of group to add this custom update to


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 state variable initialiser snippets and parameters wrapped in VarValues object.


$Parameter ``varInitialisers``:

 variable references wrapped in WUVarReferences object.


$Returns:

pointer to newly created CustomUpdateBase)doc";

static const char *__doc_ModelSpec_addNeuronPopulation =
R"doc(Adds a new neuron group to the model using a neuron model managed by the user


$Parameter ``name``:

 string containing unique name of neuron population.


$Parameter ``size``:

 integer specifying how many neurons are in the population.


$Parameter ``model``:

 neuron model to use for neuron group.


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 state variable initialiser snippets and parameters wrapped in VarValues object.


$Returns:

pointer to newly created NeuronGroup)doc";

static const char *__doc_ModelSpec_addNeuronPopulation_2 =
R"doc(Adds a new neuron group to the model using a singleton neuron model created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros


$Template parameter ``NeuronModel``:

 type of neuron model (derived from NeuronModels::Base).


$Parameter ``name``:

 string containing unique name of neuron population.


$Parameter ``size``:

 integer specifying how many neurons are in the population.


$Parameter ``paramValues``:

 parameters for model wrapped in ParamValues object.


$Parameter ``varInitialisers``:

 state variable initialiser snippets and parameters wrapped in VarValues object.


$Returns:

pointer to newly created NeuronGroup)doc";

static const char *__doc_ModelSpec_addSynapsePopulation =
R"doc(Adds a synapse population to the model using weight update and postsynaptic models managed by the user


$Parameter ``name``:

                             string containing unique name of synapse population.


$Parameter ``mtype``:

                            how the synaptic matrix associated with this synapse population should be represented.


$Parameter ``src``:

                              pointer to presynaptic neuron group


$Parameter ``trg``:

                              pointer to postsynaptic neuron group


$Parameter ``wum``:

                              weight update model to use for synapse group.


$Parameter ``wumInitialiser``:

                   WeightUpdateModels::Init object used to initialiser weight update model


$Parameter ``psmInitialiser``:

                   PostsynapticModels::Init object used to initialiser postsynaptic model


$Parameter ``connectivityInitialiser``:

          sparse connectivity initialisation snippet used to initialise connectivity for
SynapseMatrixConnectivity::SPARSE or SynapseMatrixConnectivity::BITMASK.
Typically wrapped with it's parameters using ``initConnectivity`` function


$Returns:

pointer to newly created SynapseGroup)doc";

static const char *__doc_ModelSpec_addSynapsePopulation_2 =
R"doc(Adds a synapse population to the model using weight update and postsynaptic models managed by the user


$Parameter ``name``:

                             string containing unique name of synapse population.


$Parameter ``mtype``:

                            how the synaptic matrix associated with this synapse population should be represented.


$Parameter ``src``:

                              pointer to presynaptic neuron group


$Parameter ``trg``:

                              pointer to postsynaptic neuron group


$Parameter ``wum``:

                              weight update model to use for synapse group.


$Parameter ``wumInitialiser``:

                   WeightUpdateModels::Init object used to initialiser weight update model


$Parameter ``psmInitialiser``:

                   PostsynapticModels::Init object used to initialiser postsynaptic model


$Parameter ``connectivityInitialiser``:

          toeplitz connectivity initialisation snippet used to initialise connectivity for
SynapseMatrixConnectivity::TOEPLITZ. Typically wrapped with it's parameters using ``initToeplitzConnectivity`` function


$Returns:

pointer to newly created SynapseGroup)doc";

static const char *__doc_ModelSpec_addSynapsePopulation_3 = R"doc()doc";

static const char *__doc_ModelSpec_finalise = R"doc(Finalise model)doc";

static const char *__doc_ModelSpec_findCurrentSource = R"doc(Find a current source by name)doc";

static const char *__doc_ModelSpec_findNeuronGroup = R"doc(Find a neuron group by name)doc";

static const char *__doc_ModelSpec_findNeuronGroup_2 = R"doc(Find a neuron group by name)doc";

static const char *__doc_ModelSpec_findSynapseGroup = R"doc(Find a synapse group by name)doc";

static const char *__doc_ModelSpec_findSynapseGroup_2 = R"doc(Find a synapse group by name)doc";

static const char *__doc_ModelSpec_getBatchSize = R"doc()doc";

static const char *__doc_ModelSpec_getCustomConnectivityUpdates = R"doc(Get std::map containing named CustomConnectivity objects in model)doc";

static const char *__doc_ModelSpec_getCustomUpdates = R"doc(Get std::map containing named CustomUpdate objects in model)doc";

static const char *__doc_ModelSpec_getCustomWUUpdates = R"doc()doc";

static const char *__doc_ModelSpec_getDT = R"doc(Gets the model integration step size)doc";

static const char *__doc_ModelSpec_getHashDigest = R"doc(Get hash digest used for detecting changes)doc";

static const char *__doc_ModelSpec_getLocalCurrentSources = R"doc(Get std::map containing local named CurrentSource objects in model)doc";

static const char *__doc_ModelSpec_getName = R"doc(Gets the name of the neuronal network model)doc";

static const char *__doc_ModelSpec_getNeuronGroups = R"doc(Get std::map containing local named NeuronGroup objects in model)doc";

static const char *__doc_ModelSpec_getNumNeurons = R"doc(How many neurons make up the entire model)doc";

static const char *__doc_ModelSpec_getPrecision = R"doc(Gets the floating point numerical precision)doc";

static const char *__doc_ModelSpec_getSeed = R"doc(Get the random seed)doc";

static const char *__doc_ModelSpec_getSynapseGroups = R"doc(Get std::map containing local named SynapseGroup objects in model)doc";

static const char *__doc_ModelSpec_getTimePrecision = R"doc(Gets the floating point numerical precision used to represent time)doc";

static const char *__doc_ModelSpec_getTypeContext = R"doc()doc";

static const char *__doc_ModelSpec_isRecordingInUse = R"doc(Is recording enabled on any population in this model?)doc";

static const char *__doc_ModelSpec_isTimingEnabled = R"doc(Are timers and timing commands enabled)doc";

static const char *__doc_ModelSpec_m_BatchSize = R"doc(Batch size of this model - efficiently duplicates model)doc";

static const char *__doc_ModelSpec_m_CustomConnectivityUpdates = R"doc(Named custom connectivity updates)doc";

static const char *__doc_ModelSpec_m_CustomUpdates = R"doc(Named custom updates)doc";

static const char *__doc_ModelSpec_m_CustomWUUpdates = R"doc()doc";

static const char *__doc_ModelSpec_m_DT = R"doc(The integration time step of the model)doc";

static const char *__doc_ModelSpec_m_DefaultExtraGlobalParamLocation = R"doc(What is the default location for model extra global parameters? Historically, this was just left up to the user to handle)doc";

static const char *__doc_ModelSpec_m_DefaultNarrowSparseIndEnabled = R"doc(The default for whether narrow i.e. less than 32-bit types are used for sparse matrix indices)doc";

static const char *__doc_ModelSpec_m_DefaultSparseConnectivityLocation = R"doc(What is the default location for sparse synaptic connectivity? Historically, everything was allocated on both the host AND device)doc";

static const char *__doc_ModelSpec_m_DefaultVarLocation = R"doc(What is the default location for model state variables? Historically, everything was allocated on both host AND device)doc";

static const char *__doc_ModelSpec_m_LocalCurrentSources = R"doc(Named local current sources)doc";

static const char *__doc_ModelSpec_m_LocalNeuronGroups = R"doc(Named local neuron groups)doc";

static const char *__doc_ModelSpec_m_LocalSynapseGroups = R"doc(Named local synapse groups)doc";

static const char *__doc_ModelSpec_m_Name = R"doc(Name of the neuronal newtwork model)doc";

static const char *__doc_ModelSpec_m_Precision = R"doc(Type of floating point variables (float, double, ...; default: float))doc";

static const char *__doc_ModelSpec_m_Seed = R"doc(RNG seed)doc";

static const char *__doc_ModelSpec_m_ShouldFusePostsynapticModels =
R"doc(Should compatible postsynaptic models and dendritic delay buffers be fused?
This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved)doc";

static const char *__doc_ModelSpec_m_ShouldFusePrePostWeightUpdateModels =
R"doc(Should compatible pre and postsynaptic weight update model variables and updates be fused?
This can significantly reduce the cost of updating neuron populations but means that per-synaptic group per and postsynaptic variables cannot be retrieved)doc";

static const char *__doc_ModelSpec_m_TimePrecision = R"doc(Type of floating point variables used to store time)doc";

static const char *__doc_ModelSpec_m_TimingEnabled = R"doc(Whether timing code should be inserted into model)doc";

static const char *__doc_ModelSpec_m_TypeContext = R"doc()doc";

static const char *__doc_ModelSpec_operator_assign = R"doc()doc";

static const char *__doc_ModelSpec_setBatchSize = R"doc()doc";

static const char *__doc_ModelSpec_setDT = R"doc(Set the integration step size of the model)doc";

static const char *__doc_ModelSpec_setDefaultExtraGlobalParamLocation =
R"doc(What is the default location for model extra global parameters?
Historically, this was just left up to the user to handle)doc";

static const char *__doc_ModelSpec_setDefaultNarrowSparseIndEnabled = R"doc(Sets default for whether narrow i.e. less than 32-bit types are used for sparse matrix indices)doc";

static const char *__doc_ModelSpec_setDefaultSparseConnectivityLocation =
R"doc(What is the default location for sparse synaptic connectivity?
Historically, everything was allocated on both the host AND device)doc";

static const char *__doc_ModelSpec_setDefaultVarLocation =
R"doc(What is the default location for model state variables?
Historically, everything was allocated on both the host AND device)doc";

static const char *__doc_ModelSpec_setFusePostsynapticModels =
R"doc(Should compatible postsynaptic models and dendritic delay buffers be fused?
This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved)doc";

static const char *__doc_ModelSpec_setFusePrePostWeightUpdateModels =
R"doc(Should compatible pre and postsynaptic weight update model variables and updates be fused?
This can significantly reduce the cost of updating neuron populations but means that per-synaptic group per and postsynaptic variables cannot be retrieved)doc";

static const char *__doc_ModelSpec_setMergePostsynapticModels =
R"doc(Should compatible postsynaptic models and dendritic delay buffers be fused?
This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved)doc";

static const char *__doc_ModelSpec_setName = R"doc(Method to set the neuronal network model name)doc";

static const char *__doc_ModelSpec_setPrecision = R"doc(Set numerical precision for scalar type)doc";

static const char *__doc_ModelSpec_setSeed = R"doc(Set the random seed (disables automatic seeding if argument not 0).)doc";

static const char *__doc_ModelSpec_setTimePrecision = R"doc(Set numerical precision for time type)doc";

static const char *__doc_ModelSpec_setTiming = R"doc(Set whether timers and timing commands are to be included)doc";

static const char *__doc_ModelSpec_zeroCopyInUse = R"doc(Are any variables in any populations in this model using zero-copy memory?)doc";

static const char *__doc_Models_Base = R"doc()doc";

static const char *__doc_Models_Base_CustomUpdateVar = R"doc()doc";

static const char *__doc_Models_Base_CustomUpdateVar_CustomUpdateVar = R"doc()doc";

static const char *__doc_Models_Base_CustomUpdateVar_CustomUpdateVar_2 = R"doc()doc";

static const char *__doc_Models_Base_EGPRef = R"doc()doc";

static const char *__doc_Models_Base_EGPRef_EGPRef = R"doc()doc";

static const char *__doc_Models_Base_EGPRef_EGPRef_2 = R"doc()doc";

static const char *__doc_Models_Base_EGPRef_name = R"doc()doc";

static const char *__doc_Models_Base_EGPRef_operator_eq = R"doc()doc";

static const char *__doc_Models_Base_EGPRef_type = R"doc()doc";

static const char *__doc_Models_Base_Var = R"doc()doc";

static const char *__doc_Models_Base_VarBase =
R"doc(A variable has a name, a type and an access type
Explicit constructors required as although, through the wonders of C++
aggregate initialization, access would default to VarAccess::READ_WRITE
if not specified, this results in a -Wmissing-field-initializers warning on GCC and Clang)doc";

static const char *__doc_Models_Base_VarBase_VarBase = R"doc()doc";

static const char *__doc_Models_Base_VarBase_VarBase_2 = R"doc()doc";

static const char *__doc_Models_Base_VarBase_access = R"doc()doc";

static const char *__doc_Models_Base_VarBase_name = R"doc()doc";

static const char *__doc_Models_Base_VarBase_operator_eq = R"doc()doc";

static const char *__doc_Models_Base_VarBase_type = R"doc()doc";

static const char *__doc_Models_Base_VarRef = R"doc()doc";

static const char *__doc_Models_Base_VarRef_VarRef = R"doc()doc";

static const char *__doc_Models_Base_VarRef_VarRef_2 = R"doc()doc";

static const char *__doc_Models_Base_VarRef_access = R"doc()doc";

static const char *__doc_Models_Base_VarRef_name = R"doc()doc";

static const char *__doc_Models_Base_VarRef_operator_eq = R"doc()doc";

static const char *__doc_Models_Base_VarRef_type = R"doc()doc";

static const char *__doc_Models_Base_Var_Var = R"doc()doc";

static const char *__doc_Models_Base_Var_Var_2 = R"doc()doc";

static const char *__doc_Models_EGPReference = R"doc()doc";

static const char *__doc_Models_EGPReference_CUWURef = R"doc()doc";

static const char *__doc_Models_EGPReference_Detail =
R"doc(Minimal helper class for definining unique struct
wrappers around group pointers for use with std::variant)doc";

static const char *__doc_Models_EGPReference_Detail_egp = R"doc()doc";

static const char *__doc_Models_EGPReference_Detail_group = R"doc()doc";

static const char *__doc_Models_EGPReference_EGPReference = R"doc()doc";

static const char *__doc_Models_EGPReference_WURef = R"doc()doc";

static const char *__doc_Models_EGPReference_createEGPRef = R"doc()doc";

static const char *__doc_Models_EGPReference_createEGPRef_2 = R"doc()doc";

static const char *__doc_Models_EGPReference_createEGPRef_3 = R"doc()doc";

static const char *__doc_Models_EGPReference_createEGPRef_4 = R"doc()doc";

static const char *__doc_Models_EGPReference_createPSMEGPRef = R"doc()doc";

static const char *__doc_Models_EGPReference_createWUEGPRef = R"doc()doc";

static const char *__doc_Models_EGPReference_getEGP = R"doc()doc";

static const char *__doc_Models_EGPReference_getEGPName = R"doc()doc";

static const char *__doc_Models_EGPReference_getTargetArray = R"doc(Get array associated with referenced EGP)doc";

static const char *__doc_Models_EGPReference_getTargetName = R"doc()doc";

static const char *__doc_Models_EGPReference_m_Detail = R"doc()doc";

static const char *__doc_Models_VarReference = R"doc()doc";

static const char *__doc_Models_VarReferenceBase = R"doc()doc";

static const char *__doc_Models_VarReferenceBase_Detail =
R"doc(Minimal helper class for definining unique struct
wrappers around group pointers for use with std::variant)doc";

static const char *__doc_Models_VarReferenceBase_Detail_group = R"doc()doc";

static const char *__doc_Models_VarReferenceBase_Detail_var = R"doc()doc";

static const char *__doc_Models_VarReference_CCUPostRef = R"doc()doc";

static const char *__doc_Models_VarReference_CCUPreRef = R"doc()doc";

static const char *__doc_Models_VarReference_CSRef = R"doc()doc";

static const char *__doc_Models_VarReference_CURef = R"doc()doc";

static const char *__doc_Models_VarReference_NGRef = R"doc()doc";

static const char *__doc_Models_VarReference_PSMRef = R"doc()doc";

static const char *__doc_Models_VarReference_VarReference = R"doc()doc";

static const char *__doc_Models_VarReference_WUPostRef = R"doc()doc";

static const char *__doc_Models_VarReference_WUPreRef = R"doc()doc";

static const char *__doc_Models_VarReference_createPSMVarRef = R"doc()doc";

static const char *__doc_Models_VarReference_createPostVarRef = R"doc()doc";

static const char *__doc_Models_VarReference_createPreVarRef = R"doc()doc";

static const char *__doc_Models_VarReference_createVarRef = R"doc()doc";

static const char *__doc_Models_VarReference_createVarRef_2 = R"doc()doc";

static const char *__doc_Models_VarReference_createVarRef_3 = R"doc()doc";

static const char *__doc_Models_VarReference_createWUPostVarRef = R"doc()doc";

static const char *__doc_Models_VarReference_createWUPreVarRef = R"doc()doc";

static const char *__doc_Models_VarReference_getDelayNeuronGroup = R"doc(If variable is delayed, get neuron group which manages its delay)doc";

static const char *__doc_Models_VarReference_getNumNeurons = R"doc(Get size of variable)doc";

static const char *__doc_Models_VarReference_getReferencedCustomUpdate =
R"doc(If this reference points to another custom update, return pointer to it
This is used to detect circular dependencies)doc";

static const char *__doc_Models_VarReference_getTargetArray = R"doc(Get array associated with referenced variable)doc";

static const char *__doc_Models_VarReference_getTargetName = R"doc()doc";

static const char *__doc_Models_VarReference_getVarDims = R"doc()doc";

static const char *__doc_Models_VarReference_getVarName = R"doc(Get name of targetted variable)doc";

static const char *__doc_Models_VarReference_getVarType = R"doc()doc";

static const char *__doc_Models_VarReference_isTargetNeuronGroup = R"doc(Does this variable reference's target belong to neuron group)doc";

static const char *__doc_Models_VarReference_m_Detail = R"doc()doc";

static const char *__doc_Models_VarReference_operator_lt = R"doc()doc";

static const char *__doc_Models_WUVarReference = R"doc()doc";

static const char *__doc_Models_WUVarReference_CCURef = R"doc()doc";

static const char *__doc_Models_WUVarReference_WURef =
R"doc(Struct for storing weight update group variable reference - needs
Additional field to store synapse group associated with transpose)doc";

static const char *__doc_Models_WUVarReference_WURef_group = R"doc()doc";

static const char *__doc_Models_WUVarReference_WURef_transposeGroup = R"doc()doc";

static const char *__doc_Models_WUVarReference_WURef_transposeVar = R"doc()doc";

static const char *__doc_Models_WUVarReference_WURef_var = R"doc()doc";

static const char *__doc_Models_WUVarReference_WUVarReference = R"doc()doc";

static const char *__doc_Models_WUVarReference_createWUVarReference = R"doc()doc";

static const char *__doc_Models_WUVarReference_createWUVarReference_2 = R"doc()doc";

static const char *__doc_Models_WUVarReference_createWUVarReference_3 = R"doc()doc";

static const char *__doc_Models_WUVarReference_getReferencedCustomUpdate =
R"doc(If this reference points to another custom update, return pointer to it
This is used to detect circular dependencies)doc";

static const char *__doc_Models_WUVarReference_getSynapseGroup = R"doc()doc";

static const char *__doc_Models_WUVarReference_getSynapseGroupInternal = R"doc()doc";

static const char *__doc_Models_WUVarReference_getTargetArray = R"doc(Get array associated with referenced variable)doc";

static const char *__doc_Models_WUVarReference_getTargetName = R"doc()doc";

static const char *__doc_Models_WUVarReference_getTransposeSynapseGroup = R"doc()doc";

static const char *__doc_Models_WUVarReference_getTransposeSynapseGroupInternal = R"doc()doc";

static const char *__doc_Models_WUVarReference_getTransposeTargetArray = R"doc(Get array associated with referenced transpose variable)doc";

static const char *__doc_Models_WUVarReference_getTransposeTargetName = R"doc()doc";

static const char *__doc_Models_WUVarReference_getTransposeVarDims = R"doc(Get dimensions of transpose variable being referenced)doc";

static const char *__doc_Models_WUVarReference_getTransposeVarName = R"doc()doc";

static const char *__doc_Models_WUVarReference_getTransposeVarType = R"doc()doc";

static const char *__doc_Models_WUVarReference_getVarDims = R"doc()doc";

static const char *__doc_Models_WUVarReference_getVarName = R"doc()doc";

static const char *__doc_Models_WUVarReference_getVarType = R"doc()doc";

static const char *__doc_Models_WUVarReference_m_Detail = R"doc()doc";

static const char *__doc_Models_WUVarReference_operator_lt = R"doc()doc";

static const char *__doc_Models_checkLocalVarReferences = R"doc(Helper function to check if local variable references are configured correctly)doc";

static const char *__doc_Models_checkVarReferenceTypes = R"doc(Helper function to check if variable reference types match those specified in model)doc";

static const char *__doc_Models_updateHash = R"doc()doc";

static const char *__doc_Models_updateHash_2 = R"doc()doc";

static const char *__doc_Models_updateHash_3 = R"doc()doc";

static const char *__doc_Models_updateHash_4 = R"doc()doc";

static const char *__doc_NeuronEGPAdapter = R"doc()doc";

static const char *__doc_NeuronEGPAdapter_NeuronEGPAdapter = R"doc()doc";

static const char *__doc_NeuronEGPAdapter_getDefs = R"doc()doc";

static const char *__doc_NeuronEGPAdapter_getLoc = R"doc()doc";

static const char *__doc_NeuronEGPAdapter_m_NG = R"doc()doc";

static const char *__doc_NeuronGroup = R"doc()doc";

static const char *__doc_NeuronGroup_2 = R"doc()doc";

static const char *__doc_NeuronGroupInternal = R"doc()doc";

static const char *__doc_NeuronGroupInternal_2 = R"doc()doc";

static const char *__doc_NeuronGroupInternal_3 = R"doc()doc";

static const char *__doc_NeuronGroupInternal_4 = R"doc()doc";

static const char *__doc_NeuronGroupInternal_NeuronGroupInternal = R"doc()doc";

static const char *__doc_NeuronGroup_NeuronGroup = R"doc()doc";

static const char *__doc_NeuronGroup_NeuronGroup_2 = R"doc()doc";

static const char *__doc_NeuronGroup_NeuronGroup_3 = R"doc()doc";

static const char *__doc_NeuronGroup_addInSyn = R"doc()doc";

static const char *__doc_NeuronGroup_addOutSyn = R"doc()doc";

static const char *__doc_NeuronGroup_checkNumDelaySlots = R"doc(Checks delay slots currently provided by the neuron group against a required delay and extends if required)doc";

static const char *__doc_NeuronGroup_finalise = R"doc()doc";

static const char *__doc_NeuronGroup_fusePrePostSynapses = R"doc(Fuse incoming postsynaptic models)doc";

static const char *__doc_NeuronGroup_getCurrentSources = R"doc(Gets pointers to all current sources which provide input to this neuron group)doc";

static const char *__doc_NeuronGroup_getDerivedParams = R"doc()doc";

static const char *__doc_NeuronGroup_getExtraGlobalParamLocation = R"doc(Get location of neuron model extra global parameter by name)doc";

static const char *__doc_NeuronGroup_getFusedInSynWithPostCode = R"doc(Helper to get vector of incoming synapse groups which have postsynaptic update code)doc";

static const char *__doc_NeuronGroup_getFusedInSynWithPostVars = R"doc(Helper to get vector of incoming synapse groups which have postsynaptic variables)doc";

static const char *__doc_NeuronGroup_getFusedOutSynWithPreCode = R"doc(Helper to get vector of outgoing synapse groups which have presynaptic update code)doc";

static const char *__doc_NeuronGroup_getFusedOutSynWithPreVars = R"doc(Helper to get vector of outgoing synapse groups which have presynaptic variables)doc";

static const char *__doc_NeuronGroup_getFusedPSMInSyn = R"doc()doc";

static const char *__doc_NeuronGroup_getFusedPreOutputOutSyn = R"doc()doc";

static const char *__doc_NeuronGroup_getFusedSpike = R"doc()doc";

static const char *__doc_NeuronGroup_getFusedSpikeEvent = R"doc()doc";

static const char *__doc_NeuronGroup_getFusedWUPostInSyn = R"doc()doc";

static const char *__doc_NeuronGroup_getFusedWUPreOutSyn = R"doc()doc";

static const char *__doc_NeuronGroup_getHashDigest =
R"doc(Updates hash with neuron group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_NeuronGroup_getInSyn = R"doc(Gets pointers to all synapse groups which provide input to this neuron group)doc";

static const char *__doc_NeuronGroup_getInitHashDigest =
R"doc(Updates hash with neuron group initialisation
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_NeuronGroup_getModel = R"doc(Gets the neuron model used by this group)doc";

static const char *__doc_NeuronGroup_getName = R"doc()doc";

static const char *__doc_NeuronGroup_getNumDelaySlots = R"doc()doc";

static const char *__doc_NeuronGroup_getNumNeurons = R"doc(Gets number of neurons in group)doc";

static const char *__doc_NeuronGroup_getOutSyn = R"doc(Gets pointers to all synapse groups emanating from this neuron group)doc";

static const char *__doc_NeuronGroup_getParams = R"doc()doc";

static const char *__doc_NeuronGroup_getPrevSpikeEventTimeLocation = R"doc(Get location of this neuron group's previous output spike-like-event times)doc";

static const char *__doc_NeuronGroup_getPrevSpikeTimeLocation = R"doc(Get location of this neuron group's previous output spike times)doc";

static const char *__doc_NeuronGroup_getPrevSpikeTimeUpdateHashDigest = R"doc()doc";

static const char *__doc_NeuronGroup_getResetCodeTokens = R"doc(Tokens produced by scanner from reset code)doc";

static const char *__doc_NeuronGroup_getSimCodeTokens = R"doc(Tokens produced by scanner from simc ode)doc";

static const char *__doc_NeuronGroup_getSpikeEventLocation = R"doc(Get location of this neuron group's output spike events)doc";

static const char *__doc_NeuronGroup_getSpikeEventTimeLocation = R"doc(Get location of this neuron group's output spike-like-event times)doc";

static const char *__doc_NeuronGroup_getSpikeLocation = R"doc(Get location of this neuron group's output spikes)doc";

static const char *__doc_NeuronGroup_getSpikeQueueUpdateHashDigest = R"doc()doc";

static const char *__doc_NeuronGroup_getSpikeTimeLocation = R"doc(Get location of this neuron group's output spike times)doc";

static const char *__doc_NeuronGroup_getThresholdConditionCodeTokens = R"doc(Tokens produced by scanner from threshold condition code)doc";

static const char *__doc_NeuronGroup_getVarInitialisers = R"doc()doc";

static const char *__doc_NeuronGroup_getVarLocation = R"doc(Get location of neuron model state variable by name)doc";

static const char *__doc_NeuronGroup_getVarLocationHashDigest = R"doc()doc";

static const char *__doc_NeuronGroup_injectCurrent = R"doc(add input current source)doc";

static const char *__doc_NeuronGroup_isDelayRequired = R"doc()doc";

static const char *__doc_NeuronGroup_isInitRNGRequired = R"doc(Does this neuron group require an RNG for it's init code?)doc";

static const char *__doc_NeuronGroup_isParamDynamic = R"doc(Is parameter dynamic i.e. it can be changed at runtime)doc";

static const char *__doc_NeuronGroup_isPrevSpikeEventTimeRequired = R"doc()doc";

static const char *__doc_NeuronGroup_isPrevSpikeTimeRequired = R"doc()doc";

static const char *__doc_NeuronGroup_isRecordingEnabled = R"doc(Does this neuron group require any sort of recording?)doc";

static const char *__doc_NeuronGroup_isRecordingZeroCopyEnabled =
R"doc(Get whether zero-copy memory (if available) should
be used for spike and spike-like event recording)doc";

static const char *__doc_NeuronGroup_isSimRNGRequired = R"doc(Does this neuron group require an RNG to simulate?)doc";

static const char *__doc_NeuronGroup_isSpikeEventRecordingEnabled = R"doc(Is spike event recording enabled for this population?)doc";

static const char *__doc_NeuronGroup_isSpikeEventRequired = R"doc()doc";

static const char *__doc_NeuronGroup_isSpikeEventTimeRequired = R"doc()doc";

static const char *__doc_NeuronGroup_isSpikeRecordingEnabled = R"doc(Is spike recording enabled for this population?)doc";

static const char *__doc_NeuronGroup_isSpikeTimeRequired = R"doc()doc";

static const char *__doc_NeuronGroup_isTrueSpikeRequired = R"doc()doc";

static const char *__doc_NeuronGroup_isVarInitRequired =
R"doc(Does this neuron group require any variables initializing?
Because it occurs in the same kernel, this includes current source variables;
postsynaptic model variables and postsynaptic weight update variables
from incoming synapse groups; and presynaptic weight update variables from outgoing synapse groups)doc";

static const char *__doc_NeuronGroup_isVarQueueRequired = R"doc()doc";

static const char *__doc_NeuronGroup_isZeroCopyEnabled = R"doc()doc";

static const char *__doc_NeuronGroup_m_CurrentSourceGroups = R"doc()doc";

static const char *__doc_NeuronGroup_m_DerivedParams = R"doc()doc";

static const char *__doc_NeuronGroup_m_DynamicParams = R"doc(Data structure tracking whether parameters are dynamic or not)doc";

static const char *__doc_NeuronGroup_m_ExtraGlobalParamLocation = R"doc(Location of extra global parameters)doc";

static const char *__doc_NeuronGroup_m_FusedPSMInSyn = R"doc()doc";

static const char *__doc_NeuronGroup_m_FusedPreOutputOutSyn = R"doc()doc";

static const char *__doc_NeuronGroup_m_FusedSpike = R"doc()doc";

static const char *__doc_NeuronGroup_m_FusedSpikeEvent = R"doc()doc";

static const char *__doc_NeuronGroup_m_FusedWUPostInSyn = R"doc()doc";

static const char *__doc_NeuronGroup_m_FusedWUPreOutSyn = R"doc()doc";

static const char *__doc_NeuronGroup_m_InSyn = R"doc()doc";

static const char *__doc_NeuronGroup_m_Model = R"doc()doc";

static const char *__doc_NeuronGroup_m_Name = R"doc()doc";

static const char *__doc_NeuronGroup_m_NumDelaySlots = R"doc()doc";

static const char *__doc_NeuronGroup_m_NumNeurons = R"doc()doc";

static const char *__doc_NeuronGroup_m_OutSyn = R"doc()doc";

static const char *__doc_NeuronGroup_m_Params = R"doc()doc";

static const char *__doc_NeuronGroup_m_PrevSpikeEventTimeLocation = R"doc(Location of previous spike-like-event times)doc";

static const char *__doc_NeuronGroup_m_PrevSpikeTimeLocation = R"doc(Location of previous spike times)doc";

static const char *__doc_NeuronGroup_m_RecordingZeroCopyEnabled =
R"doc(Should zero-copy memory (if available) be used
for spike and spike-like event recording?)doc";

static const char *__doc_NeuronGroup_m_ResetCodeTokens = R"doc(Tokens produced by scanner from reset code)doc";

static const char *__doc_NeuronGroup_m_SimCodeTokens = R"doc(Tokens produced by scanner from simc ode)doc";

static const char *__doc_NeuronGroup_m_SpikeEventLocation = R"doc(Location of spike-like events from neuron group)doc";

static const char *__doc_NeuronGroup_m_SpikeEventRecordingEnabled = R"doc(Is spike event recording enabled?)doc";

static const char *__doc_NeuronGroup_m_SpikeEventTimeLocation = R"doc(Location of spike-like-event times)doc";

static const char *__doc_NeuronGroup_m_SpikeLocation = R"doc(Location of spikes from neuron group)doc";

static const char *__doc_NeuronGroup_m_SpikeRecordingEnabled = R"doc(Is spike recording enabled for this population?)doc";

static const char *__doc_NeuronGroup_m_SpikeTimeLocation = R"doc(Location of spike times from neuron group)doc";

static const char *__doc_NeuronGroup_m_ThresholdConditionCodeTokens = R"doc(Tokens produced by scanner from threshold condition code)doc";

static const char *__doc_NeuronGroup_m_VarInitialisers = R"doc()doc";

static const char *__doc_NeuronGroup_m_VarLocation = R"doc(Location of individual state variables)doc";

static const char *__doc_NeuronGroup_m_VarQueueRequired = R"doc(Set of names of variable requiring queueing)doc";

static const char *__doc_NeuronGroup_setExtraGlobalParamLocation =
R"doc(Set location of neuron model extra global parameter
This is ignored for simulations on hardware with a single memory space.)doc";

static const char *__doc_NeuronGroup_setParamDynamic = R"doc(Set whether parameter is dynamic or not i.e. it can be changed at runtime)doc";

static const char *__doc_NeuronGroup_setPrevSpikeEventTimeLocation =
R"doc(Set location of this neuron group's previous output spike-like-event times
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_NeuronGroup_setPrevSpikeTimeLocation =
R"doc(Set location of this neuron group's previous output spike times
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_NeuronGroup_setRecordingZeroCopyEnabled =
R"doc(Set whether zero-copy memory (if available) should be
used for spike and spike-like event recording
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_NeuronGroup_setSpikeEventLocation =
R"doc(Set location of this neuron group's output spike events
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_NeuronGroup_setSpikeEventRecordingEnabled = R"doc(Enables and disable spike event recording for this population)doc";

static const char *__doc_NeuronGroup_setSpikeEventTimeLocation =
R"doc(Set location of this neuron group's output spike-like-event times
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_NeuronGroup_setSpikeLocation =
R"doc(Set location of this neuron group's output spikes
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_NeuronGroup_setSpikeRecordingEnabled = R"doc(Enables and disable spike recording for this population)doc";

static const char *__doc_NeuronGroup_setSpikeTimeLocation =
R"doc(Set location of this neuron group's output spike times
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_NeuronGroup_setVarLocation =
R"doc(Set variable location of neuron model state variable
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_NeuronGroup_setVarQueueRequired = R"doc()doc";

static const char *__doc_NeuronModels_Base = R"doc(Base class for all neuron models)doc";

static const char *__doc_NeuronModels_Base_getAdditionalInputVars =
R"doc(Gets names, types (as strings) and initial values of local variables into which
the 'apply input code' of (potentially) multiple postsynaptic input models can apply input)doc";

static const char *__doc_NeuronModels_Base_getHashDigest = R"doc(Update hash from model)doc";

static const char *__doc_NeuronModels_Base_getResetCode = R"doc(Gets code that defines the reset action taken after a spike occurred. This can be empty)doc";

static const char *__doc_NeuronModels_Base_getSimCode =
R"doc(Gets the code that defines the execution of one timestep of integration of the neuron model.
The code will refer to NN for the value of the variable with name "NN".
It needs to refer to the predefined variable "ISYN", i.e. contain ISYN, if it is to receive input.)doc";

static const char *__doc_NeuronModels_Base_getThresholdConditionCode =
R"doc(Gets code which defines the condition for a true spike in the described neuron model.
This evaluates to a bool (e.g. "V > 20").)doc";

static const char *__doc_NeuronModels_Base_getVar = R"doc(Find the named variable)doc";

static const char *__doc_NeuronModels_Base_getVars = R"doc(Gets model variables)doc";

static const char *__doc_NeuronModels_Base_isAutoRefractoryRequired = R"doc(Does this model require auto-refractory logic?)doc";

static const char *__doc_NeuronModels_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_NeuronModels_Izhikevich =
R"doc(Izhikevich neuron with fixed parameters \cite izhikevich2003simple.
It is usually described as
\f{eqnarray*}
\frac{dV}{dt} &=& 0.04 V^2 + 5 V + 140 - U + I, \\
\frac{dU}{dt} &=& a (bV-U),
\f}
I is an external input current and the voltage V is reset to parameter c and U incremented by parameter d, whenever V >= 30 mV. This is paired with a particular integration procedure of two 0.5 ms Euler time steps for the V equation followed by one 1 ms time step of the U equation. Because of its popularity we provide this model in this form here event though due to the details of the usual implementation it is strictly speaking inconsistent with the displayed equations.

Variables are:

- ``V`` - Membrane potential
- ``U`` - Membrane recovery variable

Parameters are:

- ``a`` - time scale of U
- ``b`` - sensitivity of U
- ``c`` - after-spike reset value of V
- ``d`` - after-spike reset value of U)doc";

static const char *__doc_NeuronModels_IzhikevichVariable =
R"doc(Izhikevich neuron with variable parameters \cite izhikevich2003simple.
This is the same model as NeuronModels::Izhikevich but parameters are defined as
"variables" in order to allow users to provide individual values for each
individual neuron instead of fixed values for all neurons across the population.

Accordingly, the model has the Variables:
- ``V`` - Membrane potential
- ``U`` - Membrane recovery variable
- ``a`` - time scale of U
- ``b`` - sensitivity of U
- ``c`` - after-spike reset value of V
- ``d`` - after-spike reset value of U

and no parameters.)doc";

static const char *__doc_NeuronModels_IzhikevichVariable_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_IzhikevichVariable_getParams = R"doc()doc";

static const char *__doc_NeuronModels_IzhikevichVariable_getVars = R"doc()doc";

static const char *__doc_NeuronModels_Izhikevich_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_Izhikevich_getParams = R"doc()doc";

static const char *__doc_NeuronModels_Izhikevich_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_Izhikevich_getThresholdConditionCode = R"doc()doc";

static const char *__doc_NeuronModels_Izhikevich_getVars = R"doc()doc";

static const char *__doc_NeuronModels_Izhikevich_isAutoRefractoryRequired = R"doc()doc";

static const char *__doc_NeuronModels_LIF = R"doc()doc";

static const char *__doc_NeuronModels_LIF_getDerivedParams = R"doc()doc";

static const char *__doc_NeuronModels_LIF_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_LIF_getParams = R"doc()doc";

static const char *__doc_NeuronModels_LIF_getResetCode = R"doc()doc";

static const char *__doc_NeuronModels_LIF_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_LIF_getThresholdConditionCode = R"doc()doc";

static const char *__doc_NeuronModels_LIF_getVars = R"doc()doc";

static const char *__doc_NeuronModels_LIF_isAutoRefractoryRequired = R"doc()doc";

static const char *__doc_NeuronModels_Poisson =
R"doc(Poisson neurons
Poisson neurons have constant membrane potential (``Vrest``) unless they are
activated randomly to the ``Vspike`` value if (t- ``spikeTime`` ) > ``trefract``.

It has 2 variables:

- ``V`` - Membrane potential (mV)
- ``spikeTime`` - Time at which the neuron spiked for the last time (ms)

and 4 parameters:

- ``trefract`` - Refractory period (ms)
- ``tspike`` - duration of spike (ms)
- ``Vspike`` - Membrane potential at spike (mV)
- ``Vrest`` - Membrane potential at rest (mV)

\note The initial values array for the `Poisson` type needs two entries
for `V`, and `spikeTime` and the parameter array needs four entries for
`trefract`, `tspike`, `Vspike` and `Vrest`,  *in that order*.
\note The refractory period and the spike duration both start at the beginning of the spike. That means that the refractory period should be longer or equal to the spike duration. If this is not the case, undefined model behaviour occurs.

It has two extra global parameters:

- ``firingProb`` - an array of firing probabilities/ average rates; this can extend to :math:`n \cdot N`, where :math:`N` is the number of neurons, for :math:`n > 0` firing patterns
- ``offset`` - an unsigned integer that points to the start of the currently used input pattern; typically taking values of :math:`i \cdot N`, :math:`0 \leq i < n`.

\note This model uses a linear approximation for the probability
of firing a spike in a given time step of size `DT`, i.e. the
probability of firing is :math:`\lambda` times `DT`: :math:` p = \lambda \Delta t`,
where $\lambda$ corresponds to the value of the relevant entry of `firingProb`.
This approximation is usually very good, especially for typical,
quite small time steps and moderate firing rates. However, it is worth
noting that the approximation becomes poor for very high firing rates
and large time steps.)doc";

static const char *__doc_NeuronModels_PoissonNew =
R"doc(Poisson neurons
This neuron model emits spikes according to the Poisson distribution with a mean firing
rate as determined by its single parameter.
It has 1 state variable:

- ``timeStepToSpike`` - Number of timesteps to next spike

and 1 parameter:

- ``rate`` - Mean firing rate (Hz)

\note Internally this samples from the exponential distribution using
the C++ 11 \<random\> library on the CPU and by transforming the
uniform distribution, generated using cuRAND, with a natural log on the GPU.)doc";

static const char *__doc_NeuronModels_PoissonNew_getDerivedParams = R"doc()doc";

static const char *__doc_NeuronModels_PoissonNew_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_PoissonNew_getParams = R"doc()doc";

static const char *__doc_NeuronModels_PoissonNew_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_PoissonNew_getThresholdConditionCode = R"doc()doc";

static const char *__doc_NeuronModels_PoissonNew_getVars = R"doc()doc";

static const char *__doc_NeuronModels_PoissonNew_isAutoRefractoryRequired = R"doc()doc";

static const char *__doc_NeuronModels_Poisson_getExtraGlobalParams = R"doc()doc";

static const char *__doc_NeuronModels_Poisson_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_Poisson_getParams = R"doc()doc";

static const char *__doc_NeuronModels_Poisson_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_Poisson_getThresholdConditionCode = R"doc()doc";

static const char *__doc_NeuronModels_Poisson_getVars = R"doc()doc";

static const char *__doc_NeuronModels_RulkovMap =
R"doc(Rulkov Map neuron
The RulkovMap type is a map based neuron model based on \cite Rulkov2002 but in
the 1-dimensional map form used in \cite nowotny2005self :
\f{eqnarray*}{
V(t+\Delta t) &=& \left\{ \begin{array}{ll}
V_{\rm spike} \Big(\frac{\alpha V_{\rm spike}}{V_{\rm spike}-V(t) \beta I_{\rm syn}} + y \Big) & V(t) \leq 0 \\
V_{\rm spike} \big(\alpha+y\big) & V(t) \leq V_{\rm spike} \big(\alpha + y\big) \; \& \; V(t-\Delta t) \leq 0 \\
-V_{\rm spike} & {\rm otherwise}
\end{array}
\right.
\f}
\note
The `RulkovMap` type only works as intended for the single time step size of `DT`= 0.5.

The `RulkovMap` type has 2 variables:
- ``V`` - the membrane potential
- ``preV`` - the membrane potential at the previous time step

and it has 4 parameters:
- ``Vspike`` - determines the amplitude of spikes, typically -60mV
- ``alpha`` - determines the shape of the iteration function, typically :math:`\alpha `= 3
- ``y`` - "shift / excitation" parameter, also determines the iteration function,originally, y= -2.468
- ``beta`` - roughly speaking equivalent to the input resistance, i.e. it regulates the scale of the input into the neuron, typically :math:`\beta`= 2.64 :math:`{\rm M}\Omega`.

\note
The initial values array for the `RulkovMap` type needs two entries for `V` and `preV` and the
parameter array needs four entries for `Vspike`, `alpha`, `y` and `beta`,  *in that order*.)doc";

static const char *__doc_NeuronModels_RulkovMap_getDerivedParams = R"doc()doc";

static const char *__doc_NeuronModels_RulkovMap_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_RulkovMap_getParams = R"doc()doc";

static const char *__doc_NeuronModels_RulkovMap_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_RulkovMap_getThresholdConditionCode = R"doc()doc";

static const char *__doc_NeuronModels_RulkovMap_getVars = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSource =
R"doc(Empty neuron which allows setting spikes from external sources
This model does not contain any update code and can be used to implement
the equivalent of a SpikeGeneratorGroup in Brian or a SpikeSourceArray in PyNN.)doc";

static const char *__doc_NeuronModels_SpikeSourceArray =
R"doc(Spike source array
A neuron which reads spike times from a global spikes array.
It has 2 variables:

- ``startSpike`` - Index of the next spike in the global array
- ``endSpike``   - Index of the spike next to the last in the globel array

and 1 extra global parameter:

- ``spikeTimes`` - Array with all spike times)doc";

static const char *__doc_NeuronModels_SpikeSourceArray_getExtraGlobalParams = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSourceArray_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSourceArray_getResetCode = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSourceArray_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSourceArray_getThresholdConditionCode = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSourceArray_getVars = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSourceArray_isAutoRefractoryRequired = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSource_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSource_getThresholdConditionCode = R"doc()doc";

static const char *__doc_NeuronModels_SpikeSource_isAutoRefractoryRequired = R"doc()doc";

static const char *__doc_NeuronModels_TraubMiles =
R"doc(Hodgkin-Huxley neurons with Traub & Miles algorithm.
This conductance based model has been taken from \cite Traub1991 and can be described by the equations:
\f{eqnarray*}{
C \frac{d V}{dt}  &=& -I_{{\rm Na}} -I_K-I_{{\rm leak}}-I_M-I_{i,DC}-I_{i,{\rm syn}}-I_i, \\
I_{{\rm Na}}(t) &=& g_{{\rm Na}} m_i(t)^3 h_i(t)(V_i(t)-E_{{\rm Na}}) \\
I_{{\rm K}}(t) &=& g_{{\rm K}} n_i(t)^4(V_i(t)-E_{{\rm K}})  \\
\frac{dy(t)}{dt} &=& \alpha_y (V(t))(1-y(t))-\beta_y(V(t)) y(t), \f}
where :math:`y_i= m, h, n`, and
\f{eqnarray*}{
\alpha_n&=& 0.032(-50-V)/\big(\exp((-50-V)/5)-1\big)  \\
\beta_n &=& 0.5\exp((-55-V)/40)  \\
\alpha_m &=& 0.32(-52-V)/\big(\exp((-52-V)/4)-1\big)  \\
\beta_m &=& 0.28(25+V)/\big(\exp((25+V)/5)-1\big)  \\
\alpha_h &=& 0.128\exp((-48-V)/18)  \\
\beta_h &=& 4/\big(\exp((-25-V)/5)+1\big).
\f}
and typical parameters are :math:`C=0.143` nF, :math:`g_{{\rm leak}}= 0.02672`
:math:`\mu`S, :math:`E_{{\rm leak}}= -63.563` mV, :math:`g_{{\rm Na}}=7.15` :math:`\mu`S,
:math:`E_{{\rm Na}}= 50` mV, :math:`g_{{\rm {\rm K}}}=1.43` :math:`\mu`S,
:math:`E_{{\rm K}}= -95` mV.

It has 4 variables:

- ``V`` - membrane potential E
- ``m`` - probability for Na channel activation m
- ``h`` - probability for not Na channel blocking h
- ``n`` - probability for K channel activation n

and 7 parameters:

- ``gNa`` - Na conductance in 1/(mOhms * cm^2)
- ``ENa`` - Na equi potential in mV
- ``gK`` - K conductance in 1/(mOhms * cm^2)
- ``EK`` - K equi potential in mV
- ``gl`` - Leak conductance in 1/(mOhms * cm^2)
- ``El`` - Leak equi potential in mV
- ``C`` - Membrane capacity density in muF/cm^2

\note
Internally, the ordinary differential equations defining the model are integrated with a
linear Euler algorithm and GeNN integrates 25 internal time steps for each neuron for each
network time step. I.e., if the network is simulated at `DT= 0.1` ms, then the neurons are
integrated with a linear Euler algorithm with `lDT= 0.004` ms.
This variant uses IF statements to check for a value at which a singularity would be hit.
If so, value calculated by L'Hospital rule is used.)doc";

static const char *__doc_NeuronModels_TraubMilesAlt =
R"doc(Hodgkin-Huxley neurons with Traub & Miles algorithm
Using a workaround to avoid singularity: adding the munimum numerical value of the floating point precision used.
\note See NeuronModels::TraubMiles for variable and parameter names.)doc";

static const char *__doc_NeuronModels_TraubMilesAlt_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_TraubMilesAlt_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_TraubMilesFast =
R"doc(Hodgkin-Huxley neurons with Traub & Miles algorithm: Original fast implementation, using 25 inner iterations.
There are singularities in this model, which can be easily hit in float precision
\note See NeuronModels::TraubMiles for variable and parameter names.)doc";

static const char *__doc_NeuronModels_TraubMilesFast_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_TraubMilesFast_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_TraubMilesNStep =
R"doc(Hodgkin-Huxley neurons with Traub & Miles algorithm.
Same as standard TraubMiles model but number of inner loops can be set using a parameter
\note See NeuronModels::TraubMiles for variable and parameter names.)doc";

static const char *__doc_NeuronModels_TraubMilesNStep_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_TraubMilesNStep_getParams = R"doc()doc";

static const char *__doc_NeuronModels_TraubMilesNStep_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_TraubMiles_getInstance = R"doc()doc";

static const char *__doc_NeuronModels_TraubMiles_getParams = R"doc()doc";

static const char *__doc_NeuronModels_TraubMiles_getSimCode = R"doc()doc";

static const char *__doc_NeuronModels_TraubMiles_getThresholdConditionCode = R"doc()doc";

static const char *__doc_NeuronModels_TraubMiles_getVars = R"doc()doc";

static const char *__doc_NeuronVarAdapter = R"doc()doc";

static const char *__doc_NeuronVarAdapter_NeuronVarAdapter = R"doc()doc";

static const char *__doc_NeuronVarAdapter_getDefs = R"doc()doc";

static const char *__doc_NeuronVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_NeuronVarAdapter_getLoc = R"doc()doc";

static const char *__doc_NeuronVarAdapter_getTarget = R"doc()doc";

static const char *__doc_NeuronVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_NeuronVarAdapter_isVarDelayed = R"doc()doc";

static const char *__doc_NeuronVarAdapter_m_NG = R"doc()doc";

static const char *__doc_PostsynapticModels_Base = R"doc(Base class for all postsynaptic models)doc";

static const char *__doc_PostsynapticModels_Base_getHashDigest = R"doc(Update hash from model)doc";

static const char *__doc_PostsynapticModels_Base_getNeuronVarRefs = R"doc(Gets names and types of model variable references)doc";

static const char *__doc_PostsynapticModels_Base_getSimCode = R"doc()doc";

static const char *__doc_PostsynapticModels_Base_getVar = R"doc(Find the named variable)doc";

static const char *__doc_PostsynapticModels_Base_getVars = R"doc(Gets model variables)doc";

static const char *__doc_PostsynapticModels_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_PostsynapticModels_DeltaCurr =
R"doc(Simple delta current synapse.
Synaptic input provides a direct inject of instantaneous current)doc";

static const char *__doc_PostsynapticModels_DeltaCurr_getInstance = R"doc()doc";

static const char *__doc_PostsynapticModels_DeltaCurr_getSimCode = R"doc()doc";

static const char *__doc_PostsynapticModels_ExpCond =
R"doc(Exponential decay with synaptic input treated as a conductance value.
This model has no variables, two parameters and a variable reference
- ``tau`` : Decay time constant
- ``E``   : Reversal potential
- ``V``   : Is a reference to the neuron's membrane voltage
``tau`` is used by the derived parameter ``expdecay`` which returns expf(-dt/tau).)doc";

static const char *__doc_PostsynapticModels_ExpCond_getDerivedParams = R"doc()doc";

static const char *__doc_PostsynapticModels_ExpCond_getInstance = R"doc()doc";

static const char *__doc_PostsynapticModels_ExpCond_getNeuronVarRefs = R"doc()doc";

static const char *__doc_PostsynapticModels_ExpCond_getParams = R"doc()doc";

static const char *__doc_PostsynapticModels_ExpCond_getSimCode = R"doc()doc";

static const char *__doc_PostsynapticModels_ExpCurr =
R"doc(Exponential decay with synaptic input treated as a current value.
This model has no variables and a single parameter:
- ``tau`` : Decay time constant)doc";

static const char *__doc_PostsynapticModels_ExpCurr_getDerivedParams = R"doc()doc";

static const char *__doc_PostsynapticModels_ExpCurr_getInstance = R"doc()doc";

static const char *__doc_PostsynapticModels_ExpCurr_getParams = R"doc()doc";

static const char *__doc_PostsynapticModels_ExpCurr_getSimCode = R"doc()doc";

static const char *__doc_PostsynapticModels_Init = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_Init = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_finalise = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_getNeuronVarReferences = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_getSimCodeTokens = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_getVarInitialisers = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_isRNGRequired = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_isVarInitRequired = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_m_NeuronVarReferences = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_m_SimCodeTokens = R"doc()doc";

static const char *__doc_PostsynapticModels_Init_m_VarInitialisers = R"doc()doc";

static const char *__doc_Runtime_ArrayBase = R"doc()doc";

static const char *__doc_Runtime_Runtime = R"doc()doc";

static const char *__doc_Snippet_Base = R"doc()doc";

static const char *__doc_Snippet_Base_DerivedParam = R"doc(A derived parameter has a name and a function for obtaining its value)doc";

static const char *__doc_Snippet_Base_DerivedParam_DerivedParam = R"doc()doc";

static const char *__doc_Snippet_Base_DerivedParam_DerivedParam_2 = R"doc()doc";

static const char *__doc_Snippet_Base_DerivedParam_func = R"doc()doc";

static const char *__doc_Snippet_Base_DerivedParam_name = R"doc()doc";

static const char *__doc_Snippet_Base_DerivedParam_operator_eq = R"doc()doc";

static const char *__doc_Snippet_Base_DerivedParam_type = R"doc()doc";

static const char *__doc_Snippet_Base_EGP = R"doc(An extra global parameter has a name and a type)doc";

static const char *__doc_Snippet_Base_EGP_EGP = R"doc()doc";

static const char *__doc_Snippet_Base_EGP_EGP_2 = R"doc()doc";

static const char *__doc_Snippet_Base_EGP_name = R"doc()doc";

static const char *__doc_Snippet_Base_EGP_operator_eq = R"doc()doc";

static const char *__doc_Snippet_Base_EGP_type = R"doc()doc";

static const char *__doc_Snippet_Base_Param = R"doc(A parameter has a name and a type)doc";

static const char *__doc_Snippet_Base_ParamVal = R"doc(Additional input variables, row state variables and other things have a name, a type and an initial value)doc";

static const char *__doc_Snippet_Base_ParamVal_ParamVal = R"doc()doc";

static const char *__doc_Snippet_Base_ParamVal_ParamVal_2 = R"doc()doc";

static const char *__doc_Snippet_Base_ParamVal_name = R"doc()doc";

static const char *__doc_Snippet_Base_ParamVal_operator_eq = R"doc()doc";

static const char *__doc_Snippet_Base_ParamVal_type = R"doc()doc";

static const char *__doc_Snippet_Base_ParamVal_value = R"doc()doc";

static const char *__doc_Snippet_Base_Param_Param = R"doc()doc";

static const char *__doc_Snippet_Base_Param_Param_2 = R"doc()doc";

static const char *__doc_Snippet_Base_Param_Param_3 = R"doc()doc";

static const char *__doc_Snippet_Base_Param_name = R"doc()doc";

static const char *__doc_Snippet_Base_Param_operator_eq = R"doc()doc";

static const char *__doc_Snippet_Base_Param_type = R"doc()doc";

static const char *__doc_Snippet_Base_getDerivedParams =
R"doc(Gets names of derived model parameters and the function objects to call to
Calculate their value from a vector of model parameter values)doc";

static const char *__doc_Snippet_Base_getExtraGlobalParam = R"doc(Find the named extra global parameter)doc";

static const char *__doc_Snippet_Base_getExtraGlobalParams =
R"doc(Gets names and types (as strings) of additional
per-population parameters for the snippet)doc";

static const char *__doc_Snippet_Base_getNamed = R"doc()doc";

static const char *__doc_Snippet_Base_getParam = R"doc(Find the named parameter)doc";

static const char *__doc_Snippet_Base_getParams = R"doc(Gets names and types of (independent) model parameters)doc";

static const char *__doc_Snippet_Base_updateHash = R"doc()doc";

static const char *__doc_Snippet_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_Snippet_DynamicParameterContainer = R"doc()doc";

static const char *__doc_Snippet_DynamicParameterContainer_get = R"doc()doc";

static const char *__doc_Snippet_DynamicParameterContainer_m_Dynamic = R"doc()doc";

static const char *__doc_Snippet_DynamicParameterContainer_m_Snippet = R"doc()doc";

static const char *__doc_Snippet_DynamicParameterContainer_set = R"doc()doc";

static const char *__doc_Snippet_DynamicParameterContainer_updateHash = R"doc()doc";

static const char *__doc_Snippet_Init =
R"doc(Class used to bind together everything required to utilize a snippet
1. A pointer to a variable initialisation snippet
2. The parameters required to control the variable initialisation snippet)doc";

static const char *__doc_Snippet_Init_Init = R"doc()doc";

static const char *__doc_Snippet_Init_finalise = R"doc()doc";

static const char *__doc_Snippet_Init_getDerivedParams = R"doc()doc";

static const char *__doc_Snippet_Init_getHashDigest = R"doc()doc";

static const char *__doc_Snippet_Init_getParams = R"doc()doc";

static const char *__doc_Snippet_Init_getSnippet = R"doc()doc";

static const char *__doc_Snippet_Init_m_DerivedParams = R"doc()doc";

static const char *__doc_Snippet_Init_m_Params = R"doc()doc";

static const char *__doc_Snippet_Init_m_Snippet = R"doc()doc";

static const char *__doc_Snippet_updateHash = R"doc()doc";

static const char *__doc_Snippet_updateHash_2 = R"doc()doc";

static const char *__doc_Snippet_updateHash_3 = R"doc()doc";

static const char *__doc_Snippet_updateHash_4 = R"doc()doc";

static const char *__doc_SynapseGroup = R"doc()doc";

static const char *__doc_SynapseGroup_2 = R"doc()doc";

static const char *__doc_SynapseGroupInternal = R"doc()doc";

static const char *__doc_SynapseGroupInternal_2 = R"doc()doc";

static const char *__doc_SynapseGroupInternal_3 = R"doc()doc";

static const char *__doc_SynapseGroupInternal_4 = R"doc()doc";

static const char *__doc_SynapseGroupInternal_SynapseGroupInternal = R"doc()doc";

static const char *__doc_SynapseGroup_ParallelismHint = R"doc()doc";

static const char *__doc_SynapseGroup_ParallelismHint_POSTSYNAPTIC = R"doc()doc";

static const char *__doc_SynapseGroup_ParallelismHint_PRESYNAPTIC = R"doc()doc";

static const char *__doc_SynapseGroup_ParallelismHint_WORD_PACKED_BITMASK = R"doc()doc";

static const char *__doc_SynapseGroup_SynapseGroup = R"doc()doc";

static const char *__doc_SynapseGroup_SynapseGroup_2 = R"doc()doc";

static const char *__doc_SynapseGroup_SynapseGroup_3 = R"doc()doc";

static const char *__doc_SynapseGroup_addCustomUpdateReference = R"doc(Add reference to custom connectivity update, referencing this synapse group)doc";

static const char *__doc_SynapseGroup_addCustomUpdateReference_2 = R"doc(Add reference to custom update, referencing this synapse group)doc";

static const char *__doc_SynapseGroup_canPSBeFused = R"doc(Can postsynaptic update component of this synapse group be safely fused with others whose hashes match so only one needs simulating at all?)doc";

static const char *__doc_SynapseGroup_canPreOutputBeFused = R"doc(Can presynaptic output component of this synapse group's weight update model be safely fused with other whose hashes match so only one needs simulating at all?)doc";

static const char *__doc_SynapseGroup_canSpikeBeFused = R"doc(Can spike generation for this synapse group be safely fused?)doc";

static const char *__doc_SynapseGroup_canWUMPrePostUpdateBeFused = R"doc(Can presynaptic/postsynaptic update component of this synapse group's weight update model be safely fused with other whose hashes match so only one needs simulating at all?)doc";

static const char *__doc_SynapseGroup_canWUSpikeEventBeFused = R"doc(Can spike event generation for this synapse group be safely fused?)doc";

static const char *__doc_SynapseGroup_finalise = R"doc()doc";

static const char *__doc_SynapseGroup_getAxonalDelaySteps = R"doc()doc";

static const char *__doc_SynapseGroup_getBackPropDelaySteps = R"doc()doc";

static const char *__doc_SynapseGroup_getConnectivityHostInitHashDigest =
R"doc(Generate hash of host connectivity initialisation of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getConnectivityInitHashDigest =
R"doc(Generate hash of connectivity initialisation of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getConnectivityInitialiser = R"doc()doc";

static const char *__doc_SynapseGroup_getCustomConnectivityUpdateReferences =
R"doc(Gets custom connectivity updates which reference this synapse group
Because, if connectivity is sparse, all groups share connectivity this is required if connectivity changes.)doc";

static const char *__doc_SynapseGroup_getCustomUpdateReferences =
R"doc(Gets custom updates which reference this synapse group
Because, if connectivity is sparse, all groups share connectivity this is required if connectivity changes.)doc";

static const char *__doc_SynapseGroup_getDendriticDelayLocation = R"doc(Get variable mode used for this synapse group's dendritic delay buffers)doc";

static const char *__doc_SynapseGroup_getDendriticDelayUpdateHashDigest = R"doc()doc";

static const char *__doc_SynapseGroup_getFusedPSTarget = R"doc()doc";

static const char *__doc_SynapseGroup_getFusedPreOutputTarget = R"doc()doc";

static const char *__doc_SynapseGroup_getFusedSpikeEventTarget = R"doc()doc";

static const char *__doc_SynapseGroup_getFusedSpikeTarget = R"doc()doc";

static const char *__doc_SynapseGroup_getFusedWUPostTarget = R"doc()doc";

static const char *__doc_SynapseGroup_getFusedWUPreTarget = R"doc()doc";

static const char *__doc_SynapseGroup_getKernelSize = R"doc()doc";

static const char *__doc_SynapseGroup_getKernelSizeFlattened = R"doc()doc";

static const char *__doc_SynapseGroup_getMatrixType = R"doc()doc";

static const char *__doc_SynapseGroup_getMaxConnections = R"doc()doc";

static const char *__doc_SynapseGroup_getMaxDendriticDelayTimesteps = R"doc()doc";

static const char *__doc_SynapseGroup_getMaxSourceConnections = R"doc()doc";

static const char *__doc_SynapseGroup_getName = R"doc()doc";

static const char *__doc_SynapseGroup_getNumThreadsPerSpike = R"doc()doc";

static const char *__doc_SynapseGroup_getOutputLocation = R"doc(Get variable mode used for outputs from this synapse group e.g. outPre and outPost)doc";

static const char *__doc_SynapseGroup_getPSExtraGlobalParamLocation = R"doc(Get location of postsynaptic model extra global parameter by name)doc";

static const char *__doc_SynapseGroup_getPSFuseHashDigest =
R"doc(Generate hash of postsynaptic update component of this synapse group with additional components to ensure PSMs
with matching hashes can not only be simulated using the same code, but fused so only one needs simulating at all
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getPSHashDigest =
R"doc(Generate hash of postsynaptic update component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getPSInitHashDigest =
R"doc(Generate hash of postsynaptic model variable initialisation component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getPSInitialiser = R"doc()doc";

static const char *__doc_SynapseGroup_getPSVarLocation = R"doc(Get location of postsynaptic model state variable)doc";

static const char *__doc_SynapseGroup_getParallelismHint = R"doc()doc";

static const char *__doc_SynapseGroup_getPostTargetVar =
R"doc(Get name of neuron input variable postsynaptic model will target
This will either be 'Isyn' or the name of one of the postsynaptic neuron's additional input variables.)doc";

static const char *__doc_SynapseGroup_getPreOutputHashDigest =
R"doc(Generate hash of presynaptic output update component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getPreOutputInitHashDigest =
R"doc(Generate hash of presynaptic output initialization component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getPreTargetVar =
R"doc(Get name of neuron input variable which a presynaptic output specified with $(addToPre) will target
This will either be 'Isyn' or the name of one of the presynaptic neuron's additional input variables.)doc";

static const char *__doc_SynapseGroup_getSparseConnectivityLocation = R"doc(Get variable mode used for sparse connectivity)doc";

static const char *__doc_SynapseGroup_getSparseIndType = R"doc(Get the type to use for sparse connectivity indices for synapse group)doc";

static const char *__doc_SynapseGroup_getSpikeHashDigest =
R"doc(Generate hash of presynaptic or postsynaptic spike generation component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getSrcNeuronGroup = R"doc()doc";

static const char *__doc_SynapseGroup_getSrcNeuronGroup_2 = R"doc()doc";

static const char *__doc_SynapseGroup_getToeplitzConnectivityInitialiser = R"doc()doc";

static const char *__doc_SynapseGroup_getTrgNeuronGroup = R"doc()doc";

static const char *__doc_SynapseGroup_getTrgNeuronGroup_2 = R"doc()doc";

static const char *__doc_SynapseGroup_getVarLocationHashDigest = R"doc()doc";

static const char *__doc_SynapseGroup_getWUExtraGlobalParamLocation = R"doc(Get location of weight update model extra global parameter by name)doc";

static const char *__doc_SynapseGroup_getWUHashDigest =
R"doc(Generate hash of weight update component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getWUInitHashDigest =
R"doc(Generate hash of initialisation component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getWUInitialiser = R"doc()doc";

static const char *__doc_SynapseGroup_getWUPostVarLocation = R"doc(Get location of weight update model postsynaptic state variable by name)doc";

static const char *__doc_SynapseGroup_getWUPreInitHashDigest =
R"doc(Generate hash of presynaptic variable initialisation component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getWUPrePostFuseHashDigest =
R"doc(Generate hash of presynaptic or postsynaptic weight update component of this synapse group with additional components to ensure those
with matching hashes can not only be simulated using the same code, but fused so only one needs simulating at all
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getWUPrePostHashDigest =
R"doc(Generate hash of presynaptic or postsynaptic update component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getWUPrePostInitHashDigest =
R"doc(Generate hash of presynaptic or postsynaptic variable initialisation component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getWUPreVarLocation = R"doc(Get location of weight update model presynaptic state variable by name)doc";

static const char *__doc_SynapseGroup_getWUSpikeEventFuseHashDigest =
R"doc(Generate hash of presynaptic or postsynaptic spike event generation of this synapse group with additional components to ensure PSMs
with matching hashes can not only be simulated using the same code, but fused so only one needs simulating at all
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getWUSpikeEventHashDigest =
R"doc(Generate hash of presynaptic or postsynaptic spike event generation component of this synapse group
NOTE: this can only be called after model is finalized)doc";

static const char *__doc_SynapseGroup_getWUVarLocation = R"doc(Get location of weight update model per-synapse state variable by name)doc";

static const char *__doc_SynapseGroup_isDendriticDelayRequired = R"doc(Does this synapse group require dendritic delay?)doc";

static const char *__doc_SynapseGroup_isPSModelFused = R"doc(Has this synapse group's postsynaptic model been fused with those from other synapse groups?)doc";

static const char *__doc_SynapseGroup_isPSParamDynamic = R"doc(Is postsynaptic model parameter dynamic i.e. it can be changed at runtime)doc";

static const char *__doc_SynapseGroup_isPSVarInitRequired = R"doc(Is var init code required for any variables in this synapse group's postsynaptic update model?)doc";

static const char *__doc_SynapseGroup_isPostSpikeEventFused = R"doc(Has this synapse group's postsynaptic spike event generation been fused with those from other synapse groups?)doc";

static const char *__doc_SynapseGroup_isPostSpikeEventRequired = R"doc(Does synapse group need to handle postsynaptic spike-like events?)doc";

static const char *__doc_SynapseGroup_isPostSpikeEventTimeRequired = R"doc(Are postsynaptic spike-like-event times needed?)doc";

static const char *__doc_SynapseGroup_isPostSpikeFused = R"doc(Has this synapse group's postsynaptic spike generation been fused with those from other synapse groups?)doc";

static const char *__doc_SynapseGroup_isPostSpikeRequired = R"doc(Does synapse group need to handle postsynaptic spikes?)doc";

static const char *__doc_SynapseGroup_isPostSpikeTimeRequired = R"doc(Are postsynaptic spike times needed?)doc";

static const char *__doc_SynapseGroup_isPostTimeReferenced = R"doc(Is the postsynaptic time variable with identifier referenced in weight update model?)doc";

static const char *__doc_SynapseGroup_isPostsynapticOutputRequired = R"doc(Does this synapse group provide postsynaptic output?)doc";

static const char *__doc_SynapseGroup_isPreSpikeEventFused = R"doc(Has this synapse group's presynaptic spike event generation been fused with those from other synapse groups?)doc";

static const char *__doc_SynapseGroup_isPreSpikeEventRequired = R"doc(Does synapse group need to handle presynaptic spike-like events?)doc";

static const char *__doc_SynapseGroup_isPreSpikeEventTimeRequired = R"doc(Are presynaptic spike-like-event times needed?)doc";

static const char *__doc_SynapseGroup_isPreSpikeFused = R"doc(Has this synapse group's presynaptic spike generation been fused with those from other synapse groups?)doc";

static const char *__doc_SynapseGroup_isPreSpikeRequired = R"doc(Does synapse group need to handle 'true' spikes/)doc";

static const char *__doc_SynapseGroup_isPreSpikeTimeRequired = R"doc(Are presynaptic spike times needed?)doc";

static const char *__doc_SynapseGroup_isPreTimeReferenced = R"doc(Is the presynaptic time variable with identifier referenced in weight update model?)doc";

static const char *__doc_SynapseGroup_isPresynapticOutputRequired = R"doc(Does this synapse group provide presynaptic output?)doc";

static const char *__doc_SynapseGroup_isPrevPostSpikeEventTimeRequired = R"doc(Are PREVIOUS postsynaptic spike-event times needed?)doc";

static const char *__doc_SynapseGroup_isPrevPostSpikeTimeRequired = R"doc(Are PREVIOUS postsynaptic spike times needed?)doc";

static const char *__doc_SynapseGroup_isPrevPreSpikeEventTimeRequired = R"doc(Are PREVIOUS presynaptic spike-like-event times needed?)doc";

static const char *__doc_SynapseGroup_isPrevPreSpikeTimeRequired = R"doc(Are PREVIOUS presynaptic spike times needed?)doc";

static const char *__doc_SynapseGroup_isProceduralConnectivityRNGRequired = R"doc(Does this synapse group require an RNG to generate procedural connectivity?)doc";

static const char *__doc_SynapseGroup_isSparseConnectivityInitRequired = R"doc(Is sparse connectivity initialisation code required for this synapse group?)doc";

static const char *__doc_SynapseGroup_isWUInitRNGRequired = R"doc(Does this synapse group require an RNG for it's weight update init code?)doc";

static const char *__doc_SynapseGroup_isWUParamDynamic = R"doc(Is weight update model parameter dynamic i.e. it can be changed at runtime)doc";

static const char *__doc_SynapseGroup_isWUPostModelFused =
R"doc(Has the postsynaptic component of this synapse group's weight update
model been fused with those from other synapse groups?)doc";

static const char *__doc_SynapseGroup_isWUPostVarInitRequired = R"doc(Is var init code required for any presynaptic variables in this synapse group's weight update model?)doc";

static const char *__doc_SynapseGroup_isWUPreModelFused =
R"doc(Has the presynaptic component of this synapse group's weight update
model been fused with those from other synapse groups?)doc";

static const char *__doc_SynapseGroup_isWUPreVarInitRequired = R"doc(Is var init code required for any presynaptic variables in this synapse group's weight update model?)doc";

static const char *__doc_SynapseGroup_isWUVarInitRequired = R"doc(Is var init code required for any variables in this synapse group's weight update model?)doc";

static const char *__doc_SynapseGroup_isZeroCopyEnabled = R"doc()doc";

static const char *__doc_SynapseGroup_m_AxonalDelaySteps = R"doc(Global synaptic conductance delay for the group (in time steps))doc";

static const char *__doc_SynapseGroup_m_BackPropDelaySteps = R"doc(Global backpropagation delay for postsynaptic spikes to synapse (in time)doc";

static const char *__doc_SynapseGroup_m_CustomConnectivityUpdateReferences =
R"doc(Custom connectivity updates which reference this synapse group
Because, if connectivity is sparse, all groups share connectivity this is required if connectivity changes.)doc";

static const char *__doc_SynapseGroup_m_CustomUpdateReferences =
R"doc(Custom updates which reference this synapse group
Because, if connectivity is sparse, all groups share connectivity this is required if connectivity changes.)doc";

static const char *__doc_SynapseGroup_m_DendriticDelayLocation = R"doc(Variable mode used for this synapse group's dendritic delay buffers)doc";

static const char *__doc_SynapseGroup_m_FusedPSTarget =
R"doc(Synapse group postsynaptic model has been fused with
If this is nullptr, postsynaptic model has not been fused)doc";

static const char *__doc_SynapseGroup_m_FusedPostSpikeEventTarget =
R"doc(Synapse group postsynaptic spike event generation has been fused with
If this is nullptr, postsynaptic spike event generation has not been fused)doc";

static const char *__doc_SynapseGroup_m_FusedPostSpikeTarget =
R"doc(Synapse group postsynaptic spike generation has been fused with
If this is nullptr, presynaptic spike generation has not been fused)doc";

static const char *__doc_SynapseGroup_m_FusedPreOutputTarget =
R"doc(Synapse group presynaptic output has been fused with
If this is nullptr, presynaptic output has not been fused)doc";

static const char *__doc_SynapseGroup_m_FusedPreSpikeEventTarget =
R"doc(Synapse group presynaptic spike event generation has been fused with
If this is nullptr, presynaptic spike event generation has not been fused)doc";

static const char *__doc_SynapseGroup_m_FusedPreSpikeTarget =
R"doc(Synapse group presynaptic spike generation has been fused with
If this is nullptr, presynaptic spike generation has not been fused)doc";

static const char *__doc_SynapseGroup_m_FusedWUPostTarget =
R"doc(Synapse group postsynaptic weight update has been fused with
If this is nullptr, postsynaptic weight update  has not been fused)doc";

static const char *__doc_SynapseGroup_m_FusedWUPreTarget =
R"doc(Synapse group presynaptic weight update has been fused with
If this is nullptr, presynaptic weight update has not been fused)doc";

static const char *__doc_SynapseGroup_m_KernelSize = R"doc(Kernel size)doc";

static const char *__doc_SynapseGroup_m_MatrixType = R"doc(Connectivity type of synapses)doc";

static const char *__doc_SynapseGroup_m_MaxConnections = R"doc(Maximum number of target neurons any source neuron can connect to)doc";

static const char *__doc_SynapseGroup_m_MaxDendriticDelayTimesteps = R"doc(Maximum dendritic delay timesteps supported for synapses in this population)doc";

static const char *__doc_SynapseGroup_m_MaxSourceConnections = R"doc(Maximum number of source neurons any target neuron can connect to)doc";

static const char *__doc_SynapseGroup_m_Name = R"doc(Name of the synapse group)doc";

static const char *__doc_SynapseGroup_m_NarrowSparseIndEnabled = R"doc(Should narrow i.e. less than 32-bit types be used for sparse matrix indices)doc";

static const char *__doc_SynapseGroup_m_NumThreadsPerSpike = R"doc(How many threads CUDA implementation uses to process each spike when span type is PRESYNAPTIC)doc";

static const char *__doc_SynapseGroup_m_OutputLocation = R"doc(Variable mode used for outputs from this synapse group e.g. outPre and outPost)doc";

static const char *__doc_SynapseGroup_m_PSDynamicParams = R"doc(Data structure tracking whether postsynaptic model parameters are dynamic or not)doc";

static const char *__doc_SynapseGroup_m_PSExtraGlobalParamLocation = R"doc(Location of postsynaptic model extra global parameters)doc";

static const char *__doc_SynapseGroup_m_PSInitialiser = R"doc(Initialiser used for creating postsynaptic update model)doc";

static const char *__doc_SynapseGroup_m_PSVarLocation = R"doc(Whether indidividual state variables of post synapse should use zero-copied memory)doc";

static const char *__doc_SynapseGroup_m_ParallelismHint = R"doc(Hint as to how synapse group should be parallelised)doc";

static const char *__doc_SynapseGroup_m_PostTargetVar =
R"doc(Name of neuron input variable postsynaptic model will target
This should either be 'Isyn' or the name of one of the postsynaptic neuron's additional input variables.)doc";

static const char *__doc_SynapseGroup_m_PreTargetVar =
R"doc(Name of neuron input variable a presynaptic output specified with $(addToPre) will target
This will either be 'Isyn' or the name of one of the presynaptic neuron's additional input variables.)doc";

static const char *__doc_SynapseGroup_m_SparseConnectivityInitialiser = R"doc(Initialiser used for creating sparse connectivity)doc";

static const char *__doc_SynapseGroup_m_SparseConnectivityLocation = R"doc(Location of sparse connectivity)doc";

static const char *__doc_SynapseGroup_m_SrcNeuronGroup = R"doc(Pointer to presynaptic neuron group)doc";

static const char *__doc_SynapseGroup_m_ToeplitzConnectivityInitialiser = R"doc(Initialiser used for creating toeplitz connectivity)doc";

static const char *__doc_SynapseGroup_m_TrgNeuronGroup = R"doc(Pointer to postsynaptic neuron group)doc";

static const char *__doc_SynapseGroup_m_WUDynamicParams = R"doc(Data structure tracking whether weight update model parameters are dynamic or not)doc";

static const char *__doc_SynapseGroup_m_WUExtraGlobalParamLocation = R"doc(Location of weight update model extra global parameters)doc";

static const char *__doc_SynapseGroup_m_WUInitialiser = R"doc(Initialiser used for creating weight update model)doc";

static const char *__doc_SynapseGroup_m_WUPostVarLocation = R"doc(Location of individual postsynaptic state variables)doc";

static const char *__doc_SynapseGroup_m_WUPreVarLocation = R"doc(Location of individual presynaptic state variables)doc";

static const char *__doc_SynapseGroup_m_WUVarLocation = R"doc(Location of individual per-synapse state variables)doc";

static const char *__doc_SynapseGroup_setAxonalDelaySteps = R"doc(Sets the number of delay steps used to delay events and variables between presynaptic neuron and synapse)doc";

static const char *__doc_SynapseGroup_setBackPropDelaySteps = R"doc(Sets the number of delay steps used to delay events and variables between postsynaptic neuron and synapse)doc";

static const char *__doc_SynapseGroup_setDendriticDelayLocation = R"doc(Set variable mode used for this synapse group's dendritic delay buffers)doc";

static const char *__doc_SynapseGroup_setFusedPSTarget = R"doc()doc";

static const char *__doc_SynapseGroup_setFusedPreOutputTarget = R"doc()doc";

static const char *__doc_SynapseGroup_setFusedSpikeEventTarget = R"doc()doc";

static const char *__doc_SynapseGroup_setFusedSpikeTarget = R"doc()doc";

static const char *__doc_SynapseGroup_setFusedWUPrePostTarget = R"doc()doc";

static const char *__doc_SynapseGroup_setMaxConnections =
R"doc(Sets the maximum number of target neurons any source neurons can connect to
Use with synaptic matrix types with SynapseMatrixConnectivity::SPARSE to optimise CUDA implementation)doc";

static const char *__doc_SynapseGroup_setMaxDendriticDelayTimesteps = R"doc(Sets the maximum dendritic delay for synapses in this synapse group)doc";

static const char *__doc_SynapseGroup_setMaxSourceConnections =
R"doc(Sets the maximum number of source neurons any target neuron can connect to
Use with synaptic matrix types with SynapseMatrixConnectivity::SPARSE and postsynaptic learning to optimise CUDA implementation)doc";

static const char *__doc_SynapseGroup_setNarrowSparseIndEnabled = R"doc(Enables or disables using narrow i.e. less than 32-bit types for sparse matrix indices)doc";

static const char *__doc_SynapseGroup_setNumThreadsPerSpike = R"doc(Provide hint as to how many threads SIMT backend might use to process each spike if PRESYNAPTIC parallelism is selected)doc";

static const char *__doc_SynapseGroup_setOutputLocation =
R"doc(Set location of variables used for outputs from this synapse group e.g. outPre and outPost
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_SynapseGroup_setPSExtraGlobalParamLocation =
R"doc(Set location of postsynaptic model extra global parameter
This is ignored for simulations on hardware with a single memory space.)doc";

static const char *__doc_SynapseGroup_setPSParamDynamic = R"doc(Set whether weight update model parameter is dynamic or not i.e. it can be changed at runtime)doc";

static const char *__doc_SynapseGroup_setPSVarLocation =
R"doc(Set location of postsynaptic model state variable
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_SynapseGroup_setParallelismHint = R"doc(Provide a hint as to how this synapse group should be parallelised)doc";

static const char *__doc_SynapseGroup_setPostTargetVar =
R"doc(Set name of neuron input variable postsynaptic model will target
This should either be 'Isyn' or the name of one of the postsynaptic neuron's additional input variables.)doc";

static const char *__doc_SynapseGroup_setPreTargetVar =
R"doc(Set name of neuron input variable $(addToPre, . ) commands will target
This should either be 'Isyn' or the name of one of the presynaptic neuron's additional input variables.)doc";

static const char *__doc_SynapseGroup_setSparseConnectivityLocation =
R"doc(Set variable mode used for sparse connectivity
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_SynapseGroup_setWUExtraGlobalParamLocation =
R"doc(Set location of weight update model extra global parameter
This is ignored for simulations on hardware with a single memory space.)doc";

static const char *__doc_SynapseGroup_setWUParamDynamic = R"doc(Set whether weight update model parameter is dynamic or not i.e. it can be changed at runtime)doc";

static const char *__doc_SynapseGroup_setWUPostVarLocation =
R"doc(Set location of weight update model postsynaptic state variable
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_SynapseGroup_setWUPreVarLocation =
R"doc(Set location of weight update model presynaptic state variable
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_SynapseGroup_setWUVarLocation =
R"doc(Set location of weight update model state variable
This is ignored for simulations on hardware with a single memory space)doc";

static const char *__doc_SynapseMatrixConnectivity = R"doc(Flags defining differnet types of synaptic matrix connectivity)doc";

static const char *__doc_SynapseMatrixConnectivity_BITMASK = R"doc()doc";

static const char *__doc_SynapseMatrixConnectivity_DENSE = R"doc()doc";

static const char *__doc_SynapseMatrixConnectivity_PROCEDURAL = R"doc()doc";

static const char *__doc_SynapseMatrixConnectivity_SPARSE = R"doc()doc";

static const char *__doc_SynapseMatrixConnectivity_TOEPLITZ = R"doc()doc";

static const char *__doc_SynapseMatrixType = R"doc(Supported combinations of SynapticMatrixConnectivity and SynapticMatrixWeight)doc";

static const char *__doc_SynapseMatrixType_BITMASK = R"doc()doc";

static const char *__doc_SynapseMatrixType_DENSE = R"doc()doc";

static const char *__doc_SynapseMatrixType_DENSE_PROCEDURALG = R"doc()doc";

static const char *__doc_SynapseMatrixType_PROCEDURAL = R"doc()doc";

static const char *__doc_SynapseMatrixType_PROCEDURAL_KERNELG = R"doc()doc";

static const char *__doc_SynapseMatrixType_SPARSE = R"doc()doc";

static const char *__doc_SynapseMatrixType_TOEPLITZ = R"doc()doc";

static const char *__doc_SynapseMatrixWeight = R"doc(Flags defining different types of synaptic matrix connectivity)doc";

static const char *__doc_SynapseMatrixWeight_INDIVIDUAL = R"doc()doc";

static const char *__doc_SynapseMatrixWeight_KERNEL = R"doc()doc";

static const char *__doc_SynapseMatrixWeight_PROCEDURAL = R"doc()doc";

static const char *__doc_SynapsePSMEGPAdapter = R"doc()doc";

static const char *__doc_SynapsePSMEGPAdapter_SynapsePSMEGPAdapter = R"doc()doc";

static const char *__doc_SynapsePSMEGPAdapter_getDefs = R"doc()doc";

static const char *__doc_SynapsePSMEGPAdapter_getLoc = R"doc()doc";

static const char *__doc_SynapsePSMEGPAdapter_m_SG = R"doc()doc";

static const char *__doc_SynapsePSMNeuronVarRefAdapter = R"doc()doc";

static const char *__doc_SynapsePSMNeuronVarRefAdapter_SynapsePSMNeuronVarRefAdapter = R"doc()doc";

static const char *__doc_SynapsePSMNeuronVarRefAdapter_getDefs = R"doc()doc";

static const char *__doc_SynapsePSMNeuronVarRefAdapter_getInitialisers = R"doc()doc";

static const char *__doc_SynapsePSMNeuronVarRefAdapter_m_SG = R"doc()doc";

static const char *__doc_SynapsePSMVarAdapter = R"doc()doc";

static const char *__doc_SynapsePSMVarAdapter_SynapsePSMVarAdapter = R"doc()doc";

static const char *__doc_SynapsePSMVarAdapter_getDefs = R"doc()doc";

static const char *__doc_SynapsePSMVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_SynapsePSMVarAdapter_getLoc = R"doc()doc";

static const char *__doc_SynapsePSMVarAdapter_getTarget = R"doc()doc";

static const char *__doc_SynapsePSMVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_SynapsePSMVarAdapter_isVarDelayed = R"doc()doc";

static const char *__doc_SynapsePSMVarAdapter_m_SG = R"doc()doc";

static const char *__doc_SynapseWUEGPAdapter = R"doc()doc";

static const char *__doc_SynapseWUEGPAdapter_SynapseWUEGPAdapter = R"doc()doc";

static const char *__doc_SynapseWUEGPAdapter_getDefs = R"doc()doc";

static const char *__doc_SynapseWUEGPAdapter_getLoc = R"doc()doc";

static const char *__doc_SynapseWUEGPAdapter_m_SG = R"doc()doc";

static const char *__doc_SynapseWUPostNeuronVarRefAdapter = R"doc()doc";

static const char *__doc_SynapseWUPostNeuronVarRefAdapter_SynapseWUPostNeuronVarRefAdapter = R"doc()doc";

static const char *__doc_SynapseWUPostNeuronVarRefAdapter_getDefs = R"doc()doc";

static const char *__doc_SynapseWUPostNeuronVarRefAdapter_getInitialisers = R"doc()doc";

static const char *__doc_SynapseWUPostNeuronVarRefAdapter_m_SG = R"doc()doc";

static const char *__doc_SynapseWUPostVarAdapter = R"doc()doc";

static const char *__doc_SynapseWUPostVarAdapter_SynapseWUPostVarAdapter = R"doc()doc";

static const char *__doc_SynapseWUPostVarAdapter_getDefs = R"doc()doc";

static const char *__doc_SynapseWUPostVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_SynapseWUPostVarAdapter_getLoc = R"doc()doc";

static const char *__doc_SynapseWUPostVarAdapter_getTarget = R"doc()doc";

static const char *__doc_SynapseWUPostVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_SynapseWUPostVarAdapter_isVarDelayed = R"doc()doc";

static const char *__doc_SynapseWUPostVarAdapter_m_SG = R"doc()doc";

static const char *__doc_SynapseWUPreNeuronVarRefAdapter = R"doc()doc";

static const char *__doc_SynapseWUPreNeuronVarRefAdapter_SynapseWUPreNeuronVarRefAdapter = R"doc()doc";

static const char *__doc_SynapseWUPreNeuronVarRefAdapter_getDefs = R"doc()doc";

static const char *__doc_SynapseWUPreNeuronVarRefAdapter_getInitialisers = R"doc()doc";

static const char *__doc_SynapseWUPreNeuronVarRefAdapter_m_SG = R"doc()doc";

static const char *__doc_SynapseWUPreVarAdapter = R"doc()doc";

static const char *__doc_SynapseWUPreVarAdapter_SynapseWUPreVarAdapter = R"doc()doc";

static const char *__doc_SynapseWUPreVarAdapter_getDefs = R"doc()doc";

static const char *__doc_SynapseWUPreVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_SynapseWUPreVarAdapter_getLoc = R"doc()doc";

static const char *__doc_SynapseWUPreVarAdapter_getTarget = R"doc()doc";

static const char *__doc_SynapseWUPreVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_SynapseWUPreVarAdapter_isVarDelayed = R"doc()doc";

static const char *__doc_SynapseWUPreVarAdapter_m_SG = R"doc()doc";

static const char *__doc_SynapseWUVarAdapter = R"doc()doc";

static const char *__doc_SynapseWUVarAdapter_SynapseWUVarAdapter = R"doc()doc";

static const char *__doc_SynapseWUVarAdapter_getDefs = R"doc()doc";

static const char *__doc_SynapseWUVarAdapter_getInitialisers = R"doc()doc";

static const char *__doc_SynapseWUVarAdapter_getLoc = R"doc()doc";

static const char *__doc_SynapseWUVarAdapter_getTarget = R"doc()doc";

static const char *__doc_SynapseWUVarAdapter_getVarDims = R"doc()doc";

static const char *__doc_SynapseWUVarAdapter_m_SG = R"doc()doc";

static const char *__doc_Type_NumericValue =
R"doc(ResolvedType::Numeric has various values attached e.g. min and max. These
Cannot be represented using any single type (double can't represent all uint64_t for example)
Therefore, this type is used as a wrapper.)doc";

static const char *__doc_Type_NumericValue_NumericValue = R"doc()doc";

static const char *__doc_Type_NumericValue_NumericValue_2 = R"doc()doc";

static const char *__doc_Type_NumericValue_NumericValue_3 = R"doc()doc";

static const char *__doc_Type_NumericValue_NumericValue_4 = R"doc()doc";

static const char *__doc_Type_NumericValue_NumericValue_5 = R"doc()doc";

static const char *__doc_Type_NumericValue_cast = R"doc()doc";

static const char *__doc_Type_NumericValue_get = R"doc()doc";

static const char *__doc_Type_NumericValue_m_Value = R"doc()doc";

static const char *__doc_Type_NumericValue_operator_eq = R"doc()doc";

static const char *__doc_Type_NumericValue_operator_ge = R"doc()doc";

static const char *__doc_Type_NumericValue_operator_gt = R"doc()doc";

static const char *__doc_Type_NumericValue_operator_le = R"doc()doc";

static const char *__doc_Type_NumericValue_operator_lt = R"doc()doc";

static const char *__doc_Type_NumericValue_operator_ne = R"doc()doc";

static const char *__doc_Type_Qualifier = R"doc()doc";

static const char *__doc_Type_Qualifier_CONSTANT = R"doc()doc";

static const char *__doc_Type_ResolvedType = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function_Function = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function_Function_2 = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function_argTypes = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function_operator_assign = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function_operator_eq = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function_operator_lt = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function_operator_ne = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function_returnType = R"doc()doc";

static const char *__doc_Type_ResolvedType_Function_variadic = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_isIntegral = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_isSigned = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_literalSuffix = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_lowest = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_max = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_maxDigits10 = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_min = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_operator_eq = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_operator_lt = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_operator_ne = R"doc()doc";

static const char *__doc_Type_ResolvedType_Numeric_rank = R"doc()doc";

static const char *__doc_Type_ResolvedType_Pointer = R"doc()doc";

static const char *__doc_Type_ResolvedType_Pointer_Pointer = R"doc()doc";

static const char *__doc_Type_ResolvedType_Pointer_Pointer_2 = R"doc()doc";

static const char *__doc_Type_ResolvedType_Pointer_operator_assign = R"doc()doc";

static const char *__doc_Type_ResolvedType_Pointer_operator_eq = R"doc()doc";

static const char *__doc_Type_ResolvedType_Pointer_operator_lt = R"doc()doc";

static const char *__doc_Type_ResolvedType_Pointer_operator_ne = R"doc()doc";

static const char *__doc_Type_ResolvedType_Pointer_valueType = R"doc()doc";

static const char *__doc_Type_ResolvedType_ResolvedType = R"doc()doc";

static const char *__doc_Type_ResolvedType_ResolvedType_2 = R"doc()doc";

static const char *__doc_Type_ResolvedType_ResolvedType_3 = R"doc()doc";

static const char *__doc_Type_ResolvedType_ResolvedType_4 = R"doc()doc";

static const char *__doc_Type_ResolvedType_ResolvedType_5 = R"doc()doc";

static const char *__doc_Type_ResolvedType_Value = R"doc()doc";

static const char *__doc_Type_ResolvedType_Value_device = R"doc()doc";

static const char *__doc_Type_ResolvedType_Value_ffiType = R"doc()doc";

static const char *__doc_Type_ResolvedType_Value_name = R"doc()doc";

static const char *__doc_Type_ResolvedType_Value_numeric = R"doc()doc";

static const char *__doc_Type_ResolvedType_Value_operator_eq = R"doc()doc";

static const char *__doc_Type_ResolvedType_Value_operator_lt = R"doc()doc";

static const char *__doc_Type_ResolvedType_Value_operator_ne = R"doc()doc";

static const char *__doc_Type_ResolvedType_Value_size = R"doc()doc";

static const char *__doc_Type_ResolvedType_addConst = R"doc()doc";

static const char *__doc_Type_ResolvedType_addQualifier = R"doc()doc";

static const char *__doc_Type_ResolvedType_createFunction = R"doc()doc";

static const char *__doc_Type_ResolvedType_createNumeric = R"doc()doc";

static const char *__doc_Type_ResolvedType_createPointer = R"doc()doc";

static const char *__doc_Type_ResolvedType_createValue = R"doc()doc";

static const char *__doc_Type_ResolvedType_detail = R"doc()doc";

static const char *__doc_Type_ResolvedType_getFFIType = R"doc()doc";

static const char *__doc_Type_ResolvedType_getFunction = R"doc()doc";

static const char *__doc_Type_ResolvedType_getName = R"doc()doc";

static const char *__doc_Type_ResolvedType_getNumeric = R"doc()doc";

static const char *__doc_Type_ResolvedType_getPointer = R"doc()doc";

static const char *__doc_Type_ResolvedType_getSize = R"doc()doc";

static const char *__doc_Type_ResolvedType_getValue = R"doc()doc";

static const char *__doc_Type_ResolvedType_hasQualifier = R"doc()doc";

static const char *__doc_Type_ResolvedType_isFunction = R"doc()doc";

static const char *__doc_Type_ResolvedType_isNumeric = R"doc()doc";

static const char *__doc_Type_ResolvedType_isPointer = R"doc()doc";

static const char *__doc_Type_ResolvedType_isPointerToPointer = R"doc()doc";

static const char *__doc_Type_ResolvedType_isValue = R"doc()doc";

static const char *__doc_Type_ResolvedType_isVoid = R"doc()doc";

static const char *__doc_Type_ResolvedType_operator_eq = R"doc()doc";

static const char *__doc_Type_ResolvedType_operator_lt = R"doc()doc";

static const char *__doc_Type_ResolvedType_operator_ne = R"doc()doc";

static const char *__doc_Type_ResolvedType_qualifiers = R"doc()doc";

static const char *__doc_Type_ResolvedType_removeQualifiers = R"doc()doc";

static const char *__doc_Type_UnresolvedType = R"doc()doc";

static const char *__doc_Type_UnresolvedType_UnresolvedType = R"doc()doc";

static const char *__doc_Type_UnresolvedType_UnresolvedType_2 = R"doc()doc";

static const char *__doc_Type_UnresolvedType_detail = R"doc()doc";

static const char *__doc_Type_UnresolvedType_operator_eq = R"doc()doc";

static const char *__doc_Type_UnresolvedType_operator_lt = R"doc()doc";

static const char *__doc_Type_UnresolvedType_operator_ne = R"doc()doc";

static const char *__doc_Type_UnresolvedType_resolve = R"doc()doc";

static const char *__doc_Type_getCommonType = R"doc(Apply C rules to get common type between numeric types a and b)doc";

static const char *__doc_Type_getPromotedType = R"doc(Apply C type promotion rules to numeric type)doc";

static const char *__doc_Type_operator_band = R"doc()doc";

static const char *__doc_Type_operator_bor = R"doc()doc";

static const char *__doc_Type_serialiseNumeric = R"doc(Serialise numeric value to bytes)doc";

static const char *__doc_Type_updateHash = R"doc()doc";

static const char *__doc_Type_updateHash_2 = R"doc()doc";

static const char *__doc_Type_updateHash_3 = R"doc()doc";

static const char *__doc_Type_updateHash_4 = R"doc()doc";

static const char *__doc_Type_updateHash_5 = R"doc()doc";

static const char *__doc_Type_updateHash_6 = R"doc()doc";

static const char *__doc_Type_updateHash_7 = R"doc()doc";

static const char *__doc_Type_writeNumeric = R"doc(Write numeric value to string, formatting correctly for type)doc";

static const char *__doc_Utils = R"doc()doc";

static const char *__doc_Utils_Overload = R"doc(Boilerplate for overloading base std::visit)doc";

static const char *__doc_Utils_SHA1Hash = R"doc(Functor for generating a hash suitable for use in std::unordered_map etc (i.e. size_t size) from a SHA1 digests)doc";

static const char *__doc_Utils_SHA1Hash_operator_call = R"doc()doc";

static const char *__doc_Utils_areTokensEmpty =
R"doc(Is this sequence of tokens empty?
For ease of parsing and as an extra check that we have scanned SOMETHING,
empty token sequences should have a single EOF token)doc";

static const char *__doc_Utils_clz = R"doc(Count leading zeros)doc";

static const char *__doc_Utils_handleLegacyEGPType =
R"doc(Extra global parameters used to support both pointer and non-pointer types. Now only the behaviour that used to
be provided by pointer types is provided but, internally, non-pointer types are used. This handles pointer types specified by string.)doc";

static const char *__doc_Utils_isIdentifierReferenced = R"doc(Checks whether the sequence of token references a given identifier)doc";

static const char *__doc_Utils_isRNGRequired = R"doc(Checks whether the sequence of token includes an RNG function identifier)doc";

static const char *__doc_Utils_isRNGRequired_2 = R"doc(Checks whether any of the variable initialisers in the vector require an RNG for initialisation)doc";

static const char *__doc_Utils_parseNumericType = R"doc(Helper to scan a type specifier string e.g "unsigned int" and parse it into a resolved type)doc";

static const char *__doc_Utils_scanCode = R"doc(Helper to scan a multi-line code string, giving meaningful errors with the specified context string)doc";

static const char *__doc_Utils_updateHash = R"doc(Hash arithmetic types and enums)doc";

static const char *__doc_Utils_updateHash_2 = R"doc(Hash monostate)doc";

static const char *__doc_Utils_updateHash_3 = R"doc(Hash strings)doc";

static const char *__doc_Utils_updateHash_4 = R"doc(Hash arrays of types which can, themselves, be hashed)doc";

static const char *__doc_Utils_updateHash_5 = R"doc(Hash vectors of types which can, themselves, be hashed)doc";

static const char *__doc_Utils_updateHash_6 = R"doc(Hash vectors of bools)doc";

static const char *__doc_Utils_updateHash_7 = R"doc(Hash unordered maps of types which can, themselves, be hashed)doc";

static const char *__doc_Utils_updateHash_8 = R"doc(Hash unordered sets of types which can, themselves, be hashed)doc";

static const char *__doc_Utils_updateHash_9 = R"doc(Hash optional types which can, themeselves, be hashed)doc";

static const char *__doc_Utils_updateHash_10 = R"doc(Hash variants of types which can, themeselves, be hashed)doc";

static const char *__doc_Utils_validateInitialisers = R"doc(Checks that initialisers provided for all of the the item names in the vector?)doc";

static const char *__doc_Utils_validatePopName = R"doc(Checks whether population name is valid? GeNN population names obey C variable naming rules but can start with a number)doc";

static const char *__doc_Utils_validateVarName = R"doc(Checks variable name is valid? GeNN variable names must obey C variable naming rules)doc";

static const char *__doc_Utils_validateVecNames = R"doc(Checks whether the 'name' fields of all structs in vector valid? GeNN variables and population names must obey C variable naming rules)doc";

static const char *__doc_VarAccess = R"doc(Supported combinations of access mode and dimension for neuron and synapse variables)doc";

static const char *__doc_VarAccessDim = R"doc(Flags defining dimensions this variables has)doc";

static const char *__doc_VarAccessDim_BATCH = R"doc()doc";

static const char *__doc_VarAccessDim_ELEMENT = R"doc()doc";

static const char *__doc_VarAccessMode = R"doc(Supported combination of VarAccessModeAttribute)doc";

static const char *__doc_VarAccessModeAttribute =
R"doc(Flags defining attributes of var access models
**NOTE** Read-only and read-write are seperate flags rather than read and write so you can test mode & VarAccessMode::READ_ONLY)doc";

static const char *__doc_VarAccessModeAttribute_MAX = R"doc(This variable's reduction operation is a summation)doc";

static const char *__doc_VarAccessModeAttribute_READ_ONLY = R"doc()doc";

static const char *__doc_VarAccessModeAttribute_READ_WRITE = R"doc(This variable is read only)doc";

static const char *__doc_VarAccessModeAttribute_REDUCE = R"doc(This variable is read-write)doc";

static const char *__doc_VarAccessModeAttribute_SUM = R"doc(This variable is a reduction target)doc";

static const char *__doc_VarAccessMode_READ_ONLY = R"doc()doc";

static const char *__doc_VarAccessMode_READ_WRITE = R"doc()doc";

static const char *__doc_VarAccessMode_REDUCE_MAX = R"doc()doc";

static const char *__doc_VarAccessMode_REDUCE_SUM = R"doc()doc";

static const char *__doc_VarAccess_READ_ONLY = R"doc()doc";

static const char *__doc_VarAccess_READ_ONLY_DUPLICATE = R"doc()doc";

static const char *__doc_VarAccess_READ_ONLY_SHARED_NEURON = R"doc()doc";

static const char *__doc_VarAccess_READ_WRITE = R"doc()doc";

static const char *__doc_VarLocation = R"doc()doc";

static const char *__doc_VarLocation_DEVICE = R"doc()doc";

static const char *__doc_VarLocation_HOST = R"doc()doc";

static const char *__doc_VarLocation_HOST_DEVICE = R"doc()doc";

static const char *__doc_VarLocation_HOST_DEVICE_ZERO_COPY = R"doc()doc";

static const char *__doc_VarLocation_ZERO_COPY = R"doc()doc";

static const char *__doc_WeightUpdateModels_Base = R"doc(Base class for all weight update models)doc";

static const char *__doc_WeightUpdateModels_Base_getHashDigest = R"doc(Update hash from model)doc";

static const char *__doc_WeightUpdateModels_Base_getPostDynamicsCode =
R"doc(Gets code to be run after postsynaptic neuron update
This is typically for the code to update postsynaptic variables. Presynaptic
and synapse variables are not accesible from within this code)doc";

static const char *__doc_WeightUpdateModels_Base_getPostEventHashDigest = R"doc(Update hash from postsynaptic event-triggering components of model)doc";

static const char *__doc_WeightUpdateModels_Base_getPostEventSynCode =
R"doc(Gets code run when a postsynaptic spike-like event is received at the synapse
Postsynaptic events are triggered for all postsynaptic neurons where
the postsynaptic event threshold condition is met)doc";

static const char *__doc_WeightUpdateModels_Base_getPostEventThresholdConditionCode = R"doc(Gets codes to test for postsynaptic events)doc";

static const char *__doc_WeightUpdateModels_Base_getPostHashDigest = R"doc(Update hash from postsynaptic components of  model)doc";

static const char *__doc_WeightUpdateModels_Base_getPostNeuronVarRefs = R"doc(Gets names and types of variable references to postsynaptic neuron)doc";

static const char *__doc_WeightUpdateModels_Base_getPostSpikeCode =
R"doc(Gets code to be run once per spiking postsynaptic neuron before learn post code is run on synapses
This is typically for the code to update postsynaptic variables. Presynaptic
and synapse variables are not accesible from within this code)doc";

static const char *__doc_WeightUpdateModels_Base_getPostSpikeSynCode =
R"doc(Gets code run when a postsynaptic spike is received at the synapse
For examples when modelling STDP, this is where the effect of postsynaptic
spikes which occur _after_ presynaptic spikes are applied.)doc";

static const char *__doc_WeightUpdateModels_Base_getPostVar = R"doc(Find the named postsynaptic variable)doc";

static const char *__doc_WeightUpdateModels_Base_getPostVars =
R"doc(Gets names and types (as strings) of state variables that are common
across all synapses going to the same postsynaptic neuron)doc";

static const char *__doc_WeightUpdateModels_Base_getPreDynamicsCode =
R"doc(Gets code to be run after presynaptic neuron update
This is typically for the code to update presynaptic variables. Postsynaptic
and synapse variables are not accesible from within this code)doc";

static const char *__doc_WeightUpdateModels_Base_getPreEventHashDigest = R"doc(Update hash from presynaptic event-triggering components of model)doc";

static const char *__doc_WeightUpdateModels_Base_getPreEventSynCode =
R"doc(Gets code run when a presynaptic spike-like event is received at the synapse
Presynaptic events are triggered for all presynaptic neurons where
the presynaptic event threshold condition is met)doc";

static const char *__doc_WeightUpdateModels_Base_getPreEventThresholdConditionCode = R"doc(Gets codes to test for presynaptic events)doc";

static const char *__doc_WeightUpdateModels_Base_getPreHashDigest = R"doc(Update hash from presynaptic components of model)doc";

static const char *__doc_WeightUpdateModels_Base_getPreNeuronVarRefs = R"doc(Gets names and types of variable references to presynaptic neuron)doc";

static const char *__doc_WeightUpdateModels_Base_getPreSpikeCode =
R"doc(Gets code to be run once per spiking presynaptic neuron before sim code is run on synapses
This is typically for the code to update presynaptic variables. Postsynaptic
and synapse variables are not accesible from within this code)doc";

static const char *__doc_WeightUpdateModels_Base_getPreSpikeSynCode = R"doc(Gets code run when a presynaptic spike is received at the synapse)doc";

static const char *__doc_WeightUpdateModels_Base_getPreVar = R"doc(Find the named presynaptic variable)doc";

static const char *__doc_WeightUpdateModels_Base_getPreVars =
R"doc(Gets names and types (as strings) of state variables that are common
across all synapses coming from the same presynaptic neuron)doc";

static const char *__doc_WeightUpdateModels_Base_getSynapseDynamicsCode = R"doc(Gets code for synapse dynamics which are independent of spike detection)doc";

static const char *__doc_WeightUpdateModels_Base_getVar = R"doc(Find the named variable)doc";

static const char *__doc_WeightUpdateModels_Base_getVars = R"doc(Gets model variables)doc";

static const char *__doc_WeightUpdateModels_Base_validate = R"doc(Validate names of parameters etc)doc";

static const char *__doc_WeightUpdateModels_Init = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_Init = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_finalise = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPostDynamicsCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPostEventSynCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPostEventThresholdCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPostNeuronVarReferences = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPostSpikeCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPostSpikeSynCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPostVarInitialisers = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPreDynamicsCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPreEventSynCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPreEventThresholdCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPreNeuronVarReferences = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPreSpikeCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPreSpikeSynCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getPreVarInitialisers = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getSynapseDynamicsCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_getVarInitialisers = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_isRNGRequired = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PostDynamicsCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PostEventSynCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PostEventThresholdCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PostNeuronVarReferences = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PostSpikeCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PostSpikeSynCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PostVarInitialisers = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PreDynamicsCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PreEventSynCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PreEventThresholdCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PreNeuronVarReferences = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PreSpikeCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PreSpikeSynCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_PreVarInitialisers = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_SynapseDynamicsCodeTokens = R"doc()doc";

static const char *__doc_WeightUpdateModels_Init_m_VarInitialisers = R"doc()doc";

static const char *__doc_WeightUpdateModels_PiecewiseSTDP =
R"doc(This is a simple STDP rule including a time delay for the finite transmission speed of the synapse.
The STDP window is defined as a piecewise function:
\image html LEARN1SYNAPSE_explain_html.png
\image latex LEARN1SYNAPSE_explain.png width=10cm

The STDP curve is applied to the raw synaptic conductance `gRaw`, which is then filtered through the sugmoidal filter displayed above to obtain the value of `g`.

\note
The STDP curve implies that unpaired pre- and post-synaptic spikes incur a negative increment in `gRaw` (and hence in `g`).

\note
The time of the last spike in each neuron, "sTXX", where XX is the name of a neuron population is (somewhat arbitrarily) initialised to -10.0 ms. If neurons never spike, these spike times are used.

\note
It is the raw synaptic conductance `gRaw` that is subject to the STDP rule. The resulting synaptic conductance is a sigmoid filter of `gRaw`. This implies that `g` is initialised but not `gRaw`, the synapse will revert to the value that corresponds to `gRaw`.

An example how to use this synapse correctly is given in `map_classol.cc` (MBody1 userproject):
```
for (int i= 0; i < model.neuronN[1]*model.neuronN[3]; i++) {
if (gKCDN[i] < 2.0*SCALAR_MIN){
cnt++;
fprintf(stdout, "Too low conductance value %e detected and set to 2*SCALAR_MIN= %e, at index %d \n", gKCDN[i], 2*SCALAR_MIN, i);
gKCDN[i] = 2.0*SCALAR_MIN; //to avoid log(0)/0 below
}
scalar tmp = gKCDN[i] / myKCDN_p[5]*2.0 ;
gRawKCDN[i]=  0.5 * log( tmp / (2.0 - tmp)) /myKCDN_p[7] + myKCDN_p[6];
}
cerr << "Total number of low value corrections: " << cnt << endl;
```


\note
One cannot set values of `g` fully to `0`, as this leads to `gRaw`= -infinity and this is not support. I.e., 'g' needs to be some nominal value > 0 (but can be extremely small so that it acts like it's 0).

<!--
If no spikes at t: :math:` g_{raw}(t+dt) = g_0 + (g_{raw}(t)-g_0)*\exp(-dt/\tau_{decay}) `
If pre or postsynaptic spike at t: :math:` g_{raw}(t+dt) = g_0 + (g_{raw}(t)-g_0)*\exp(-dt/\tau_{decay})
+A(t_{post}-t_{pre}-\tau_{decay}) `
-->

The model has 2 variables:
- ``g:`` conductance of ``scalar`` type
- ``gRaw:`` raw conductance of ``scalar`` type

Parameters are (compare to the figure above):
- ``tLrn:`` Time scale of learning changes
- ``tChng:`` Width of learning window
- ``tDecay:`` Time scale of synaptic strength decay
- ``tPunish10:`` Time window of suppression in response to 1/0
- ``tPunish01:`` Time window of suppression in response to 0/1
- ``gMax:`` Maximal conductance achievable
- ``gMid:`` Midpoint of sigmoid g filter curve
- ``gSlope:`` Slope of sigmoid g filter curve
- ``tauShift:`` Shift of learning curve
- ``gSyn0:`` Value of syn conductance g decays to)doc";

static const char *__doc_WeightUpdateModels_PiecewiseSTDP_getDerivedParams = R"doc()doc";

static const char *__doc_WeightUpdateModels_PiecewiseSTDP_getInstance = R"doc()doc";

static const char *__doc_WeightUpdateModels_PiecewiseSTDP_getParams = R"doc()doc";

static const char *__doc_WeightUpdateModels_PiecewiseSTDP_getPostSpikeSynCode = R"doc()doc";

static const char *__doc_WeightUpdateModels_PiecewiseSTDP_getPreSpikeSynCode = R"doc()doc";

static const char *__doc_WeightUpdateModels_PiecewiseSTDP_getVars = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticGraded =
R"doc(Graded-potential, static synapse
In a graded synapse, the conductance is updated gradually with the rule:
\f[ gSyn= g * tanh((V - E_{pre}) / V_{slope} \f]
whenever the membrane potential :math:`V` is larger than the threshold :math:`E_{pre}`.
The model has 1 variable:
- ``g:`` conductance of ``scalar`` type

The parameters are:
- ``Epre:`` Presynaptic threshold potential
- ``Vslope:`` Activation slope of graded release

``event`` code is:
```
addToPost(fmax(0.0, g * tanh((V_pre - Epre) / Vslope) * dt));
```


``event`` threshold condition code is:

```
V_pre > Epre
```

\note The pre-synaptic variables are referenced with the suffix `_pre` in synapse related code
such as an the event threshold test. Users can also access post-synaptic neuron variables using the suffix `_post`.)doc";

static const char *__doc_WeightUpdateModels_StaticGraded_getInstance = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticGraded_getParams = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticGraded_getPreEventSynCode = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticGraded_getPreEventThresholdConditionCode = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticGraded_getPreNeuronVarRefs = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticGraded_getVars = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticPulse =
R"doc(Pulse-coupled, static synapse.
No learning rule is applied to the synapse and for each pre-synaptic spikes,
the synaptic conductances are simply added to the postsynaptic input variable.
The model has 1 variable:
- g - conductance of scalar type
and no other parameters.

``sim`` code is:

```
"addToPost(g);\n"
```)doc";

static const char *__doc_WeightUpdateModels_StaticPulseConstantWeight =
R"doc(Pulse-coupled, static synapse.
No learning rule is applied to the synapse and for each pre-synaptic spikes,
the synaptic conductances are simply added to the postsynaptic input variable.
The model has 1 parameter:
- g - conductance
and no other variables.

``sim`` code is:

```
"addToPost(g);"
```)doc";

static const char *__doc_WeightUpdateModels_StaticPulseConstantWeight_getInstance = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticPulseConstantWeight_getParams = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticPulseConstantWeight_getPreSpikeSynCode = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticPulseDendriticDelay =
R"doc(Pulse-coupled, static synapse with heterogenous dendritic delays
No learning rule is applied to the synapse and for each pre-synaptic spikes,
the synaptic conductances are simply added to the postsynaptic input variable.
The model has 2 variables:
- g - conductance of scalar type
- d - dendritic delay in timesteps
and no other parameters.

``sim`` code is:

```
"addToPostDelay(g, d);"
```)doc";

static const char *__doc_WeightUpdateModels_StaticPulseDendriticDelay_getInstance = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticPulseDendriticDelay_getPreSpikeSynCode = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticPulseDendriticDelay_getVars = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticPulse_getInstance = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticPulse_getPreSpikeSynCode = R"doc()doc";

static const char *__doc_WeightUpdateModels_StaticPulse_getVars = R"doc()doc";

static const char *__doc_binomialInverseCDF = R"doc()doc";

static const char *__doc_clearVarAccessDim = R"doc()doc";

static const char *__doc_createEGPRef = R"doc(Creates a reference to a neuron group extra global parameter)doc";

static const char *__doc_createEGPRef_2 = R"doc(Creates a reference to a current source extra global parameter)doc";

static const char *__doc_createEGPRef_3 = R"doc(Creates a reference to a custom update extra global parameter)doc";

static const char *__doc_createEGPRef_4 = R"doc(Creates a reference to a custom weight update extra global parameter)doc";

static const char *__doc_createPSMEGPRef = R"doc(Creates a reference to a postsynaptic model extra global parameter)doc";

static const char *__doc_createPSMVarRef = R"doc(Creates a reference to a postsynaptic model variable)doc";

static const char *__doc_createPostVarRef = R"doc(Creates a reference to a postsynaptic custom connectivity update variable)doc";

static const char *__doc_createPreVarRef = R"doc(Creates a reference to a presynaptic custom connectivity update variable)doc";

static const char *__doc_createVarRef = R"doc(Creates a reference to a neuron group variable)doc";

static const char *__doc_createVarRef_2 = R"doc(Creates a reference to a current source variable)doc";

static const char *__doc_createVarRef_3 = R"doc(Creates a reference to a custom update variable)doc";

static const char *__doc_createWUEGPRef = R"doc(Creates a reference to a weight update model extra global parameter)doc";

static const char *__doc_createWUPostVarRef = R"doc(Creates a reference to a weight update model postsynapticvariable)doc";

static const char *__doc_createWUPreVarRef = R"doc(Creates a reference to a weight update model presynaptic variable)doc";

static const char *__doc_createWUVarRef = R"doc(Creates a reference to a weight update model variable)doc";

static const char *__doc_createWUVarRef_2 = R"doc(Creates a reference to a custom weight update variable)doc";

static const char *__doc_createWUVarRef_3 = R"doc(Creates a reference to a custom connectivity update update variable)doc";

static const char *__doc_getSynapseMatrixConnectivity = R"doc()doc";

static const char *__doc_getSynapseMatrixWeight = R"doc()doc";

static const char *__doc_getVarAccessDim = R"doc()doc";

static const char *__doc_getVarAccessDim_2 = R"doc()doc";

static const char *__doc_getVarAccessMode = R"doc()doc";

static const char *__doc_getVarAccessMode_2 = R"doc()doc";

static const char *__doc_getVarAccessMode_3 = R"doc()doc";

static const char *__doc_initConnectivity =
R"doc(Initialise connectivity using a sparse connectivity snippet


$Template parameter ``S``:

       type of sparse connectivitiy initialisation snippet (derived from InitSparseConnectivitySnippet::Base).


$Parameter ``params``:

   parameters for snippet wrapped in ParamValues object.


$Returns:

InitSparseConnectivitySnippet::Init object for passing to ``ModelSpec::addSynapsePopulation``)doc";

static const char *__doc_initPostsynaptic =
R"doc(Initialise postsynaptic update model


$Template parameter ``S``:

               type of postsynaptic model initialisation snippet (derived from PostSynapticModels::Base).


$Parameter ``params``:

           parameters for snippet wrapped in ParamValues object.


$Parameter ``vars``:

             variables for snippet wrapped in VarValues object.


$Parameter ``neuronVarRefs``:

    neuron variable references for snippet wrapped in VarReferences object.


$Returns:

PostsynapticModels::Init object for passing to ``ModelSpec::addSynapsePopulation``)doc";

static const char *__doc_initToeplitzConnectivity =
R"doc(Initialise toeplitz connectivity using a toeplitz connectivity snippet


$Template parameter ``S``:

       type of toeplitz connectivitiy initialisation snippet (derived from InitToeplitzConnectivitySnippet::Base).


$Parameter ``params``:

   parameters for snippet wrapped in ParamValues object.


$Returns:

InitToeplitzConnectivitySnippet::Init object for passing to ``ModelSpec::addSynapsePopulation``)doc";

static const char *__doc_initVar =
R"doc(Initialise a variable using an initialisation snippet


$Template parameter ``S``:

       type of variable initialisation snippet (derived from InitVarSnippet::Base).


$Parameter ``params``:

   parameters for snippet wrapped in ParamValues object.


$Returns:

InitVarSnippet::Init object for use within model's VarValues)doc";

static const char *__doc_initWeightUpdate =
R"doc(Initialise weight update model


$Template parameter ``S``:

                   type of postsynaptic model initialisation snippet (derived from PostSynapticModels::Base).


$Parameter ``params``:

               parameters for snippet wrapped in ParamValues object.


$Parameter ``vars``:

                 variables for snippet wrapped in VarValues object.


$Parameter ``preVars``:

              presynaptic variables for snippet wrapped in VarValues object.


$Parameter ``postVars``:

             postsynaptic variables for snippet wrapped in VarValues object.


$Parameter ``preNeuronVarRefs``:

     presynaptic neuron variable references for snippet wrapped in VarReferences object.


$Parameter ``postNeuronVarRefs``:

    postsynaptic neuron variable references for snippet wrapped in VarReferences object.


$Returns:

PostsynapticModels::Init object for passing to ``ModelSpec::addSynapsePopulation``)doc";

static const char *__doc_operator_band = R"doc()doc";

static const char *__doc_operator_band_2 = R"doc()doc";

static const char *__doc_operator_band_3 = R"doc()doc";

static const char *__doc_operator_band_4 = R"doc()doc";

static const char *__doc_operator_band_5 = R"doc()doc";

static const char *__doc_operator_band_6 = R"doc()doc";

static const char *__doc_operator_band_7 = R"doc()doc";

static const char *__doc_operator_bor = R"doc()doc";

static const char *__doc_operator_bor_2 = R"doc()doc";

static const char *__doc_plog_IAppender = R"doc()doc";

static const char *__doc_uninitialisedConnectivity =
R"doc(Mark a synapse group's sparse connectivity as uninitialised
This means that the backend will not generate any automatic initialization code, but will instead
copy the connectivity from host to device during ``initializeSparse`` function
(and, if necessary generate any additional data structures it requires))doc";

static const char *__doc_uninitialisedVar =
R"doc(Mark a variable as uninitialised
This means that the backend will not generate any automatic initialization code, but will instead
copy the variable from host to device during ``initializeSparse`` function)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

