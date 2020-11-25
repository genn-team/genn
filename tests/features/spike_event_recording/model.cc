//--------------------------------------------------------------------------
/*! \file spike_event_recording/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// PreNeuron
//----------------------------------------------------------------------------
class PreNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PreNeuron, 0, 3);
    SET_SIM_CODE(
        "if($(startSpike) != $(endSpike) && $(t) >= $(spikeTimes)[$(startSpike)]) {\n"
        "   $(output) = true;\n"
        "   $(startSpike)++;\n"
        "}\n"
        "else {\n"
        "   $(output) = false;\n"
        "}\n");
    SET_VARS({{"startSpike", "unsigned int"}, {"endSpike", "unsigned int", VarAccess::READ_ONLY}, {"output", "bool"}});
    SET_EXTRA_GLOBAL_PARAMS( {{"spikeTimes", "scalar*"}} );
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(PreNeuron);

//----------------------------------------------------------------------------
// PostNeuron
//----------------------------------------------------------------------------
class PostNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PostNeuron, 0, 0);
    
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(PostNeuron);

//----------------------------------------------------------------------------
// WUM
//----------------------------------------------------------------------------
class WUM : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WUM, 0, 0, 0, 0);

    SET_EVENT_CODE("while(false);");

    SET_EVENT_THRESHOLD_CONDITION_CODE("$(output_pre)");
};
IMPLEMENT_MODEL(WUM);

void modelDefinition(ModelSpec &model)
{
#ifdef CL_HPP_TARGET_OPENCL_VERSION
    if(std::getenv("OPENCL_DEVICE") != nullptr) {
        GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
        GENN_PREFERENCES.manualDeviceID = std::atoi(std::getenv("OPENCL_DEVICE"));
    }
    if(std::getenv("OPENCL_PLATFORM") != nullptr) {
        GENN_PREFERENCES.manualPlatformID = std::atoi(std::getenv("OPENCL_PLATFORM"));
    }
#endif
    model.setDT(1.0);
    model.setName("spike_event_recording");
    
    PreNeuron::VarValues varInit(uninitialisedVar(), uninitialisedVar(), false);
    auto *pre = model.addNeuronPopulation<PreNeuron>("Pre", 100, {}, varInit);
    model.addNeuronPopulation<PostNeuron>("Post", 100, {}, {});
    
    pre->setSpikeEventRecordingEnabled(true);
    
    model.addSynapsePopulation<WUM, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY, "Pre", "Post",
        {}, {},
        {}, {});
}
