//--------------------------------------------------------------------------
/*! \file decode_shared_matrix_globalg_bitmask/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 1);

    SET_SIM_CODE("$(x)= $(Isyn);\n");

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);


void modelDefinition(ModelSpec &model)
{
#ifdef OPENCL_DEVICE
    GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
    GENN_PREFERENCES.manualDeviceID = OPENCL_DEVICE;
#endif
#ifdef OPENCL_PLATFORM
    GENN_PREFERENCES.manualPlatformID = OPENCL_PLATFORM;
#endif
    model.setDT(0.1);
    model.setName("decode_shared_matrix_globalg_bitmask");

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(1.0);    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<Neuron>("Post1", 4, {}, Neuron::VarValues(0.0));
    model.addNeuronPopulation<Neuron>("Post2", 4, {}, Neuron::VarValues(0.0));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn1", SynapseMatrixType::BITMASK_GLOBALG, NO_DELAY, "Pre", "Post1",
        {}, staticSynapseInit,
        {}, {});
    model.addSlaveSynapsePopulation<PostsynapticModels::DeltaCurr>(
        "Syn2", "Syn1", NO_DELAY, "Pre", "Post2",
        {}, {});
    
    model.setPrecision(GENN_FLOAT);
}
