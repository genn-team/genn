//--------------------------------------------------------------------------
/*! \file decode_matrix_den_delay_individualg_ragged_pre/model.cc

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
    model.setName("decode_matrix_den_delay_individualg_ragged_pre");

    // Static synapse parameters
    WeightUpdateModels::StaticPulseDendriticDelay::VarValues staticSynapseInit(
        1.0,                    // 0 - Wij (nA)
        uninitialisedVar());    // 1 - Dij (timestep)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<Neuron>("Post", 1, {}, Neuron::VarValues(0.0));


    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "Pre", "Post",
        {}, staticSynapseInit,
        {}, {});
    syn->setMaxDendriticDelayTimesteps(10);
    syn->setMaxConnections(1);
    syn->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);

    model.setPrecision(GENN_FLOAT);
}
