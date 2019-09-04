//--------------------------------------------------------------------------
/*! \file decode_matrix_merged_globalg_ragged/model.cc

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
    model.setDT(0.1);
    model.setName("decode_matrix_merged_globalg_ragged");
    model.setMergePostsynapticModels(true);

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(1.0);    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre1", 10, {}, {});
    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre2", 10, {}, {});
    model.addNeuronPopulation<Neuron>("Post", 4, {}, Neuron::VarValues(0.0));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY, "Pre1", "Post",
        {}, staticSynapseInit,
        {}, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn2", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY, "Pre2", "Post",
        {}, staticSynapseInit,
        {}, {});

    model.setPrecision(GENN_FLOAT);
}
