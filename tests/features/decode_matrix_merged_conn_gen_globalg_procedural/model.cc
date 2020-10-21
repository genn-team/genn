//--------------------------------------------------------------------------
/*! \file decode_matrix_merged_conn_gen_globalg_procedural/model.cc

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
    model.setName("decode_matrix_merged_conn_gen_globalg_procedural");

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(1.0);    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<Neuron>("Post1", 32, {}, Neuron::VarValues(0.0));
    model.addNeuronPopulation<Neuron>("Post2", 32, {}, Neuron::VarValues(0.0));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn1", SynapseMatrixType::PROCEDURAL_GLOBALG, NO_DELAY, "Pre", "Post1",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>({0.5}));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn2", SynapseMatrixType::PROCEDURAL_GLOBALG, NO_DELAY, "Pre", "Post2",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>({0.5}));
    model.setPrecision(GENN_FLOAT);
}
