//--------------------------------------------------------------------------
/*! \file decode_matrix_conn_gen_proceduralg_dense_egp/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// Decoder
//----------------------------------------------------------------------------
class Decoder : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(Decoder, 0);

    SET_EXTRA_GLOBAL_PARAMS({{"rowWeights", "scalar*"}});
    SET_CODE("$(value) = $(id_pre) * $(rowWeights)[$(id_post)];\n")
};
IMPLEMENT_SNIPPET(Decoder);

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
    model.setName("decode_matrix_conn_gen_proceduralg_dense_egp");

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(initVar<Decoder>());    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<Neuron>("Post", 4, {}, Neuron::VarValues(0.0));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::DENSE_PROCEDURALG, NO_DELAY, "Pre", "Post",
        {}, staticSynapseInit,
        {}, {});

    model.setPrecision(GENN_FLOAT);
}
