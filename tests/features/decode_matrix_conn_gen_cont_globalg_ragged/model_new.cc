#include "modelSpec.h"

//----------------------------------------------------------------------------
// Decoder
//----------------------------------------------------------------------------
class Decoder : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(Decoder, 0);

    SET_ROW_BUILD_CODE(
        "if($(isPostNeuronValid, j)) {\n"
        "   const unsigned int jValue = (1 << j);\n"
        "   if((($(id_pre) + 1) & jValue) != 0)\n"
        "   {\n"
        "       $(addSynapse, j);\n"
        "   }\n"
        "}\n"
        "else {\n"
        "   $(endRow);\n"
        "}\n"
        "j++;\n");
    SET_ROW_BUILD_STATE_VARS({{"j", {"unsigned int", 0}}});
};
IMPLEMENT_SNIPPET(Decoder);

//----------------------------------------------------------------------------
// PreNeuron
//----------------------------------------------------------------------------
class PreNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PreNeuron, 0, 1);

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(PreNeuron);

//----------------------------------------------------------------------------
// PostNeuron
//----------------------------------------------------------------------------
class PostNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PostNeuron, 0, 1);

    SET_SIM_CODE("$(x)= $(Isyn);\n");

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(PostNeuron);

//---------------------------------------------------------------------------
// Continuous
//---------------------------------------------------------------------------
class Continuous : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Continuous, 0, 1);

    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "$(addtoinSyn) = $(g) * $(x_pre);\n"
        "$(updatelinsyn);\n");
};
IMPLEMENT_MODEL(Continuous);


void modelDefinition(NNmodel &model)
{
    initGeNN();

    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::defaultSparseConnectivityMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;

    model.setDT(1.0);
    model.setName("decode_matrix_conn_gen_cont_globalg_ragged_new");

    // Continuous synapse parameters
    Continuous::VarValues staticSynapseInit(1.0);    // 0 - Wij (nA)

    model.addNeuronPopulation<PreNeuron>("Pre", 10, {}, PreNeuron::VarValues(0.0));
    model.addNeuronPopulation<PostNeuron>("Post", 4, {}, PostNeuron::VarValues(0.0));

    model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY, "Pre", "Post",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<Decoder>({}));

    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
