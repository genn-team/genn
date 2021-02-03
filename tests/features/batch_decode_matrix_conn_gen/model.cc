//--------------------------------------------------------------------------
/*! \file batch_decode_matrix_conn_gen/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// DecoderVar
//----------------------------------------------------------------------------
class DecoderVar : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(DecoderVar, 0);

    SET_CODE(
        "const unsigned int j_value = (1 << $(id_post));\n"
        "$(value) = ((($(id_pre) + 1) & j_value) != 0) ? 1.0f : 0.0f;\n")
};
IMPLEMENT_SNIPPET(DecoderVar);

//----------------------------------------------------------------------------
// DecoderSparse
//----------------------------------------------------------------------------
class DecoderSparse : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(DecoderSparse, 0);

    SET_ROW_BUILD_CODE(
        "if(j < $(num_post)) {\n"
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
    SET_ROW_BUILD_STATE_VARS({{"j", "unsigned int", 0}});
};
IMPLEMENT_SNIPPET(DecoderSparse);

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
    model.setDT(0.1);
    model.setName("batch_decode_matrix_conn_gen");
    model.setBatchSize(2);

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<Neuron>("PostDense", 4, {}, Neuron::VarValues(0.0));
    model.addNeuronPopulation<Neuron>("PostSparse", 4, {}, Neuron::VarValues(0.0));
    model.addNeuronPopulation<Neuron>("PostProcedural", 4, {}, Neuron::VarValues(0.0));
    model.addNeuronPopulation<Neuron>("PostBitmask", 4, {}, Neuron::VarValues(0.0));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "SynDense", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY, "Pre", "PostDense",
        {}, {initVar<DecoderVar>()},
        {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "SynSparse", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY, "Pre", "PostSparse",
        {}, {1.0},
        {}, {},
        initConnectivity<DecoderSparse>());
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "SynProcedural", SynapseMatrixType::PROCEDURAL_GLOBALG, NO_DELAY, "Pre", "PostProcedural",
        {}, {1.0},
        {}, {},
        initConnectivity<DecoderSparse>());
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "SynBitmask", SynapseMatrixType::BITMASK_GLOBALG, NO_DELAY, "Pre", "PostBitmask",
        {}, {1.0},
        {}, {},
        initConnectivity<DecoderSparse>());
    model.setPrecision(GENN_FLOAT);
}
