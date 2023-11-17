// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
class AlphaCurr : public PostsynapticModels::Base
{
public:
    DECLARE_SNIPPET(AlphaCurr);

    SET_DECAY_CODE(
        "$(x) = (DT * $(expDecay) * $(inSyn) * $(init)) + ($(expDecay) * $(x));\n"
        "$(inSyn)*=$(expDecay);\n");

    SET_CURRENT_CONVERTER_CODE("$(x)");

    SET_PARAMS({"tau"});

    SET_VARS({{"x", "scalar"}});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const auto &pars, double dt) { return std::exp(-dt / pars.at("tau").cast<double>()); }},
        {"init", [](const auto &pars, double) { return (std::exp(1) / pars.at("tau").cast<double>()); }}});
};
IMPLEMENT_SNIPPET(AlphaCurr);


class Sum : public CustomUpdateModels::Base
{
    DECLARE_SNIPPET(Sum);

    SET_UPDATE_CODE("$(sum) = $(a) + $(b);\n");

    SET_CUSTOM_UPDATE_VARS({{"sum", "scalar"}});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_ONLY}, 
                  {"b", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_SNIPPET(Sum);

class RemoveSynapse : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_SNIPPET(RemoveSynapse);
    
    SET_VARS({{"a", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "for_each_synapse{\n"
        "   if(id_post == (id_pre + 1)) {\n"
        "       remove_synapse();\n"
        "       break;\n"
        "   }\n"
        "};\n");
};
IMPLEMENT_SNIPPET(RemoveSynapse);
}

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(ModelSpec, NeuronGroupZeroCopy)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    ng->setSpikeLocation(VarLocation::HOST_DEVICE_ZERO_COPY);

    ASSERT_TRUE(model.zeroCopyInUse());
}
//--------------------------------------------------------------------------
TEST(ModelSpec, CurrentSourceZeroCopy)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 10, paramVals, varVals);

    ParamValues csParamVals{{"weight", 0.1}, {"tauSyn", 5.0}, {"rate", 10.0}};
    VarValues csVarVals{{"current", 0.0}};
    CurrentSource *cs = model.addCurrentSource<CurrentSourceModels::PoissonExp>("CS", "Neurons", csParamVals, csVarVals);
    cs->setVarLocation("current", VarLocation::HOST_DEVICE_ZERO_COPY);

    ASSERT_TRUE(model.zeroCopyInUse());
}
//--------------------------------------------------------------------------
TEST(ModelSpec, PSMZeroCopy)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    SynapseGroup *sg = model.addSynapsePopulation(
        "Synapse", SynapseMatrixType::DENSE, NO_DELAY,
        "Neurons0", "Neurons1",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<AlphaCurr>({{"tau", 5.0}}, {{"x", 0.0}}));
    sg->setPSVarLocation("x", VarLocation::HOST_DEVICE_ZERO_COPY);

    ASSERT_TRUE(model.zeroCopyInUse());
}
//--------------------------------------------------------------------------
TEST(ModelSpec, WUZeroCopy)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    SynapseGroup *sg = model.addSynapsePopulation(
        "Synapse", SynapseMatrixType::DENSE, NO_DELAY,
        "Neurons0", "Neurons1",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    sg->setWUVarLocation("g", VarLocation::HOST_DEVICE_ZERO_COPY);

    ASSERT_TRUE(model.zeroCopyInUse());
}
//--------------------------------------------------------------------------
TEST(ModelSpec, CustomUpdateZeroCopy)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 10, paramVals, varVals);

    VarReferences varRefs{{"a", createVarRef(ng, "V")}, {"b", createVarRef(ng, "U")}};
    CustomUpdate *cu = model.addCustomUpdate<Sum>("Sum", "Test", 
                                                  {}, {{"sum", 0.0}}, varRefs);
    cu->setVarLocation("sum", VarLocation::HOST_DEVICE_ZERO_COPY);
    ASSERT_TRUE(model.zeroCopyInUse());
}
//--------------------------------------------------------------------------
TEST(ModelSpec, CustomConnectivityUpdateZeroCopy)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    model.addSynapsePopulation(
        "Synapse", SynapseMatrixType::SPARSE, NO_DELAY,
        "Neurons0", "Neurons1",
        initWeightUpdate<WeightUpdateModels::StaticPulseDendriticDelay>({}, {{"g", 1.0}, {"d", 1}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    CustomConnectivityUpdate *cu = model.addCustomConnectivityUpdate<RemoveSynapse>("RemoveSynapse", "Test", "Synapse",
                                                                                    {}, {{"a", 0.0}}, {}, {},
                                                                                    {}, {}, {});
    cu->setVarLocation("a", VarLocation::HOST_DEVICE_ZERO_COPY);
    ASSERT_TRUE(model.zeroCopyInUse());
}
