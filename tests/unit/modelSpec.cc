// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

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

    SET_PARAM_NAMES({"tau"});

    SET_VARS({{"x", "scalar"}});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const ParamValues &pars, double dt) { return std::exp(-dt / pars.at("tau")); }},
        {"init", [](const ParamValues &pars, double) { return (std::exp(1) / pars.at("tau")); }}});
};
IMPLEMENT_SNIPPET(AlphaCurr);
}

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(ModelSpec, NeuronGroupZeroCopy)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    ng->setSpikeLocation(VarLocation::HOST_DEVICE_ZERO_COPY);

    ASSERT_TRUE(model.zeroCopyInUse());
}
//--------------------------------------------------------------------------
TEST(ModelSpec, CurrentSourceZeroCopy)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
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
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    SynapseGroup *sg = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
        "Synapse", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Neurons0", "Neurons1",
        {}, {{"g", 1.0}},
        {{"tau", 5.0}}, {{"x", 0.0}});
    sg->setPSVarLocation("x", VarLocation::HOST_DEVICE_ZERO_COPY);

    ASSERT_TRUE(model.zeroCopyInUse());
}
//--------------------------------------------------------------------------
TEST(ModelSpec, WUZeroCopy)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    SynapseGroup *sg = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapse", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Neurons0", "Neurons1",
        {}, {{"g", 1.0}},
        {}, {});
    sg->setWUVarLocation("g", VarLocation::HOST_DEVICE_ZERO_COPY);

    ASSERT_TRUE(model.zeroCopyInUse());
}
