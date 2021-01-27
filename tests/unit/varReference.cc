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
    DECLARE_MODEL(AlphaCurr, 1, 1);

    SET_DECAY_CODE(
        "$(x) = (DT * $(expDecay) * $(inSyn) * $(init)) + ($(expDecay) * $(x));\n"
        "$(inSyn)*=$(expDecay);\n");

    SET_CURRENT_CONVERTER_CODE("$(x)");

    SET_PARAM_NAMES({"tau"});

    SET_VARS({{"x", "scalar"}});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const std::vector<double> &pars, double dt) { return std::exp(-dt / pars[0]); }},
        {"init", [](const std::vector<double> &pars, double) { return (std::exp(1) / pars[0]); }}});
};
IMPLEMENT_MODEL(AlphaCurr);
}

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(VarReference, Neuron)
{
    ModelSpecInternal model;

    // Add neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    const auto *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Finalize model
    model.finalize();

    auto neuronVoltage = varReference(ng, "V");
    ASSERT_EQ(neuronVoltage.getType(), Models::VarReference::Type::Neuron);

    try {
        auto neuronMagic = varReference(ng, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(VarReference, CurrentSource)
{
    ModelSpecInternal model;

    // Add neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add one poisson exp current source
    CurrentSourceModels::PoissonExp::ParamValues cs0ParamVals(0.1, 5.0, 10.0);
    CurrentSourceModels::PoissonExp::VarValues cs0VarVals(0.0);
    auto *cs0 = model.addCurrentSource<CurrentSourceModels::PoissonExp>("CS0", "Neurons0",
                                                                        cs0ParamVals, cs0VarVals);

    // Finalize model
    model.finalize();

    auto csCurrent = varReference(cs0, "current");
    ASSERT_EQ(csCurrent.getType(), Models::VarReference::Type::CurrentSource);

    try {
        auto csMagic = varReference(cs0, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(VarReference, PSM)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("Synapses1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                       "Pre", "Post",
                                                                                       {}, {1.0},
                                                                                       {5.0}, {0.0});

    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("Synapses2", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                                       "Pre", "Post",
                                                                                       {}, {1.0},
                                                                                       {5.0}, {0.0});


    // Finalize model
    model.finalize();

    auto psmX = psVarReference(sg1, "x");
    ASSERT_EQ(psmX.getType(), Models::VarReference::Type::PSM);

    // Test error if variable doesn't exist
    try {
        auto psmMagic = psVarReference(sg1, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Test error if GLOBALG
    try {
        auto psmMagic = psVarReference(sg2, "x");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(VarReference, WUM)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("Synapses1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                       "Pre", "Post",
                                                                                       {}, {1.0},
                                                                                       {5.0}, {0.0});

    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("Synapses2", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                                       "Pre", "Post",
                                                                                       {}, {1.0},
                                                                                       {5.0}, {0.0});
    // Finalize model
    model.finalize();

    auto wuG1 = wuVarReference(sg1, "g");
    ASSERT_EQ(wuG1.getType(), Models::VarReference::Type::WU);

    // Test error if variable doesn't exist
    try {
        auto wuMagic = wuVarReference(sg1, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Test error if GLOBALG
    try {
        auto wuG2 = wuVarReference(sg2, "x");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
