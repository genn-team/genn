// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(CurrentSource, CompareDifferentModel)
{
    ModelSpecInternal model;

    // Add neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pop = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add one gaussian current source
    ParamValues cs0ParamVals{{"mean", 0.0}, {"sd", 0.1}};
    auto *cs0 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS0", pop, cs0ParamVals);

    // Add one DC current source
    ParamValues cs1ParamVals{{"amp", 0.4}};
    auto *cs1 = model.addCurrentSource<CurrentSourceModels::DC>("CS1", pop, cs1ParamVals);

    // Finalize model
    model.finalise();

    CurrentSourceInternal *cs0Internal = static_cast<CurrentSourceInternal*>(cs0);
    CurrentSourceInternal *cs1Internal = static_cast<CurrentSourceInternal*>(cs1);
    ASSERT_NE(cs0Internal->getHashDigest(pop), cs1Internal->getHashDigest(pop));
}

TEST(CurrentSource, CompareDifferentParameters)
{
    ModelSpecInternal model;

    // Add neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pop = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add one gaussian current source
    ParamValues cs0ParamVals{{"mean", 0.0}, {"sd", 0.1}};
    auto *cs0 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS0", pop, cs0ParamVals);

    // Add second gaussian current source
    ParamValues cs1ParamVals{{"mean", 0.0}, {"sd", 0.5}};
    auto *cs1 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS1", pop, cs1ParamVals);

    // Finalize model
    model.finalise();

    CurrentSourceInternal *cs0Internal = static_cast<CurrentSourceInternal*>(cs0);
    CurrentSourceInternal *cs1Internal = static_cast<CurrentSourceInternal*>(cs1);
    ASSERT_EQ(cs0Internal->getHashDigest(pop), cs1Internal->getHashDigest(pop));
}

TEST(CurrentSource, CompareSameParameters)
{
    ModelSpecInternal model;

    // Add neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pop = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add one gaussian current source
    ParamValues cs0ParamVals{{"mean", 0.0}, {"sd", 0.1}};
    auto *cs0 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS0", pop, cs0ParamVals);

    // Add second gaussian current source
    ParamValues cs1ParamVals{{"mean", 0.0}, {"sd", 0.1}};
    auto *cs1 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS1", pop, cs1ParamVals);

    // Finalize model
    model.finalise();

    CurrentSourceInternal *cs0Internal = static_cast<CurrentSourceInternal*>(cs0);
    CurrentSourceInternal *cs1Internal = static_cast<CurrentSourceInternal*>(cs1);
    ASSERT_EQ(cs0Internal->getHashDigest(pop), cs1Internal->getHashDigest(pop));
}

TEST(CurrentSource, InvalidName)
{
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    
    ModelSpec model;
    auto *pop = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pop", 10, paramVals, varVals);
    
    try {
        model.addCurrentSource<CurrentSourceModels::DC>("CS-2", pop, {{"amp", 1.0}});
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
