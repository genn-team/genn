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
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add one gaussian current source
    ParamValues cs0ParamVals{{"mean", 0.0}, {"sd", 0.1}};
    CurrentSource *cs0 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS0", "Neurons0",
                                                                                   cs0ParamVals, {});

    // Add one DC current source
    ParamValues cs1ParamVals{{"amp", 0.4}};
    CurrentSource *cs1 = model.addCurrentSource<CurrentSourceModels::DC>("CS1", "Neurons0",
                                                                         cs1ParamVals, {});

    // Finalize model
    model.finalise();

    CurrentSourceInternal *cs0Internal = static_cast<CurrentSourceInternal*>(cs0);
    CurrentSourceInternal *cs1Internal = static_cast<CurrentSourceInternal*>(cs1);
    ASSERT_NE(cs0Internal->getHashDigest(), cs1Internal->getHashDigest());
}

TEST(CurrentSource, CompareDifferentParameters)
{
    ModelSpecInternal model;

    // Add neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add one gaussian current source
    ParamValues cs0ParamVals{{"mean", 0.0}, {"sd", 0.1}};
    CurrentSource *cs0 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS0", "Neurons0",
                                                                                   cs0ParamVals, {});

    // Add second gaussian current source
    ParamValues cs1ParamVals{{"mean", 0.0}, {"sd", 0.5}};
    CurrentSource *cs1 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS1", "Neurons0",
                                                                                    cs1ParamVals, {});

    // Finalize model
    model.finalise();

    CurrentSourceInternal *cs0Internal = static_cast<CurrentSourceInternal*>(cs0);
    CurrentSourceInternal *cs1Internal = static_cast<CurrentSourceInternal*>(cs1);
    ASSERT_EQ(cs0Internal->getHashDigest(), cs1Internal->getHashDigest());
}

TEST(CurrentSource, CompareSameParameters)
{
    ModelSpecInternal model;

    // Add neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add one gaussian current source
    ParamValues cs0ParamVals{{"mean", 0.0}, {"sd", 0.1}};
    CurrentSource *cs0 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS0", "Neurons0",
                                                                                   cs0ParamVals, {});

    // Add second gaussian current source
    ParamValues cs1ParamVals{{"mean", 0.0}, {"sd", 0.1}};
    CurrentSource *cs1 = model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS1", "Neurons0",
                                                                                    cs1ParamVals, {});

    // Finalize model
    model.finalise();

    CurrentSourceInternal *cs0Internal = static_cast<CurrentSourceInternal*>(cs0);
    CurrentSourceInternal *cs1Internal = static_cast<CurrentSourceInternal*>(cs1);
    ASSERT_EQ(cs0Internal->getHashDigest(), cs1Internal->getHashDigest());
}

TEST(CurrentSource, InvalidName)
{
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    
    ModelSpec model;
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pop", 10, paramVals, varVals);
    
    try {
        model.addCurrentSource<CurrentSourceModels::DC>("CS-2", "Pop", {{"amp", 1.0}}, {});
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
