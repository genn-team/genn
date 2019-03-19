// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(NeuronGroup, ConstantVarIzhikevich)
{
    ModelSpec model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    ASSERT_FALSE(ng->isZeroCopyEnabled());
    ASSERT_FALSE(ng->isSimRNGRequired());
    ASSERT_FALSE(ng->isInitRNGRequired());
}

TEST(NeuronGroup, UnitialisedVarIzhikevich)
{
    ModelSpec model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(uninitialisedVar(), uninitialisedVar());
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    ASSERT_FALSE(ng->isZeroCopyEnabled());
    ASSERT_FALSE(ng->isSimRNGRequired());
    ASSERT_FALSE(ng->isInitRNGRequired());
}

TEST(NeuronGroup, UnitialisedVarRand)
{
    ModelSpec model;

    InitVarSnippet::Uniform::ParamValues dist(0.0, 1.0);
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, initVar<InitVarSnippet::Uniform>(dist));
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    ASSERT_FALSE(ng->isZeroCopyEnabled());
    ASSERT_FALSE(ng->isSimRNGRequired());
    ASSERT_TRUE(ng->isInitRNGRequired());
}

TEST(NeuronGroup, Poisson)
{
    ModelSpec model;

    NeuronModels::PoissonNew::ParamValues paramVals(20.0);
    NeuronModels::PoissonNew::VarValues varVals(0.0);
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::PoissonNew>("Neurons0", 10, paramVals, varVals);

    ASSERT_FALSE(ng->isZeroCopyEnabled());
    ASSERT_TRUE(ng->isSimRNGRequired());
    ASSERT_FALSE(ng->isInitRNGRequired());
}
