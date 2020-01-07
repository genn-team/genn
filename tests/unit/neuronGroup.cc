// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

class WeightUpdateModelPost : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModelPost, 0, 1, 0, 1);

    SET_VARS({{"w", "scalar"}});
    SET_POST_VARS({{"s", "scalar"}});

    SET_SIM_CODE("$(w)= $(s);\n");
    SET_POST_SPIKE_CODE("$(s) = $(t);\n");
};
IMPLEMENT_MODEL(WeightUpdateModelPost);


class WeightUpdateModelPre : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModelPre, 0, 1, 1, 0);

    SET_VARS({{"w", "scalar"}});
    SET_PRE_VARS({{"s", "scalar"}});

    SET_SIM_CODE("$(w)= $(s);\n");
    SET_PRE_SPIKE_CODE("$(s) = $(t);\n");
};
IMPLEMENT_MODEL(WeightUpdateModelPre);

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

TEST(NeuronGroup, CompareDifferentParams)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramValsA(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::ParamValues paramValsB(0.02, 0.2, -65.0, 4.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramValsA, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramValsA, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramValsB, varVals);

    model.finalize();

    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal *>(ng0);
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng1));
    ASSERT_FALSE(ng0Internal->canBeMerged(*ng2));
}

TEST(NeuronGroup, CompareCurrentSources)
{
    ModelSpecInternal model;

    // Add four neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);

    // Add one gaussian and one DC current source to Neurons0
    CurrentSourceModels::GaussianNoise::ParamValues cs0ParamVals(0.0, 0.1);
    CurrentSourceModels::DC::ParamValues cs1ParamVals(0.4);
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS0", "Neurons0", cs0ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::DC>("CS1", "Neurons0", cs1ParamVals, {});

    // Do the same for Neuron1
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS2", "Neurons1", cs0ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::DC>("CS3", "Neurons1", cs1ParamVals, {});

    // Do the same, but in the opposite order for Neuron2
    model.addCurrentSource<CurrentSourceModels::DC>("CS4", "Neurons2", cs1ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS5", "Neurons2", cs0ParamVals, {});
    
    // Add two DC sources to Neurons3
    model.addCurrentSource<CurrentSourceModels::DC>("CS6", "Neurons3", cs1ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::DC>("CS7", "Neurons3", cs1ParamVals, {});

    model.finalize();

    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal *>(ng0);
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng1));
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng2));
    ASSERT_FALSE(ng0Internal->canBeMerged(*ng3));
}

TEST(NeuronGroup, ComparePostsynapticModels)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 10, {}, {});
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    WeightUpdateModels::StaticPulse::VarValues staticPulseVarVals(0.1);
    PostsynapticModels::ExpCurr::ParamValues expCurrParamVals(0.5);
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "SpikeSource", "Neurons0",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>("SG1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                             "SpikeSource", "Neurons0",
                                                                                             {}, staticPulseVarVals,
                                                                                             expCurrParamVals, {});

    // Do the same for Neuron1
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "SpikeSource", "Neurons1",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>("SG3", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                             "SpikeSource", "Neurons1",
                                                                                             {}, staticPulseVarVals,
                                                                                             expCurrParamVals, {});

    // Do the same, but in the opposite order for Neuron2
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>("SG4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                             "SpikeSource", "Neurons2",
                                                                                             {}, staticPulseVarVals,
                                                                                             expCurrParamVals, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG5", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "SpikeSource", "Neurons2",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});

    // Add two incoming synapse groups with DeltaCurr postsynaptic models sources to Neurons3
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG6", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "SpikeSource", "Neurons3",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG7", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "SpikeSource", "Neurons3",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});


    model.finalize();

    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal *>(ng0);
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng1));
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng2));
    ASSERT_FALSE(ng0Internal->canBeMerged(*ng3));
}


TEST(NeuronGroup, CompareWUPreUpdate)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    WeightUpdateModels::StaticPulse::VarValues staticPulseVarVals(0.1);
    WeightUpdateModelPre::VarValues testVarVals(0.0);
    WeightUpdateModelPre::PreVarValues testPreVarVals(0.0);

    // Connect neuron group 1 to neuron group 0 with pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons1", "Neurons0",
                                                                                    {}, testVarVals, testPreVarVals, {},
                                                                                    {}, {});

    // Also connect neuron group 2 to neuron group 0 with pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons2", "Neurons0",
                                                                                    {}, testVarVals, testPreVarVals, {},
                                                                                    {}, {});

    // Connect neuron group 3 to neuron group 0 with 2*pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons3", "Neurons0",
                                                                                    {}, testVarVals, testPreVarVals, {},
                                                                                    {}, {});
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG3", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons3", "Neurons0",
                                                                                    {}, testVarVals, testPreVarVals, {},
                                                                                    {}, {});

    // Connect neuron group 4 to neuron group 0 with pre weight update model and static pulse
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons4", "Neurons0",
                                                                                    {}, testVarVals, testPreVarVals, {},
                                                                                    {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG5", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "Neurons4", "Neurons0",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.finalize();

    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal *>(ng1);
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng2));
    ASSERT_FALSE(ng1Internal->canBeMerged(*ng3));
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng4));
}

TEST(NeuronGroup, CompareWUPostUpdate)
{
    ModelSpecInternal model;

    // **NOTE** we make sure merging is on so last test doesn't fail on that basis
    model.setMergePostsynapticModels(true);

    // Add two neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    WeightUpdateModels::StaticPulse::VarValues staticPulseVarVals(0.1);
    WeightUpdateModelPost::VarValues testVarVals(0.0);
    WeightUpdateModelPost::PostVarValues testPostVarVals(0.0);

    // Connect neuron group 0 to neuron group 1 with post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons1",
                                                                                     {}, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});

    // Also connect neuron group 0 to neuron group 2 with post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons2",
                                                                                     {}, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});

    // Connect neuron group 0 to neuron group 3 with 2*post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons3",
                                                                                     {}, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG3", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons3",
                                                                                     {}, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});

    // Connect neuron group 0 to neuron group 4 with post weight update model and static pulse
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons4",
                                                                                     {}, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG5", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "Neurons0", "Neurons4",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.finalize();

    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal *>(ng1);
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng2));
    ASSERT_FALSE(ng1Internal->canBeMerged(*ng3));
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng4));
}

TEST(NeuronGroup, InitCompareDifferentVars)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramValsA(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::ParamValues paramValsB(0.02, 0.2, -65.0, 4.0);
    NeuronModels::Izhikevich::VarValues varValsA(initVar<InitVarSnippet::Uniform>({0.0, 1.0}), 0.0);
    NeuronModels::Izhikevich::VarValues varValsB(0.0, initVar<InitVarSnippet::Uniform>({0.0, 1.0}));
    NeuronModels::Izhikevich::VarValues varValsC(0.0, 0.0);
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramValsA, varValsA);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramValsA, varValsB);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramValsB, varValsC);

    model.finalize();

    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal *>(ng0);
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng1));
    ASSERT_FALSE(ng0Internal->canBeMerged(*ng2));
}
