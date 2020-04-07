// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

class WeightUpdateModelPost : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModelPost, 1, 1, 0, 1);

    SET_VARS({{"w", "scalar"}});
    SET_PARAM_NAMES({"p"});
    SET_POST_VARS({{"s", "scalar"}});

    SET_SIM_CODE("$(w)= $(s);\n");
    SET_POST_SPIKE_CODE("$(s) = $(t) * $(p);\n");
};
IMPLEMENT_MODEL(WeightUpdateModelPost);


class WeightUpdateModelPre : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModelPre, 1, 1, 1, 0);

    SET_VARS({{"w", "scalar"}});
    SET_PARAM_NAMES({"p"});
    SET_PRE_VARS({{"s", "scalar"}});

    SET_SIM_CODE("$(w)= $(s);\n");
    SET_PRE_SPIKE_CODE("$(s) = $(t) * $(p);\n");
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

    // Check that all groups can be merged
    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal *>(ng0);
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng1));
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng2));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);
    
    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 1);

    // Check that only 'd' parameter is heterogeneous
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(0));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(1));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(2));
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(3));
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
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);

    // Add one gaussian and one DC current source to Neurons0
    CurrentSourceModels::GaussianNoise::ParamValues cs0ParamVals(0.0, 0.1);
    CurrentSourceModels::GaussianNoise::ParamValues cs1ParamVals(0.0, 0.2);
    CurrentSourceModels::DC::ParamValues cs2ParamVals(0.4);
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS0", "Neurons0", cs0ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::DC>("CS1", "Neurons0", cs2ParamVals, {});

    // Do the same for Neuron1
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS2", "Neurons1", cs0ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::DC>("CS3", "Neurons1", cs2ParamVals, {});

    // Do the same, but with different parameters for Neuron2
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS4", "Neurons2", cs1ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::DC>("CS5", "Neurons2", cs2ParamVals, {});

    // Do the same, but in the opposite order for Neuron3
    model.addCurrentSource<CurrentSourceModels::DC>("CS6", "Neurons3", cs2ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CS7", "Neurons3", cs0ParamVals, {});

    // Add two DC sources to Neurons4
    model.addCurrentSource<CurrentSourceModels::DC>("CS8", "Neurons4", cs2ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::DC>("CS9", "Neurons4", cs2ParamVals, {});

    // **TODO** heterogeneous params
    model.finalize();

    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal *>(ng0);
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng1));
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng2));
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng3));
    ASSERT_FALSE(ng0Internal->canBeMerged(*ng4));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check neurons are merged into two groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);

    // Find which merged neuron group is the one with the single population i.e. the two DC current sources
    const size_t dcDCIndex = (modelSpecMerged.getMergedNeuronUpdateGroups().at(0).getGroups().size() == 2) ? 1 : 0;
    const auto dcDCMergedGroup = modelSpecMerged.getMergedNeuronUpdateGroups().at(dcDCIndex);
    const auto dcGaussianMergedGroup = modelSpecMerged.getMergedNeuronUpdateGroups().at(1 - dcDCIndex);
    
    // Find which child in the DC + gaussian merged group is the gaussian current source
    const size_t gaussianIndex = (dcGaussianMergedGroup.getArchetype().getCurrentSources().at(0)->getCurrentSourceModel() == CurrentSourceModels::GaussianNoise::getInstance()) ? 0 : 1;
    
    // Check that only the standard deviation parameter of the gaussian current sources is heterogeneous
    ASSERT_FALSE(dcDCMergedGroup.isCurrentSourceParamHeterogeneous(0, 0));
    ASSERT_FALSE(dcDCMergedGroup.isCurrentSourceParamHeterogeneous(1, 0));
    ASSERT_FALSE(dcGaussianMergedGroup.isCurrentSourceParamHeterogeneous(gaussianIndex, 0));
    ASSERT_TRUE(dcGaussianMergedGroup.isCurrentSourceParamHeterogeneous(gaussianIndex, 1));
    ASSERT_FALSE(dcGaussianMergedGroup.isCurrentSourceParamHeterogeneous(1 - gaussianIndex, 0));
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
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    WeightUpdateModels::StaticPulse::VarValues staticPulseVarVals(0.1);
    PostsynapticModels::ExpCurr::ParamValues expCurrParamVals(0.5);
    PostsynapticModels::ExpCurr::ParamValues expCurrParamVals1(0.75);
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

    // Do the same, but with different parameters for Neuron2,
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "SpikeSource", "Neurons2",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>("SG5", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                             "SpikeSource", "Neurons2",
                                                                                             {}, staticPulseVarVals,
                                                                                             expCurrParamVals1, {});

    // Do the same, but in the opposite order for Neuron3
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>("SG6", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                             "SpikeSource", "Neurons3",
                                                                                             {}, staticPulseVarVals,
                                                                                             expCurrParamVals, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG7", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "SpikeSource", "Neurons3",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});

    // Add two incoming synapse groups with DeltaCurr postsynaptic models sources to Neurons4
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG8", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "SpikeSource", "Neurons4",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG9", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "SpikeSource", "Neurons4",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});

    model.finalize();

    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal *>(ng0);
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng1));
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng2));
    ASSERT_TRUE(ng0Internal->canBeMerged(*ng3));
    ASSERT_FALSE(ng0Internal->canBeMerged(*ng4));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check neurons are merged into three groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 3);

    // Find which merged neuron group is the one containing 4 groups i.e. the 4 neuron groups with a delta and an expcurr psm
    const auto deltaExpMergedGroup = std::find_if(modelSpecMerged.getMergedNeuronUpdateGroups().cbegin(), modelSpecMerged.getMergedNeuronUpdateGroups().cend(),
                                                  [](const CodeGenerator::NeuronUpdateGroupMerged &ng) { return (ng.getGroups().size() == 4); });

    // Find which child in the DC + gaussian merged group is the gaussian current source
    ASSERT_TRUE(deltaExpMergedGroup->getArchetype().getMergedInSyn().size() == 2);
    const size_t expIndex = (deltaExpMergedGroup->getArchetype().getMergedInSyn().at(0).first->getPSModel() == PostsynapticModels::ExpCurr::getInstance()) ? 0 : 1;

    // Check that parameter and both derived parameters are heterogeneous
    // **NOTE** tau is NOT heterogeneous because it's unused
    ASSERT_FALSE(deltaExpMergedGroup->isPSMParamHeterogeneous(expIndex, 0));
    ASSERT_TRUE(deltaExpMergedGroup->isPSMDerivedParamHeterogeneous(expIndex, 0));
    ASSERT_TRUE(deltaExpMergedGroup->isPSMDerivedParamHeterogeneous(expIndex, 1));
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
    auto *ng5 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons5", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    WeightUpdateModels::StaticPulse::VarValues staticPulseVarVals(0.1);
    WeightUpdateModelPre::VarValues testVarVals(0.0);
    WeightUpdateModelPre::ParamValues testParams(1.0);
    WeightUpdateModelPre::ParamValues testParams2(2.0);
    WeightUpdateModelPre::PreVarValues testPreVarVals(0.0);

    // Connect neuron group 1 to neuron group 0 with pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons1", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals, {},
                                                                                    {}, {});

    // Also connect neuron group 2 to neuron group 0 with pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons2", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals, {},
                                                                                    {}, {});

    // Also connect neuron group 3 to neuron group 0 with pre weight update model, but different parameters
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons3", "Neurons0",
                                                                                    testParams2, testVarVals, testPreVarVals, {},
                                                                                    {}, {});

    // Connect neuron group 4 to neuron group 0 with 2*pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG3", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons4", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals, {},
                                                                                    {}, {});
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons4", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals, {},
                                                                                    {}, {});

    // Connect neuron group 5 to neuron group 0 with pre weight update model and static pulse
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG5", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons5", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals, {},
                                                                                    {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG6", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "Neurons5", "Neurons0",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.finalize();

    // Check which groups can be merged together
    // **NOTE** NG1 and NG5 can be merged because the additional static pulse synapse population doesn't add any presynaptic update
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal *>(ng1);
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng2));
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng3));
    ASSERT_FALSE(ng1Internal->canBeMerged(*ng4));
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng5));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check neurons are merged into three groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 3);

    // Find which merged neuron group is the one containing 4 groups i.e. the 4 neuron groups with a single weight update model with presynaptic update
    const auto wumPreMergedGroup = std::find_if(modelSpecMerged.getMergedNeuronUpdateGroups().cbegin(), modelSpecMerged.getMergedNeuronUpdateGroups().cend(),
                                                [](const CodeGenerator::NeuronUpdateGroupMerged &ng) { return (ng.getGroups().size() == 4); });

    // Check that parameter is heterogeneous
    ASSERT_TRUE(wumPreMergedGroup->isOutSynWUMParamHeterogeneous(0, 0));
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
    auto *ng5 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons5", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    WeightUpdateModels::StaticPulse::VarValues staticPulseVarVals(0.1);
    WeightUpdateModelPost::VarValues testVarVals(0.0);
    WeightUpdateModelPost::ParamValues testParams(1.0);
    WeightUpdateModelPost::ParamValues testParams2(2.0);
    WeightUpdateModelPost::PostVarValues testPostVarVals(0.0);

    // Connect neuron group 0 to neuron group 1 with post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons1",
                                                                                     testParams, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});

    // Also connect neuron group 0 to neuron group 2 with post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons2",
                                                                                     testParams, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});

    // Also connect neuron group 0 to neuron group 3 with post weight update model but different parameters
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons3",
                                                                                     testParams2, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});

    // Connect neuron group 0 to neuron group 3 with 2*post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG3", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons4",
                                                                                     testParams, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons4",
                                                                                     testParams, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});

    // Connect neuron group 0 to neuron group 4 with post weight update model and static pulse
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG5", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons5",
                                                                                     testParams, testVarVals, {}, testPostVarVals,
                                                                                     {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG6", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "Neurons0", "Neurons5",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.finalize();

    // **NOTE** NG1 and NG5 can be merged because the additional static pulse synapse population doesn't add any presynaptic update
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal *>(ng1);
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng2));
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng3));
    ASSERT_FALSE(ng1Internal->canBeMerged(*ng4));
    ASSERT_TRUE(ng1Internal->canBeMerged(*ng5));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check neurons are merged into three groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 3);

    // Find which merged neuron group is the one containing 4 groups i.e. the 4 neuron groups with a single weight update model with presynaptic update
    const auto wumPostMergedGroup = std::find_if(modelSpecMerged.getMergedNeuronUpdateGroups().cbegin(), modelSpecMerged.getMergedNeuronUpdateGroups().cend(),
                                                 [](const CodeGenerator::NeuronUpdateGroupMerged &ng) { return (ng.getGroups().size() == 4); });

    // Check that parameter is heterogeneous
    ASSERT_TRUE(wumPostMergedGroup->isInSynWUMParamHeterogeneous(0, 0));

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
