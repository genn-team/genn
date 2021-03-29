// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
class Sum : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(Sum, 0, 1, 2);

    SET_UPDATE_CODE("$(sum) = $(a) + $(b);\n");

    SET_VARS({{"sum", "scalar"}});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_ONLY}, 
                  {"b", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(Sum);

class Sum2 : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(Sum2, 0, 1, 2);

    SET_UPDATE_CODE("$(a) = $(mult) * ($(a) + $(b));\n");

    SET_VARS({{"mult", "scalar", VarAccess::READ_ONLY}});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_WRITE}, 
                  {"b", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(Sum2);

class Cont : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(Cont, 0, 1, 0, 0);

    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "$(addToInSyn, $(g) * $(V_pre));\n");
};
IMPLEMENT_MODEL(Cont);

class Cont2 : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(Cont2, 0, 2, 0, 0);

    SET_VARS({{"g", "scalar"}, {"x", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "$(addToInSyn, ($(g) + $(x)) * $(V_pre));\n");
};
IMPLEMENT_MODEL(Cont2);
}
//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(CustomUpdates, VarReferenceTypeChecks)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::DeltaCurr>(
        "Synapses", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0, 4},
        {}, {});

    Sum::VarValues sumVarValues(0.0);
    Sum::WUVarReferences sumVarReferences1(createWUVarRef(sg1, "g"), createWUVarRef(sg1, "g"));
    Sum::WUVarReferences sumVarReferences2(createWUVarRef(sg1, "g"), createWUVarRef(sg1, "d"));

    model.addCustomUpdate<Sum>("SumWeight1", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences1);

    try {
        model.addCustomUpdate<Sum>("SumWeight2", "CustomUpdate",
                                   {}, sumVarValues, sumVarReferences2);
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    model.finalize();
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, VarSizeChecks)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neuron1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neuron2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neuron3", 25, paramVals, varVals);

    Sum::VarValues sumVarValues(0.0);
    Sum::VarReferences sumVarReferences1(createVarRef(ng1, "V"), createVarRef(ng1, "U"));
    Sum::VarReferences sumVarReferences2(createVarRef(ng1, "V"), createVarRef(ng2, "V"));
    Sum::VarReferences sumVarReferences3(createVarRef(ng1, "V"), createVarRef(ng3, "V"));

    model.addCustomUpdate<Sum>("Sum1", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences1);
    model.addCustomUpdate<Sum>("Sum2", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences2);

    try {
        model.addCustomUpdate<Sum>("Sum3", "CustomUpdate",
                                   {}, sumVarValues, sumVarReferences3);
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    model.finalize();
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, VarDelayChecks)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *pre1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre1", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Add synapse groups to force pre1's v to be delayed by 10 timesteps and pre2's v to be delayed by 5 timesteps
    model.addSynapsePopulation<Cont, PostsynapticModels::DeltaCurr>("Syn1", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                    10, "Pre1", "Post",
                                                                    {}, {0.1}, {}, {});

    Sum::VarValues sumVarValues(0.0);
    Sum::VarReferences sumVarReferences1(createVarRef(pre1, "V"), createVarRef(post, "V"));

    model.addCustomUpdate<Sum>("Sum1", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences1);

    model.finalize();
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, VarMixedDelayChecks)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *pre1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre1", 10, paramVals, varVals);
    auto *pre2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre2", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Add synapse groups to force pre1's v to be delayed by 10 timesteps and pre2's v to be delayed by 5 timesteps
    model.addSynapsePopulation<Cont, PostsynapticModels::DeltaCurr>("Syn1", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                    10, "Pre1", "Post",
                                                                    {}, {0.1}, {}, {});
    model.addSynapsePopulation<Cont, PostsynapticModels::DeltaCurr>("Syn2", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                    5, "Pre2", "Post",
                                                                    {}, {0.1}, {}, {});

    Sum::VarValues sumVarValues(0.0);
    Sum::VarReferences sumVarReferences2(createVarRef(pre1, "V"), createVarRef(pre2, "V"));

    model.addCustomUpdate<Sum>("Sum1", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences2);

    try {
        model.finalize();
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, WUVarSynapseGroupChecks)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::DeltaCurr>(
        "Synapses1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0, 4},
        {}, {});
    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::DeltaCurr>(
        "Synapses2", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0, 4},
        {}, {});


    Sum::VarValues sumVarValues(0.0);
    Sum::WUVarReferences sumVarReferences1(createWUVarRef(sg1, "g"), createWUVarRef(sg1, "g"));
    Sum::WUVarReferences sumVarReferences2(createWUVarRef(sg1, "g"), createWUVarRef(sg2, "d"));

    model.addCustomUpdate<Sum>("SumWeight1", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences1);

    try {
        model.addCustomUpdate<Sum>("SumWeight2", "CustomUpdate",
                                   {}, sumVarValues, sumVarReferences2);
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    model.finalize();
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, BatchingVars)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add neuron and spike source (arbitrary choice of model with read_only variables) to model
    NeuronModels::IzhikevichVariable::VarValues izkVarVals(0.0, 0.0, 0.02, 0.2, -65.0, 8.);
    auto *pop = model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("Pop", 10, {}, izkVarVals);
    

    // Create updates where variable is shared and references vary
    Sum2::VarValues sum2VarValues(1.0);
    Sum2::VarReferences sum2VarReferences1(createVarRef(pop, "V"), createVarRef(pop, "U"));
    Sum2::VarReferences sum2VarReferences2(createVarRef(pop, "a"), createVarRef(pop, "b"));
    Sum2::VarReferences sum2VarReferences3(createVarRef(pop, "V"), createVarRef(pop, "a"));

    auto *sum1 = model.addCustomUpdate<Sum2>("Sum1", "CustomUpdate",
                                             {}, sum2VarValues, sum2VarReferences1);
    auto *sum2 = model.addCustomUpdate<Sum2>("Sum2", "CustomUpdate",
                                             {}, sum2VarValues, sum2VarReferences2);
    auto *sum3 = model.addCustomUpdate<Sum2>("Sum3", "CustomUpdate",
                                             {}, sum2VarValues, sum2VarReferences3);
    model.finalize();

    EXPECT_TRUE(static_cast<CustomUpdateInternal*>(sum1)->isBatched());
    EXPECT_FALSE(static_cast<CustomUpdateInternal*>(sum2)->isBatched());
    EXPECT_TRUE(static_cast<CustomUpdateInternal*>(sum3)->isBatched());
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, BatchingWriteShared)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add neuron and spike source (arbitrary choice of model with read_only variables) to model
    NeuronModels::IzhikevichVariable::VarValues izkVarVals(0.0, 0.0, 0.02, 0.2, -65.0, 8.);
    auto *pop = model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("Pop", 10, {}, izkVarVals);
    
    // Create custom update which tries to create a read-write refernece to a (which isn't batched)
    Sum2::VarValues sum2VarValues(1.0);
    Sum2::VarReferences sum2VarReferences(createVarRef(pop, "a"), createVarRef(pop, "V"));
    model.addCustomUpdate<Sum2>("Sum1", "CustomUpdate",
                                {}, sum2VarValues, sum2VarReferences);

    try {
        model.finalize();
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, CompareDifferentModel)
{
    ModelSpecInternal model;

    // Add neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *pop = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    
    // Add three custom updates with two different models
    Sum::VarReferences sumVarReferences(createVarRef(pop, "V"), createVarRef(pop, "U"));
    Sum2::VarReferences sum2VarReferences(createVarRef(pop, "V"), createVarRef(pop, "U"));
    auto *sum0 = model.addCustomUpdate<Sum>("Sum0", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences);
    auto *sum1 = model.addCustomUpdate<Sum>("Sum1", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences);
    auto *sum2 = model.addCustomUpdate<Sum2>("Sum2", "CustomUpdate",
                                             {}, {1.0}, sum2VarReferences);

    // Finalize model
    model.finalize();

    CustomUpdateInternal *sum0Internal = static_cast<CustomUpdateInternal*>(sum0);
    ASSERT_TRUE(sum0Internal->canBeMerged(*sum1));
    ASSERT_FALSE(sum0Internal->canBeMerged(*sum2));
    ASSERT_TRUE(sum0Internal->canInitBeMerged(*sum1));
    ASSERT_FALSE(sum0Internal->canInitBeMerged(*sum2));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateInitGroups().size() == 2);
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, CompareDifferentUpdateGroup)
{
    ModelSpecInternal model;

    // Add neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *pop = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add three custom updates in two different update groups
    Sum::VarReferences sumVarReferences(createVarRef(pop, "V"), createVarRef(pop, "U"));
    auto *sum0 = model.addCustomUpdate<Sum>("Sum0", "CustomUpdate1",
                                            {}, {0.0}, sumVarReferences);
    auto *sum1 = model.addCustomUpdate<Sum>("Sum1", "CustomUpdate2",
                                            {}, {1.0}, sumVarReferences);
    auto *sum2 = model.addCustomUpdate<Sum>("Sum2", "CustomUpdate1",
                                            {}, {1.0}, sumVarReferences);
    // Finalize model
    model.finalize();

    CustomUpdateInternal *sum0Internal = static_cast<CustomUpdateInternal *>(sum0);
    ASSERT_FALSE(sum0Internal->canBeMerged(*sum1));
    ASSERT_TRUE(sum0Internal->canBeMerged(*sum2));
    ASSERT_TRUE(sum0Internal->canInitBeMerged(*sum1));
    ASSERT_TRUE(sum0Internal->canInitBeMerged(*sum2));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    // **NOTE** update groups don't matter for initialization
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateInitGroups().size() == 1);
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, CompareDifferentDelay)
{
    ModelSpecInternal model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *pre1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre1", 10, paramVals, varVals);
    auto *pre2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre2", 10, paramVals, varVals);
    auto *pre3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre3", 10, paramVals, varVals);
    auto *pre4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre4", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Add synapse groups to force pre1's v to be delayed by 0 timesteps, pre2 and pre3's v to be delayed by 5 timesteps and pre4's to be delayed by 10 timesteps
    model.addSynapsePopulation<Cont, PostsynapticModels::DeltaCurr>("Syn1", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                    NO_DELAY, "Pre1", "Post",
                                                                    {}, {0.1}, {}, {});
    model.addSynapsePopulation<Cont, PostsynapticModels::DeltaCurr>("Syn2", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                    5, "Pre2", "Post",
                                                                    {}, {0.1}, {}, {});
    model.addSynapsePopulation<Cont, PostsynapticModels::DeltaCurr>("Syn3", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                    5, "Pre3", "Post",
                                                                    {}, {0.1}, {}, {});
    model.addSynapsePopulation<Cont, PostsynapticModels::DeltaCurr>("Syn4", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                    10, "Pre4", "Post",
                                                                    {}, {0.1}, {}, {});
    

    // Add three custom updates in two different update groups
    Sum::VarReferences sumVarReferences1(createVarRef(pre1, "V"), createVarRef(pre1, "U"));
    Sum::VarReferences sumVarReferences2(createVarRef(pre2, "V"), createVarRef(pre2, "U"));
    Sum::VarReferences sumVarReferences3(createVarRef(pre3, "V"), createVarRef(pre3, "U"));
    Sum::VarReferences sumVarReferences4(createVarRef(pre4, "V"), createVarRef(pre4, "U"));
    auto *sum1 = model.addCustomUpdate<Sum>("Sum1", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences1);
    auto *sum2 = model.addCustomUpdate<Sum>("Sum2", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences2);
    auto *sum3 = model.addCustomUpdate<Sum>("Sum3", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences3);
    auto *sum4 = model.addCustomUpdate<Sum>("Sum4", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences4);

    // Finalize model
    model.finalize();

    // No delay group can't be merged with any others
    CustomUpdateInternal *sum1Internal = static_cast<CustomUpdateInternal*>(sum1);
    ASSERT_FALSE(sum1Internal->canBeMerged(*sum2));
    ASSERT_FALSE(sum1Internal->canBeMerged(*sum3));
    ASSERT_FALSE(sum1Internal->canBeMerged(*sum4));

    // Delay groups don't matter for initialisation
    ASSERT_TRUE(sum1Internal->canInitBeMerged(*sum2));
    ASSERT_TRUE(sum1Internal->canInitBeMerged(*sum3));
    ASSERT_TRUE(sum1Internal->canInitBeMerged(*sum4));

    CustomUpdateInternal *sum2Internal = static_cast<CustomUpdateInternal*>(sum2);
    ASSERT_TRUE(sum2Internal->canBeMerged(*sum3));
    ASSERT_FALSE(sum2Internal->canBeMerged(*sum4));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    // **NOTE** delay groups don't matter for initialization
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateGroups().size() == 3);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateInitGroups().size() == 1);
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, CompareDifferentBatched)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add neuron and spike source (arbitrary choice of model with read_only variables) to model
    NeuronModels::IzhikevichVariable::VarValues izkVarVals(0.0, 0.0, 0.02, 0.2, -65.0, 8.);
    auto *pop = model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("Pop", 10, {}, izkVarVals);

    // Add one custom update which sums duplicated variables (v and u) and another which sums shared variables (a and b)
    Sum::VarReferences sumVarReferences1(createVarRef(pop, "V"), createVarRef(pop, "U"));
    Sum::VarReferences sumVarReferences2(createVarRef(pop, "a"), createVarRef(pop, "b"));
    auto *sum1 = model.addCustomUpdate<Sum>("Sum1", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences1);
    auto *sum2 = model.addCustomUpdate<Sum>("Sum2", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences2);

    model.finalize();

    // Check that sum1 is batched and sum is not
    CustomUpdateInternal *sum1Internal = static_cast<CustomUpdateInternal*>(sum1);
    CustomUpdateInternal *sum2Internal = static_cast<CustomUpdateInternal*>(sum2);
    ASSERT_TRUE(sum1Internal->isBatched());
    ASSERT_FALSE(sum2Internal->isBatched());

    // Check that this means they can't be merged
    ASSERT_FALSE(sum1Internal->canBeMerged(*sum2));
    ASSERT_FALSE(sum1Internal->canInitBeMerged(*sum2));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    // **NOTE** delay groups don't matter for initialization
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateInitGroups().size() == 2);
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, CompareDifferentWUTranspose)
{
    ModelSpecInternal model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Add synapse groups to force pre1's v to be delayed by 0 timesteps, pre2 and pre3's v to be delayed by 5 timesteps and pre4's to be delayed by 10 timesteps
    auto *fwdSyn = model.addSynapsePopulation<Cont2, PostsynapticModels::DeltaCurr>("fwdSyn", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                                    NO_DELAY, "Pre", "Post",
                                                                                    {}, {0.0, 0.0}, {}, {});
    auto *backSyn = model.addSynapsePopulation<Cont, PostsynapticModels::DeltaCurr>("backSyn", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                                     NO_DELAY, "Post", "Pre",
                                                                                     {}, {0.0}, {}, {});

    // Add two custom updates which transpose different forward variables into backward population
    Sum2::WUVarReferences sumVarReferences1(createWUVarRef(fwdSyn, "g", backSyn, "g"), createWUVarRef(fwdSyn, "x"));
    Sum2::WUVarReferences sumVarReferences2(createWUVarRef(fwdSyn, "g"), createWUVarRef(fwdSyn, "x", backSyn, "g"));
    auto *sum1 = model.addCustomUpdate<Sum2>("Sum1", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences1);
    auto *sum2 = model.addCustomUpdate<Sum2>("Sum2", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences2);
    
    // Finalize model
    model.finalize();

    // Updates which transpose different variables can't be merged with any others
    CustomUpdateWUInternal *sum1Internal = static_cast<CustomUpdateWUInternal*>(sum1);
    ASSERT_FALSE(sum1Internal->canBeMerged(*sum2));

    // Again, this doesn't matter for initialisation
    ASSERT_TRUE(sum1Internal->canInitBeMerged(*sum2));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    // **NOTE** transpose variables don't matter for initialization
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateTransposeWUGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateWUGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedCustomWUUpdateDenseInitGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomWUUpdateSparseInitGroups().empty());
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, CompareDifferentWUConnectivity)
{
    ModelSpecInternal model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Add a sparse and a dense synapse group
    auto *syn1 = model.addSynapsePopulation<Cont2, PostsynapticModels::DeltaCurr>(
        "Syn1", SynapseMatrixType::DENSE_INDIVIDUALG,
        NO_DELAY, "Pre", "Post",
        {}, {0.0, 0.0}, {}, {});
    auto *syn2 = model.addSynapsePopulation<Cont2, PostsynapticModels::DeltaCurr>(
        "Syn2", SynapseMatrixType::SPARSE_INDIVIDUALG,
        NO_DELAY, "Pre", "Post",
        {}, {0.0, 0.0}, {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>({0.1}));

    // Add two custom updates which transpose different forward variables into backward population
    Sum::WUVarReferences sumVarReferences1(createWUVarRef(syn1, "g"), createWUVarRef(syn1, "x"));
    Sum::WUVarReferences sumVarReferences2(createWUVarRef(syn2, "g"), createWUVarRef(syn2, "x"));
    auto *sum1 = model.addCustomUpdate<Sum>("Decay1", "CustomUpdate",
                                             {}, {0.0}, sumVarReferences1);
    auto *sum2 = model.addCustomUpdate<Sum>("Decay2", "CustomUpdate",
                                             {}, {0.0}, sumVarReferences2);
    
    // Finalize model
    model.finalize();

    // Updates and initialisation with different connectivity can't be merged with any others
    CustomUpdateWUInternal *sum1Internal = static_cast<CustomUpdateWUInternal*>(sum1);
    ASSERT_FALSE(sum1Internal->canBeMerged(*sum2));
    ASSERT_FALSE(sum1Internal->canInitBeMerged(*sum2));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateTransposeWUGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateWUGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomWUUpdateDenseInitGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomWUUpdateSparseInitGroups().size() == 1);
}