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
class IzhikevichVariableShared : public NeuronModels::Izhikevich
{
public:
    DECLARE_MODEL(IzhikevichVariableShared, 0, 6);

    SET_PARAM_NAMES({});
    SET_VARS({{"V","scalar"}, {"U", "scalar"},
              {"a", "scalar", VarAccess::READ_ONLY_SHARED_NEURON}, {"b", "scalar", VarAccess::READ_ONLY_SHARED_NEURON},
              {"c", "scalar", VarAccess::READ_ONLY_SHARED_NEURON}, {"d", "scalar", VarAccess::READ_ONLY_SHARED_NEURON}});
};
IMPLEMENT_MODEL(IzhikevichVariableShared);

class StaticPulseDendriticDelaySplit : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(StaticPulseDendriticDelaySplit, 0, 4, 0, 0);

    SET_VARS({{"gCommon", "scalar", VarAccess::READ_ONLY}, 
              {"g", "scalar", VarAccess::READ_ONLY_DUPLICATE}, 
              {"dCommon", "scalar", VarAccess::READ_ONLY},
              {"d", "scalar", VarAccess::READ_ONLY_DUPLICATE}});

    SET_SIM_CODE("$(addToInSynDelay, $(gCommon) + $(g), $(dCommon) + $(d));\n");
};
IMPLEMENT_MODEL(StaticPulseDendriticDelaySplit);

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

class Sum3 : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(Sum3, 0, 2, 2);

    SET_UPDATE_CODE("$(sum) = $(scale) * ($(a) + $(b));\n");

    SET_VARS({{"sum", "scalar"}, {"scale", "scalar", VarAccess::READ_ONLY_SHARED_NEURON}});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_ONLY},
                  {"b", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(Sum3);

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

class Reduce : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(Reduce, 0, 0, 2);

    SET_UPDATE_CODE("$(reduction) = $(var);\n");

    SET_VAR_REFS({{"var", "scalar", VarAccessMode::READ_ONLY}, 
                  {"reduction", "scalar", VarAccessMode::REDUCE_SUM}});
};
IMPLEMENT_MODEL(Reduce);

class ReduceDouble : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(ReduceDouble, 0, 2, 2);

    SET_UPDATE_CODE(
        "$(reduction1) = $(var1);\n"
        "$(reduction2) = $(var2);\n");

    SET_VARS({{"reduction1", "scalar", VarAccess::REDUCE_BATCH_SUM},
              {"reduction2", "scalar", VarAccess::REDUCE_NEURON_SUM}});

    SET_VAR_REFS({{"var1", "scalar", VarAccessMode::READ_ONLY},
                  {"var2", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(ReduceDouble);

class ReduceSharedVar : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(ReduceSharedVar, 0, 1, 1);

    SET_UPDATE_CODE("$(reduction) = $(var);\n");

    SET_VARS({{"reduction", "scalar", VarAccess::REDUCE_BATCH_SUM}})
    SET_VAR_REFS({{"var", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(ReduceSharedVar);


class ReduceNeuronSharedVar : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(ReduceNeuronSharedVar, 0, 1, 1);

    SET_UPDATE_CODE("$(reduction) = $(var);\n");

    SET_VARS({{"reduction", "scalar", VarAccess::REDUCE_NEURON_SUM}})
    SET_VAR_REFS({{"var", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(ReduceNeuronSharedVar);
}
//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(CustomUpdates, ConstantVarSum)
{
    ModelSpecInternal model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    Sum::VarValues sumVarValues(0.0);
    Sum::VarReferences sumVarReferences1(createVarRef(ng, "V"), createVarRef(ng, "U"));
 
    CustomUpdate *cu = model.addCustomUpdate<Sum>("Sum", "CustomUpdate",
                                                  {}, sumVarValues, sumVarReferences1);
    model.finalize();

    CustomUpdateInternal *cuInternal = static_cast<CustomUpdateInternal*>(cu);
    ASSERT_FALSE(cuInternal->isZeroCopyEnabled());
    ASSERT_FALSE(cuInternal->isInitRNGRequired());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    ASSERT_FALSE(backend.isGlobalHostRNGRequired(modelSpecMerged));
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, UninitialisedVarSum)
{
    ModelSpecInternal model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    Sum::VarValues sumVarValues(uninitialisedVar());
    Sum::VarReferences sumVarReferences1(createVarRef(ng, "V"), createVarRef(ng, "U"));
 
    CustomUpdate *cu = model.addCustomUpdate<Sum>("Sum", "CustomUpdate",
                                                  {}, sumVarValues, sumVarReferences1);
    model.finalize();

    CustomUpdateInternal *cuInternal = static_cast<CustomUpdateInternal*>(cu);
    ASSERT_FALSE(cuInternal->isZeroCopyEnabled());
    ASSERT_FALSE(cuInternal->isInitRNGRequired());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    ASSERT_FALSE(backend.isGlobalHostRNGRequired(modelSpecMerged));
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, RandVarSum)
{
    ModelSpecInternal model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    InitVarSnippet::Uniform::ParamValues dist(0.0, 1.0);
    Sum::VarValues sumVarValues(initVar<InitVarSnippet::Uniform>(dist));
    Sum::VarReferences sumVarReferences1(createVarRef(ng, "V"), createVarRef(ng, "U"));
 
    CustomUpdate *cu = model.addCustomUpdate<Sum>("Sum", "CustomUpdate",
                                                  {}, sumVarValues, sumVarReferences1);
    model.finalize();

    CustomUpdateInternal *cuInternal = static_cast<CustomUpdateInternal*>(cu);
    ASSERT_FALSE(cuInternal->isZeroCopyEnabled());
    ASSERT_TRUE(cuInternal->isInitRNGRequired());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    ASSERT_TRUE(backend.isGlobalHostRNGRequired(modelSpecMerged));
}
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
    Sum::VarValues sumVarValues(0.0);
    Sum::VarReferences sumVarReferences1(createVarRef(pop, "V"), createVarRef(pop, "U"));
    Sum::VarReferences sumVarReferences2(createVarRef(pop, "a"), createVarRef(pop, "b"));
    Sum::VarReferences sumVarReferences3(createVarRef(pop, "V"), createVarRef(pop, "a"));

    auto *sum1 = model.addCustomUpdate<Sum>("Sum1", "CustomUpdate",
                                            {}, sumVarValues, sumVarReferences1);
    auto *sum2 = model.addCustomUpdate<Sum>("Sum2", "CustomUpdate",
                                            {}, sumVarValues, sumVarReferences2);
    auto *sum3 = model.addCustomUpdate<Sum>("Sum3", "CustomUpdate",
                                            {}, sumVarValues, sumVarReferences3);
    
    // Create one more update which references two variables in the non-batched sum2
    Sum::VarReferences sumVarReferences4(createVarRef(sum2, "sum"), createVarRef(sum2, "sum"));
    auto *sum4 = model.addCustomUpdate<Sum>("Sum4", "CustomUpdate",
                                            {}, sumVarValues, sumVarReferences4);
    
    model.finalize();

    EXPECT_TRUE(static_cast<CustomUpdateInternal*>(sum1)->isBatched());
    EXPECT_FALSE(static_cast<CustomUpdateInternal*>(sum2)->isBatched());
    EXPECT_TRUE(static_cast<CustomUpdateInternal*>(sum3)->isBatched());
    EXPECT_FALSE(static_cast<CustomUpdateInternal*>(sum4)->isBatched());
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
    Reduce::VarReferences reduceVarReferences(createVarRef(pop, "V"), createVarRef(pop, "U"));
    try {
        model.addCustomUpdate<Reduce>("Sum1", "CustomUpdate",
                                      {}, {}, reduceVarReferences);
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, ReduceDuplicate)
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
TEST(CustomUpdates, ReductionTypeDuplicateNeuron)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add neuron (copy of izhikevich model where a, b, c and d are shared_neuron) to model
    IzhikevichVariableShared::VarValues izkVarVals(0.0, 0.0, 0.02, 0.2, -65.0, 8.);
    auto *pop = model.addNeuronPopulation<IzhikevichVariableShared>("Pop", 10, {}, izkVarVals);

    // Create custom update which tries to reduce V into A
    Reduce::VarReferences reduceVarReferences(createVarRef(pop, "V"), createVarRef(pop, "a"));
    auto *cu = model.addCustomUpdate<Reduce>("Reduction", "CustomUpdate",
                                             {}, {}, reduceVarReferences);
    model.finalize();
    auto *cuInternal = static_cast<CustomUpdateInternal *>(cu);
    ASSERT_TRUE(cuInternal->isBatched());
    ASSERT_FALSE(cuInternal->isBatchReduction());
    ASSERT_TRUE(cuInternal->isNeuronReduction());
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, ReductionTypeDuplicateNeuronInternal)
{
    ModelSpecInternal model;
    model.setBatchSize(5); 

    // Add neuron (arbitrary choice of model with read_only variables) to model
    NeuronModels::IzhikevichVariable::VarValues izkVarVals(0.0, 0.0, 0.02, 0.2, -65.0, 8.);
    auto *pop = model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("Pop", 10, {}, izkVarVals);

    // Create custom update which tries to reduce V into A
    ReduceNeuronSharedVar::VarReferences reduceVarReferences(createVarRef(pop, "V"));
    ReduceNeuronSharedVar::VarValues reduceVars(0.0);
    auto *cu = model.addCustomUpdate<ReduceNeuronSharedVar>("Reduction", "CustomUpdate",
                                                            {}, reduceVars, reduceVarReferences);
    model.finalize();
    auto *cuInternal = static_cast<CustomUpdateInternal *>(cu);
    ASSERT_TRUE(cuInternal->isBatched());
    ASSERT_FALSE(cuInternal->isBatchReduction());
    ASSERT_TRUE(cuInternal->isNeuronReduction());
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, ReductionTypeSharedNeuronInternal)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add neuron (arbitrary choice of model with read_only variables) to model
    NeuronModels::IzhikevichVariable::VarValues izkVarVals(0.0, 0.0, 0.02, 0.2, -65.0, 8.);
    auto *pop = model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("Pop", 10, {}, izkVarVals);

    // Create custom update which tries to reduce a
    ReduceNeuronSharedVar::VarReferences reduceVarReferences(createVarRef(pop, "a"));
    ReduceNeuronSharedVar::VarValues reduceVars(0.0);
    auto *cu = model.addCustomUpdate<ReduceNeuronSharedVar>("Reduction", "CustomUpdate",
                                                            {}, reduceVars, reduceVarReferences);
    model.finalize();
    auto *cuInternal = static_cast<CustomUpdateInternal *>(cu);
    ASSERT_FALSE(cuInternal->isBatched());
    ASSERT_FALSE(cuInternal->isBatchReduction());
    ASSERT_TRUE(cuInternal->isNeuronReduction());
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, ReductionTypeDuplicateBatch)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add neuron (arbitrary choice of model with read_only variables) to model
    NeuronModels::IzhikevichVariable::VarValues izkVarVals(0.0, 0.0, 0.02, 0.2, -65.0, 8.);
    auto *pop = model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("Pop", 10, {}, izkVarVals);

    // Create custom update which tries to reduce V into A
    Reduce::VarReferences reduceVarReferences(createVarRef(pop, "V"), createVarRef(pop, "a"));
    auto *cu = model.addCustomUpdate<Reduce>("Reduction", "CustomUpdate",
                                             {}, {}, reduceVarReferences);
    model.finalize();
    auto *cuInternal = static_cast<CustomUpdateInternal *>(cu);
    ASSERT_TRUE(cuInternal->isBatched());
    ASSERT_TRUE(cuInternal->isBatchReduction());
    ASSERT_FALSE(cuInternal->isNeuronReduction());
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, ReductionTypeDuplicateBatchInternal)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add neuron (arbitrary choice of model with read_only variables) to model
    NeuronModels::IzhikevichVariable::VarValues izkVarVals(0.0, 0.0, 0.02, 0.2, -65.0, 8.);
    auto *pop = model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("Pop", 10, {}, izkVarVals);

    // Create custom update which tries to reduce V into A
    ReduceSharedVar::VarReferences reduceVarReferences(createVarRef(pop, "V"));
    ReduceSharedVar::VarValues reduceVars(0.0);
    auto *cu = model.addCustomUpdate<ReduceSharedVar>("Reduction", "CustomUpdate",
                                                      {}, reduceVars, reduceVarReferences);
    model.finalize();
    auto *cuInternal = static_cast<CustomUpdateInternal *>(cu);
    ASSERT_TRUE(cuInternal->isBatched());
    ASSERT_TRUE(cuInternal->isBatchReduction());
    ASSERT_FALSE(cuInternal->isNeuronReduction());
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, NeuronSharedCustomUpdateWU)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0},
        {}, {});

    Sum3::VarValues sumVarValues(0.0, 1.0);
    Sum3::WUVarReferences sumVarReferences(createWUVarRef(sg1, "g"), createWUVarRef(sg1, "g"));

    try {
        model.addCustomUpdate<Sum3>("SumWeight", "CustomUpdate",
                                   {}, sumVarValues, sumVarReferences);
        FAIL();
    }
    catch (const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, NeuronBatchReduction)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add neuron (arbitrary choice of model with read_only variables) to model
    NeuronModels::IzhikevichVariable::VarValues izkVarVals(0.0, 0.0, 0.02, 0.2, -65.0, 8.);
    auto *pop = model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("Pop", 10, {}, izkVarVals);

    // Create custom update which tries to reduce V into A
    ReduceDouble::VarReferences reduceVarReferences(createVarRef(pop, "V"), createVarRef(pop, "U"));
    ReduceDouble::VarValues reduceVars(0.0, 0.0);
    
    try {
        model.addCustomUpdate<ReduceDouble>("Reduction", "CustomUpdate",
                                            {}, reduceVars, reduceVarReferences);
        FAIL();
    }
    catch (const std::runtime_error &) {
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
    CustomUpdateInternal *sum1Internal = static_cast<CustomUpdateInternal*>(sum1);
    CustomUpdateInternal *sum2Internal = static_cast<CustomUpdateInternal*>(sum2);
    ASSERT_EQ(sum0Internal->getHashDigest(), sum1Internal->getHashDigest());
    ASSERT_NE(sum0Internal->getHashDigest(), sum2Internal->getHashDigest());
    ASSERT_EQ(sum0Internal->getInitHashDigest(), sum1Internal->getInitHashDigest());
    ASSERT_NE(sum0Internal->getInitHashDigest(), sum2Internal->getInitHashDigest());

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

    CustomUpdateInternal *sum0Internal = static_cast<CustomUpdateInternal*>(sum0);
    CustomUpdateInternal *sum1Internal = static_cast<CustomUpdateInternal*>(sum1);
    CustomUpdateInternal *sum2Internal = static_cast<CustomUpdateInternal*>(sum2);
    ASSERT_NE(sum0Internal->getHashDigest(), sum1Internal->getHashDigest());
    ASSERT_EQ(sum0Internal->getHashDigest(), sum2Internal->getHashDigest());
    ASSERT_EQ(sum0Internal->getInitHashDigest(), sum1Internal->getInitHashDigest());
    ASSERT_EQ(sum0Internal->getInitHashDigest(), sum2Internal->getInitHashDigest());

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
    CustomUpdateInternal *sum2Internal = static_cast<CustomUpdateInternal*>(sum2);
    CustomUpdateInternal *sum3Internal = static_cast<CustomUpdateInternal*>(sum3);
    CustomUpdateInternal *sum4Internal = static_cast<CustomUpdateInternal*>(sum4);

    ASSERT_NE(sum1Internal->getHashDigest(), sum2Internal->getHashDigest());
    ASSERT_NE(sum1Internal->getHashDigest(), sum3Internal->getHashDigest());
    ASSERT_NE(sum1Internal->getHashDigest(), sum4Internal->getHashDigest());

    // Delay groups don't matter for initialisation
    ASSERT_EQ(sum1Internal->getInitHashDigest(), sum2Internal->getInitHashDigest());
    ASSERT_EQ(sum1Internal->getInitHashDigest(), sum3Internal->getInitHashDigest());
    ASSERT_EQ(sum1Internal->getInitHashDigest(), sum4Internal->getInitHashDigest());
    
    ASSERT_EQ(sum2Internal->getHashDigest(), sum3Internal->getHashDigest());
    ASSERT_NE(sum2Internal->getHashDigest(), sum4Internal->getHashDigest());
    
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

    // Add one custom update which sums duplicated variables (v and u), another which sums shared variables (a and b) and another which sums one of each
    Sum::VarReferences sumVarReferences1(createVarRef(pop, "V"), createVarRef(pop, "U"));
    Sum::VarReferences sumVarReferences2(createVarRef(pop, "a"), createVarRef(pop, "b"));
    Sum::VarReferences sumVarReferences3(createVarRef(pop, "V"), createVarRef(pop, "a"));
    auto *sum1 = model.addCustomUpdate<Sum>("Sum1", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences1);
    auto *sum2 = model.addCustomUpdate<Sum>("Sum2", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences2);
    auto *sum3 = model.addCustomUpdate<Sum>("Sum3", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences3);
    model.finalize();

    // Check that sum1 and sum3 are batched and sum2 is not
    CustomUpdateInternal *sum1Internal = static_cast<CustomUpdateInternal*>(sum1);
    CustomUpdateInternal *sum2Internal = static_cast<CustomUpdateInternal*>(sum2);
    CustomUpdateInternal *sum3Internal = static_cast<CustomUpdateInternal*>(sum3);
    ASSERT_TRUE(sum1Internal->isBatched());
    ASSERT_FALSE(sum2Internal->isBatched());
    ASSERT_TRUE(sum3Internal->isBatched());

    // Check that neither initialisation nor update of batched and unbatched can be merged
    ASSERT_NE(sum1Internal->getHashDigest(), sum2Internal->getHashDigest());
    ASSERT_NE(sum1Internal->getInitHashDigest(), sum2Internal->getInitHashDigest());
    
    // Check that initialisation of batched and mixed can be merged but not update
    ASSERT_EQ(sum1Internal->getInitHashDigest(), sum3Internal->getInitHashDigest());
    ASSERT_NE(sum1Internal->getHashDigest(), sum3Internal->getHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    // **NOTE** delay groups don't matter for initialization
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateGroups().size() == 3);
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
    CustomUpdateWUInternal *sum2Internal = static_cast<CustomUpdateWUInternal*>(sum2);
    ASSERT_NE(sum1Internal->getHashDigest(), sum2Internal->getHashDigest());

    // Again, this doesn't matter for initialisation
    ASSERT_EQ(sum1Internal->getInitHashDigest(), sum2Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    // **NOTE** transpose variables don't matter for initialization
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateTransposeWUGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateWUGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedCustomWUUpdateInitGroups().size() == 1);
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
    CustomUpdateWUInternal *sum2Internal = static_cast<CustomUpdateWUInternal*>(sum2);
    ASSERT_NE(sum1Internal->getHashDigest(), sum2Internal->getHashDigest());
    ASSERT_NE(sum1Internal->getInitHashDigest(), sum2Internal->getInitHashDigest());
    
    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateTransposeWUGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateWUGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomWUUpdateInitGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomWUUpdateSparseInitGroups().size() == 1);
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, CompareDifferentWUBatched)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    // Add synapse group 
    StaticPulseDendriticDelaySplit::VarValues synVarInit(1.0, 1.0, 1.0, 1.0);
    auto *sg1 = model.addSynapsePopulation<StaticPulseDendriticDelaySplit, PostsynapticModels::DeltaCurr>(
        "Synapses", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, synVarInit,
        {}, {});

    // Add one custom update which sums duplicated variables (g and d), another which sums shared variables (gCommon and dCommon) and another which sums one of each
    Sum::WUVarReferences sumVarReferences1(createWUVarRef(sg1, "g"), createWUVarRef(sg1, "d"));
    Sum::WUVarReferences sumVarReferences2(createWUVarRef(sg1, "gCommon"), createWUVarRef(sg1, "dCommon"));
    Sum::WUVarReferences sumVarReferences3(createWUVarRef(sg1, "g"), createWUVarRef(sg1, "dCommon"));
    auto *sum1 = model.addCustomUpdate<Sum>("Sum1", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences1);
    auto *sum2 = model.addCustomUpdate<Sum>("Sum2", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences2);
    auto *sum3 = model.addCustomUpdate<Sum>("Sum3", "CustomUpdate",
                                            {}, {0.0}, sumVarReferences3);
    model.finalize();

    // Check that sum1 and sum3 are batched and sum2 is not
    CustomUpdateWUInternal *sum1Internal = static_cast<CustomUpdateWUInternal*>(sum1);
    CustomUpdateWUInternal *sum2Internal = static_cast<CustomUpdateWUInternal*>(sum2);
    CustomUpdateWUInternal *sum3Internal = static_cast<CustomUpdateWUInternal*>(sum3);
    ASSERT_TRUE(sum1Internal->isBatched());
    ASSERT_FALSE(sum2Internal->isBatched());
    ASSERT_TRUE(sum3Internal->isBatched());

    // Check that neither initialisation nor update of batched and unbatched can be merged
    ASSERT_NE(sum1Internal->getHashDigest(), sum2Internal->getHashDigest());
    ASSERT_NE(sum1Internal->getInitHashDigest(), sum2Internal->getInitHashDigest());
    
    // Check that initialisation of batched and mixed can be merged but not update
    ASSERT_EQ(sum1Internal->getInitHashDigest(), sum3Internal->getInitHashDigest());
    ASSERT_NE(sum1Internal->getHashDigest(), sum3Internal->getHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check correct groups are merged
    // **NOTE** delay groups don't matter for initialization
    ASSERT_TRUE(modelSpecMerged.getMergedCustomUpdateWUGroups().size() == 3);
    ASSERT_TRUE(modelSpecMerged.getMergedCustomWUUpdateInitGroups().size() == 2);
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, InvalidName)
{
    ModelSpec model;
    
     // Add neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neuron1", 10, paramVals, varVals);
    
    Sum::VarValues sumVarValues(0.0);
    Sum::VarReferences sumVarReferences1(createVarRef(ng1, "V"), createVarRef(ng1, "U"));

    try {
        model.addCustomUpdate<Sum>("Sum-1", "CustomUpdate",
                                   {}, sumVarValues, sumVarReferences1);
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomUpdates, InvalidUpdateGroupName)
{
    ModelSpec model;
    
     // Add neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neuron1", 10, paramVals, varVals);
    
    Sum::VarValues sumVarValues(0.0);
    Sum::VarReferences sumVarReferences1(createVarRef(ng1, "V"), createVarRef(ng1, "U"));

    try {
        model.addCustomUpdate<Sum>("Sum", "CustomUpdate-1",
                                   {}, sumVarValues, sumVarReferences1);
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
