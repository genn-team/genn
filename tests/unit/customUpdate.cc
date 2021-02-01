// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

//--------------------------------------------------------------------------
// Tests
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

    model.addCustomUpdate<Sum>("SumWeight1", "CustomUpdate", CustomUpdateWU::Operation::UPDATE,
                               {}, sumVarValues, sumVarReferences1);
    
    try {
        model.addCustomUpdate<Sum>("SumWeight2", "CustomUpdate", CustomUpdateWU::Operation::UPDATE,
                                   {}, sumVarValues, sumVarReferences2);
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
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

    model.addCustomUpdate<Sum>("SumWeight1", "CustomUpdate", CustomUpdateWU::Operation::UPDATE,
                               {}, sumVarValues, sumVarReferences1);
    
    try {
        model.addCustomUpdate<Sum>("SumWeight2", "CustomUpdate", CustomUpdateWU::Operation::UPDATE,
                                   {}, sumVarValues, sumVarReferences2);
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}