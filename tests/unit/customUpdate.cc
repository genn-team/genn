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
    Sum::VarReferences sumVarReferences1(wuVarReference(sg1, "g"), wuVarReference(sg1, "g"));
    Sum::VarReferences sumVarReferences2(wuVarReference(sg1, "g"), wuVarReference(sg1, "d"));

    model.addCustomUpdate<Sum>("SumWeight1", "CustomUpdate", CustomUpdate::Operation::UPDATE,
                               {}, sumVarValues, sumVarReferences1);
    
    try {
        model.addCustomUpdate<Sum>("SumWeight2", "CustomUpdate", CustomUpdate::Operation::UPDATE,
                                   {}, sumVarValues, sumVarReferences2);
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
    // Finalize model
    model.finalize();


    
}