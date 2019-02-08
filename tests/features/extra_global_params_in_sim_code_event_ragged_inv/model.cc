//--------------------------------------------------------------------------
/*! \file extra_global_params_in_sim_code_event_sparse_inv/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 2);

    SET_SIM_CODE("$(x)= $(t)+$(shift);\n");

    SET_VARS({{"x", "scalar"}, {"shift", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);

//----------------------------------------------------------------------------
// WeightUpdateModel
//----------------------------------------------------------------------------
class WeightUpdateModel : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(WeightUpdateModel, 0, 1);

    SET_VARS({{"w", "scalar"}});
    SET_EXTRA_GLOBAL_PARAMS({{"thresh", "scalar"}});

    SET_EVENT_THRESHOLD_CONDITION_CODE("(fmod($(x_pre),$(thresh)) < 1e-4)");
    SET_EVENT_CODE("$(w)= $(x_pre);");
};

IMPLEMENT_MODEL(WeightUpdateModel);

void modelDefinition(NNmodel &model)
{
    model.setDT(0.1);
    model.setName("extra_global_params_in_sim_code_event_sparse_inv");

    model.addNeuronPopulation<Neuron>("pre", 10, {}, Neuron::VarValues(0.0, 0.0));
    model.addNeuronPopulation<Neuron>("post", 10, {}, Neuron::VarValues(0.0, 0.0));

    std::string synName= "syn";
    for (int i= 0; i < 10; i++)
    {
        string theName= synName + std::to_string(i);
        model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::DeltaCurr>(
            theName, SynapseMatrixType::SPARSE_INDIVIDUALG, i, "pre", "post",
            {}, WeightUpdateModel::VarValues(0.0),
            {}, {});
        model.setSpanTypeToPre(theName);
    }

    model.setPrecision(GENN_FLOAT);
}
