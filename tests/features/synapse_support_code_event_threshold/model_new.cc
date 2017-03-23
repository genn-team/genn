#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 2);

    SET_SIM_CODE("$(x)= $(t)+$(shift);\n");

    SET_THRESHOLD_CONDITION_CODE("(fmod($(x),1.0) < 1e-4)");

    SET_VARS({{"x", "scalar"}, {"shift", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);

//----------------------------------------------------------------------------
// WeightUpdateModel
//----------------------------------------------------------------------------
class WeightUpdateModel : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(WeightUpdateModel, 1, 1);

    SET_VARS({{"w", "scalar"}});
    SET_PARAM_NAMES({"myTrigger"});

    SET_SIM_SUPPORT_CODE("__device__ __host__ bool checkThreshold(scalar x, scalar trigger){ return (fmod(x, trigger) < 1e-4); }");
    SET_EVENT_THRESHOLD_CONDITION_CODE("checkThreshold($(x_pre),$(myTrigger))");
    SET_EVENT_CODE("$(w)= $(x_pre);");
};

IMPLEMENT_MODEL(WeightUpdateModel);

void modelDefinition(NNmodel &model)
{
    initGeNN();

    model.setDT(0.1);
    model.setName("synapse_support_code_event_threshold_new");

    model.addNeuronPopulation<Neuron>("pre", 10, {}, Neuron::VarValues(0.0, 0.0));
    model.addNeuronPopulation<Neuron>("post", 10, {}, Neuron::VarValues(0.0, 0.0));

    string synName= "syn";
    for (int i= 0; i < 10; i++)
    {
        string theName= synName + std::to_string(i);
        model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::Izhikevich>(
            theName, SynapseMatrixType::DENSE_INDIVIDUALG, i, "pre", "post",
            WeightUpdateModel::ParamValues((double)2*(i+1)), WeightUpdateModel::VarValues(0.0),
            {}, {});
    }
    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
