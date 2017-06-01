#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 1, 2);

    SET_SIM_CODE("$(x)= $(t)+$(shift);\n");

    SET_THRESHOLD_CONDITION_CODE("(fmod($(x),$(ISI)) < 1e-4)");

    SET_PARAM_NAMES({"ISI"});
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

    SET_LEARN_POST_SUPPORT_CODE("__device__ __host__ scalar getWeight(scalar x){ return x; }");
    SET_LEARN_POST_CODE("$(w)= getWeight($(x_pre));");
};

IMPLEMENT_MODEL(WeightUpdateModel);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(0.1);
    model.setName("synapse_support_code_post_learn_new");

    model.addNeuronPopulation<Neuron>("pre", 10, Neuron::ParamValues(1.0), Neuron::VarValues(0.0, 0.0));
    model.addNeuronPopulation<Neuron>("post", 10, Neuron::ParamValues(2.0), Neuron::VarValues(0.0, 0.0));

    string synName= "syn";
    for (int i= 0; i < 10; i++)
    {
        string theName= synName + std::to_string(i);
        model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::DeltaCurr>(
            theName, SynapseMatrixType::DENSE_INDIVIDUALG, i, "pre", "post",
            {}, WeightUpdateModel::VarValues(0.0),
            {}, {});
    }
    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
