#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 2);

    SET_SUPPORT_CODE("__device__ __host__ scalar calcActivation(scalar t, scalar shift){ return (t + shift); }");

    SET_SIM_CODE("$(x) = calcActivation($(t), $(shift));\n");

    SET_THRESHOLD_CONDITION_CODE("(fmod($(x), 1.0) < 1e-4)");

    SET_INIT_VALS({{"x", "scalar"}, {"shift", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);

//----------------------------------------------------------------------------
// WeightUpdateModel
//----------------------------------------------------------------------------
class WeightUpdateModel : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(WeightUpdateModel, 0, 1);

    SET_INIT_VALS({{"w", "scalar"}});

    SET_SIM_CODE("$(w)= $(x_pre);");
};

IMPLEMENT_MODEL(WeightUpdateModel);

void modelDefinition(NNmodel &model)
{
    initGeNN();

    model.setDT(0.1);

    model.addNeuronPopulation<Neuron>("pre", 10, {}, Neuron::InitValues(0.0, 0.0));
    model.addNeuronPopulation<Neuron>("post", 10, {}, Neuron::InitValues(0.0, 0.0));

    string synName= "syn";
    for (int i= 0; i < 10; i++)
    {
        string theName= synName + std::to_string(i);
        model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::Izhikevich>(
            theName, DENSE, INDIVIDUALG, i,"pre", "post",
            {}, WeightUpdateModel::InitValues(0.0),
            {}, {});
    }
    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
