#include "modelSpec.h"

// NEURONS
//==============

double neuron_ini[1] = { // one neuron variable
    0.0  // 0 - individual shift
};

void modelDefinition(NNmodel &model)
{
    initGeNN();

    model.setDT(0.1);
    model.setName("neuron_rng_uniform");

    neuronModel n;
    n.varNames = {"x", "shift"};
    n.varTypes = {"scalar", "scalar"};
    n.simCode= "$(x)= $(gennrand_uniform);\n";

    const int DUMMYNEURON= nModels.size();
    nModels.push_back(n);

    model.addNeuronPopulation("Pop", 1000, DUMMYNEURON, NULL, neuron_ini);

    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
