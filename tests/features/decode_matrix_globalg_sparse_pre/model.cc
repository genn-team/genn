#include "modelSpec.h"

// NEURONS
//==============

double neuron_ini[1] = { // one neuron variable
    0.0  // 0 - individual shift
};


// Synapses
//==================================================

double synapses_ini[1]= {
    1.0 // the weight
};


void modelDefinition(NNmodel &model)
{
    initGeNN();

    model.setDT(0.1);
    model.setName("decode_matrix_globalg_sparse_pre");

    neuronModel n;
    n.varNames = {"x", "shift"};
    n.varTypes = {"scalar", "scalar"};
    n.simCode= "$(x)= $(Isyn);\n";

    const int DUMMYNEURON= nModels.size();
    nModels.push_back(n);

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(1.0);    // 0 - Wij (nA)

    model.addNeuronPopulation("Pre", 10, SPIKESOURCE, NULL, NULL);
    model.addNeuronPopulation("Post", 4, DUMMYNEURON, NULL, neuron_ini);


    model.addSynapsePopulation("Syn", NSYNAPSE, SPARSE, GLOBALG, NO_DELAY, IZHIKEVICH_PS, "Pre", "Post",
                               synapses_ini, NULL,
                               NULL, NULL);
    model.setSpanTypeToPre("Syn");
    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
