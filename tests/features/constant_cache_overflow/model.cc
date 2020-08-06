//--------------------------------------------------------------------------
/*! \file constant_cache_overflow/model.cc

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
    DECLARE_MODEL(Neuron, 0, 1);

    SET_SIM_CODE("$(x)= $(gennrand_uniform);\n");

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);


void modelDefinition(ModelSpec &model)
{
    GENN_PREFERENCES.generateEmptyStatePushPull = false;

    model.setDT(0.1);
    model.setName("constant_cache_overflow");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    
    // Each neuron group structure requires 40 bytes, 
    // 8 bytes is required for spike queue updates and 4 for indices = 52
    // Therefore floor(65536 / 52) = 1260 should naively fit in constant cache with 16 bytes to spare
    for(unsigned int i = 0; i < 1260; i++) {
        model.addNeuronPopulation<Neuron>("Pop" + std::to_string(i), 10, {}, Neuron::VarValues(0.0));
    }

    model.setPrecision(GENN_FLOAT);
}
