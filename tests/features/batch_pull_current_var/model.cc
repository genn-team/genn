//--------------------------------------------------------------------------
/*! \file batch_pull_current_var/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// PreNeuron
//----------------------------------------------------------------------------
class PreNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PreNeuron, 0, 1);
    
    SET_SIM_CODE(
        "const unsigned int timestep = (unsigned int)round($(t));\n"
        "$(x)= (float)((timestep * 100) + ($(batch) * 10) + $(id));\n");
    
    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(PreNeuron);

//----------------------------------------------------------------------------
// PostNeuron
//----------------------------------------------------------------------------
class PostNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PostNeuron, 0, 1);

    SET_SIM_CODE("$(x)= $(Isyn);\n");

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(PostNeuron);

//---------------------------------------------------------------------------
// Continuous
//---------------------------------------------------------------------------
class Continuous : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Continuous, 0, 1);

    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(g) * $(x_pre));\n");
};
IMPLEMENT_MODEL(Continuous);

void modelDefinition(ModelSpec &model)
{
#ifdef CL_HPP_TARGET_OPENCL_VERSION
    if(std::getenv("OPENCL_DEVICE") != nullptr) {
        GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
        GENN_PREFERENCES.manualDeviceID = std::atoi(std::getenv("OPENCL_DEVICE"));
    }
    if(std::getenv("OPENCL_PLATFORM") != nullptr) {
        GENN_PREFERENCES.manualPlatformID = std::atoi(std::getenv("OPENCL_PLATFORM"));
    }
#endif
    model.setBatchSize(10);
    model.setDT(1.0);
    model.setName("batch_pull_current_var");
    
    
    model.addNeuronPopulation<PreNeuron>("Pop", 10, {}, {0.0});
    model.addNeuronPopulation<PreNeuron>("PopDelay", 10, {}, {0.0});
    model.addNeuronPopulation<PostNeuron>("Post", 10, {}, {0.0});
    
    model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "PopDelay_Post", SynapseMatrixType::DENSE_GLOBALG, 5,
        "PopDelay", "Post",
        {}, {1.0},
        {}, {});
}
