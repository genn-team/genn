//--------------------------------------------------------------------------
/*! \file extra_global_param_ref/model.cc

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

    SET_SIM_CODE("$(x) = $(e)[$(id)];\n");

    SET_VARS({{"x", "scalar"}});
    SET_EXTRA_GLOBAL_PARAMS({{"e", "scalar*"}});
};
IMPLEMENT_MODEL(Neuron);

//----------------------------------------------------------------------------
// CU
//----------------------------------------------------------------------------
class CU : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL_EGP_REF(CU, 0, 0, 1, 1);
    SET_UPDATE_CODE("if($(id) == (int)round(fmod($(t), 10.0))) {\n"
                    "   $(e)[$(id)] = 1.0;\n"
                    "}\n"
                    "else {\n"
                    "   $(e)[$(id)] = 0.0;\n"
                    "}");
    SET_VAR_REFS({{"v", "scalar"}})
    SET_EXTRA_GLOBAL_PARAM_REFS({{"e", "scalar*"}});
};
IMPLEMENT_MODEL(CU);


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
    model.setDT(0.1);
    model.setName("extra_global_param_ref");

    auto *pop = model.addNeuronPopulation<Neuron>("pop", 10, {}, Neuron::VarValues(0.0));

    CU::VarReferences cuVarRefs(createVarRef(pop, "x"));
    CU::EGPReferences cuEGPRefs(createEGPRef(pop, "e"));
    model.addCustomUpdate<CU>("CU", "CustomUpdate",
                              CU::ParamValues{}, CU::VarValues{}, cuVarRefs, cuEGPRefs);

    model.setPrecision(GENN_FLOAT);
}
