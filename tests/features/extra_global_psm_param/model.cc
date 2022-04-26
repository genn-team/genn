//--------------------------------------------------------------------------
/*! \file extra_global_psm_param/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// Pre
//----------------------------------------------------------------------------
class Pre : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Pre, 0, 1);

    SET_SIM_CODE("$(x)= $(t);\n");

    SET_VARS({{"x", "scalar"}});
};
IMPLEMENT_MODEL(Pre);

//----------------------------------------------------------------------------
// Post
//----------------------------------------------------------------------------
class Post : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Post, 0, 1);

    SET_SIM_CODE("$(x)= $(Isyn);\n");

    SET_VARS({{"x", "scalar"}});
};
IMPLEMENT_MODEL(Post);

//----------------------------------------------------------------------------
// WUM
//----------------------------------------------------------------------------
class WUM : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WUM, 0, 0, 0, 0);
    
    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(x_pre));\n");
};
IMPLEMENT_MODEL(WUM);

//----------------------------------------------------------------------------
// PSM
//----------------------------------------------------------------------------
class PSM : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(PSM, 0, 0);
    
    SET_APPLY_INPUT_CODE("const scalar scale = ($(id) == (int)$(k)) ? 1.0 : 0.0; $(Isyn) += ($(inSyn) * scale);\n");
    SET_DECAY_CODE("$(inSyn) = 0;");
    SET_EXTRA_GLOBAL_PARAMS({{"k", "unsigned int"}});
};
IMPLEMENT_MODEL(PSM);


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
  model.setName("extra_global_psm_param");

  model.addNeuronPopulation<Pre>("pre", 10, {}, Pre::VarValues(0.0));
  model.addNeuronPopulation<Post>("post", 10, {}, Post::VarValues(0.0));
  model.addSynapsePopulation<WUM, PSM>("syn", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY, "pre", "post",
                                       {}, {},
                                       {}, {},
                                       initConnectivity<InitSparseConnectivitySnippet::OneToOne>());
  
  model.setPrecision(GENN_FLOAT);
}
