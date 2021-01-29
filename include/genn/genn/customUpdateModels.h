#pragma once

// GeNN includes
#include "gennExport.h"
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_CUSTOM_UPDATE_MODEL(TYPE, NUM_PARAMS, NUM_VARS, NUM_VAR_REFS)   \
    DECLARE_SNIPPET(TYPE, NUM_PARAMS);                                          \
    typedef Models::VarInitContainerBase<NUM_VARS> VarValues;                   \
    template<typename V>                                                        \
    using VarReferences = Snippet::InitialiserContainerBase<V, NUM_VAR_REFS>

#define SET_VAR_REFS(...) virtual VarRefVec getVarRefs() const override{ return __VA_ARGS__; }
#define SET_UPDATE_CODE(UPDATE_CODE) virtual std::string getUpdateCode() const override{ return UPDATE_CODE; }


//----------------------------------------------------------------------------
// CustomUpdateModels::Base
//----------------------------------------------------------------------------
namespace CustomUpdateModels
{
//! Base class for all current source models
class GENN_EXPORT Base : public Models::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets names and types (as strings) of model variable references
    virtual VarRefVec getVarRefs() const{ return {}; }

    //! Gets the code that performs the custom update 
    virtual std::string getUpdateCode() const{ return ""; }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Can this neuron model be merged with other? i.e. can they be simulated using same generated code
    bool canBeMerged(const Base *other) const;
};

//----------------------------------------------------------------------------
// CustomUpdateModels::AdamOptimizer
//----------------------------------------------------------------------------
//! Adam optimizer
/*! It has four parameters:
    - \c alpha    - 
    - \c beta1    - 
    - \c beta2    - 
    - \c alpha
*/
class AdamOptimizer : public Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(AdamOptimizer, 4, 2, 2);

    SET_UPDATE_CODE(
        "// Calculate moments **TODO** optimize out of loop\n"
        "const scalar firstMomentScale = 1.0 / (1.0 - pow($(beta1), $(t) + 1));\n"
        "const scalar secondMomentScale = 1.0 / (1.0 - pow($(beta2), $(t) + 1));\n"
        "// Update biased first moment estimate\n"
        "const scalar mT = ($(beta1) * $(m)) + ((1.0 - $(beta1)) * $(gradient));\n"
        "// Update biased second moment estimate\n"
        "const scalar vT = ($(beta2) * $(v)) + ((1.0 - $(beta2)) * $(gradient) * $(gradient));\n"
        "// Add gradient to variable, scaled by learning rate\n"
        "$(variable) -= ($(alpha) * mT * firstMomentScale) / (sqrt(vT * secondMomentScale) + $(epsilon));\n"
        "// Zero gradient\n"
        "$(gradient) = 0.0;\n");

    SET_PARAM_NAMES({"alpha", "beta1", "beta2", "epsilon"});
    SET_VARS({{"m", "scalar"}, {"v", "scalar"}});
    SET_VAR_REFS({{"gradient", "scalar", VarAccessMode::READ_WRITE}, 
                  {"variable", "scalar", VarAccessMode::READ_WRITE}});
};
}   // CustomUpdateModels

