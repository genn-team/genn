#pragma once

// Standard includes
#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

// GeNN includes
#include "codeGenUtils.h"
#include "neuronModels.h"
#include "newModels.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_SIM_CODE(SIM_CODE) virtual std::string getSimCode() const{ return SIM_CODE; }
#define SET_THRESHOLD_CONDITION_CODE(THRESHOLD_CONDITION_CODE) virtual std::string getThresholdConditionCode() const{ return THRESHOLD_CONDITION_CODE; }
#define SET_RESET_CODE(RESET_CODE) virtual std::string getResetCode() const{ return RESET_CODE; }
#define SET_SUPPORT_CODE(SUPPORT_CODE) virtual std::string getSupportCode() const{ return SUPPORT_CODE; }
#define SET_EXTRA_GLOBAL_PARAMS(...) virtual StringPairVec getExtraGlobalParams() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// NeuronModels::Base
//----------------------------------------------------------------------------
namespace NeuronModels
{
//! Base class for all neuron models
class Base : public NewModels::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets the code that defines the execution of one timestep of integration of the neuron model.
    /*! The code will refer to $(NN) for the value of the variable with name "NN".
        It needs to refer to the predefined variable "ISYN", i.e. contain $(ISYN), if it is to receive input. */
    virtual std::string getSimCode() const{ return ""; }

    //! Gets code which defines the condition for a true spike in the described neuron model.
    /*! This evaluates to a bool (e.g. "V > 20"). */
    virtual std::string getThresholdConditionCode() const{ return ""; }

    //! Gets code that defines the reset action taken after a spike occurred. This can be empty
    virtual std::string getResetCode() const{ return ""; }

    //! Gets support code to be made available within the neuron kernel/funcion.
    /*! This is intended to contain user defined device functions that are used in the neuron codes.
        Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef;
        functions should be declared as "__host__ __device__" to be available for both GPU and CPU versions. */
    virtual std::string getSupportCode() const{ return ""; }

    //! Gets names and types (as strings) of additional
    //! per-population parameters for the weight update model.
    virtual std::vector<std::pair<std::string, std::string>> getExtraGlobalParams() const{ return {}; }

    //! Is this neuron model the internal Poisson model (which requires a number of special cases)
    //! \private
    virtual bool isPoisson() const{ return false; }
};

//----------------------------------------------------------------------------
// NeuronModels::LegacyWrapper
//----------------------------------------------------------------------------
//! Wrapper around legacy weight update models stored in #nModels array of neuronModel objects.
class LegacyWrapper : public NewModels::LegacyWrapper<Base, neuronModel, nModels>
{
public:
    LegacyWrapper(unsigned int legacyTypeIndex) : NewModels::LegacyWrapper<Base, neuronModel, nModels>(legacyTypeIndex)
    {
    }

    //----------------------------------------------------------------------------
    // Base virtuals
    //----------------------------------------------------------------------------

    //! \copydoc Base::getSimCode
    virtual std::string getSimCode() const;

    //! \copydoc Base::getThresholdConditionCode
    virtual std::string getThresholdConditionCode() const;

    //! \copydoc Base::getResetCode
    virtual std::string getResetCode() const;

    //! \copydoc Base::getSupportCode
    virtual std::string getSupportCode() const;

    //! \copydoc Base::getExtraGlobalParams
    virtual NewModels::Base::StringPairVec getExtraGlobalParams() const;

    //! \copydoc Base::isPoisson
    virtual bool isPoisson() const;
};

//----------------------------------------------------------------------------
// NeuronModels::Izhikevich
//----------------------------------------------------------------------------
class Izhikevich : public Base
{
public:
    DECLARE_MODEL(NeuronModels::Izhikevich, 4, 2);

    SET_SIM_CODE(
        "    if ($(V) >= 30.0){\n"
        "      $(V)=$(c);\n"
        "                  $(U)+=$(d);\n"
        "    } \n"
        "    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability\n"
        "    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;\n"
        "    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
        "   //if ($(V) > 30.0){   //keep this only for visualisation -- not really necessaary otherwise \n"
        "   //  $(V)=30.0; \n"
        "   //}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(V) >= 29.99");

    SET_PARAM_NAMES({"a", "b", "c", "d"});
    SET_VARS({{"V","scalar"}, {"U", "scalar"}});
};

//----------------------------------------------------------------------------
// NeuronModels::SpikeSource
//----------------------------------------------------------------------------
//! Empty neuron which allows setting spikes from external sources
/*! This model does not contain any update code and can be used to implement
    the equivalent of a SpikeGeneratorGroup in Brian or a SpikeSourceArray in PyNN. */
class SpikeSource : public Base
{
public:
    DECLARE_MODEL(NeuronModels::SpikeSource, 0, 0);

    SET_THRESHOLD_CONDITION_CODE("0");
};

//----------------------------------------------------------------------------
// NeuronModels::Poisson
//----------------------------------------------------------------------------
class Poisson : public Base
{
public:
    DECLARE_MODEL(NeuronModels::Poisson, 4, 3);

    SET_SIM_CODE(
        "uint64_t theRnd;\n"
        "if ($(V) > $(Vrest)) {\n"
        "   $(V)= $(Vrest);\n"
        "}"
        "else if ($(t) - $(spikeTime) > ($(trefract))) {\n"
        "   MYRAND($(seed),theRnd);\n"
        "   if (theRnd < *($(rates)+$(offset)+$(id))) {\n"
        "       $(V)= $(Vspike);\n"
        "       $(spikeTime)= $(t);\n"
        "   }\n"
        "}\n"
    );
    SET_THRESHOLD_CONDITION_CODE("$(V) >= $(Vspike)");

    SET_PARAM_NAMES({"therate", "trefract", "Vspike", "Vrest"});
    SET_VARS({{"V", "scalar"}, {"seed", "uint64_t"}, {"spikeTime", "scalar"}});
    SET_EXTRA_GLOBAL_PARAMS({{"rates", "uint64_t *"}, {"offset", "unsigned int"}});

    virtual bool isPoisson() const{ return true; }
};
} // NeuronModels