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
#define SET_SIM_CODE(SIM_CODE) virtual std::string GetSimCode() const{ return SIM_CODE; }
#define SET_THRESHOLD_CONDITION_CODE(THRESHOLD_CONDITION_CODE) virtual std::string GetThresholdConditionCode() const{ return THRESHOLD_CONDITION_CODE; }
#define SET_RESET_CODE(RESET_CODE) virtual std::string GetResetCode() const{ return RESET_CODE; }
#define SET_SUPPORT_CODE(SUPPORT_CODE) virtual std::string GetSupportCode() const{ return SUPPORT_CODE; }
#define SET_EXTRA_GLOBAL_PARAMS(...) virtual std::vector<std::pair<std::string, std::string>> GetExtraGlobalParams() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// NeuronModels::ValueBase
//----------------------------------------------------------------------------
namespace NeuronModels
{
//----------------------------------------------------------------------------
// NeuronModels::Base
//----------------------------------------------------------------------------
class Base : public NewModels::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string GetSimCode() const{ return ""; }
    virtual std::string GetThresholdConditionCode() const{ return ""; }
    virtual std::string GetResetCode() const{ return ""; }
    virtual std::string GetSupportCode() const{ return ""; }
    virtual std::vector<std::pair<std::string, std::string>> GetExtraGlobalParams() const{ return {}; }

    virtual bool IsPoisson() const{ return false; }
};

//----------------------------------------------------------------------------
// NeuronModels::LegacyWrapper
//----------------------------------------------------------------------------
// Wrapper around neuron models stored in global
class LegacyWrapper : public NewModels::LegacyWrapper<Base, neuronModel, nModels>
{
public:
    LegacyWrapper(unsigned int legacyTypeIndex) : NewModels::LegacyWrapper<Base, neuronModel, nModels>(legacyTypeIndex)
    {
    }

    //----------------------------------------------------------------------------
    // Base virtuals
    //----------------------------------------------------------------------------
    virtual std::string GetSimCode() const;
    virtual std::string GetThresholdConditionCode() const;
    virtual std::string GetResetCode() const;
    virtual std::string GetSupportCode() const;
    virtual std::vector<std::pair<std::string, std::string>> GetExtraGlobalParams() const;

    virtual bool IsPoisson() const;
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
    SET_INIT_VALS({{"V","scalar"}, {"U", "scalar"}});
};

//----------------------------------------------------------------------------
// NeuronModels::SpikeSource
//----------------------------------------------------------------------------
class SpikeSource : public Base
{
public:
    DECLARE_MODEL(NeuronModels::SpikeSource, 0, 0);

    SET_THRESHOLD_CONDITION_CODE("0");
};
} // NeuronModels