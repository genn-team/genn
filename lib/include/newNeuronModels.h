#pragma once

// Standard includes
#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

// GeNN includes
#include "stringUtils.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_SIM_CODE(SIM_CODE) virtual std::string GetSimCode() const{ return SIM_CODE; }
#define SET_THRESHOLD_CONDITION_CODE(THRESHOLD_CONDITION_CODE) virtual std::string GetThresholdConditionCode() const{ return THRESHOLD_CONDITION_CODE; }
#define SET_RESET_CODE(RESET_CODE) virtual std::string GetResetCode() const{ return RESET_CODE; }
#define SET_SUPPORT_CODE(SUPPORT_CODE) virtual std::string GetSupportCode() const{ return SUPPORT_CODE; }

#define SET_PARAM_NAMES(...) virtual std::vector<std::string> GetParamNames() const{ return __VA_ARGS__; }
#define SET_DERIVED_PARAMS(...) virtual std::vector<std::pair<std::string, DerivedParamFunc>> GetDerivedParams() const{ return __VA_ARGS__; }
#define SET_INIT_VALS(...) virtual std::vector<std::pair<std::string, std::string>> GetInitVals() const{ return __VA_ARGS__; }
#define SET_EXTRA_GLOBAL_PARAMS(...) virtual std::vector<std::pair<std::string, std::string>> GetExtraGlobalParams() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// NeuronModels::ValueBase
//----------------------------------------------------------------------------
namespace NeuronModels
{
template<size_t NumValues>
class ValueBase
{
public:
    // **NOTE** other less terrifying forms of constructor won't complain at compile time about
    // number of parameters e.g. std::array<double, 4> can be initialized with <= 4 elements
    template<typename... T>
    ValueBase(T&&... vals) : m_Values{std::forward<double>(vals)...}
    {
        static_assert(sizeof...(vals) == NumValues, "Wrong number of values");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::vector<double> GetValues() const
    {
        return std::vector<double>(m_Values.cbegin(), m_Values.cend());
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::array<double, NumValues> m_Values;
};

//----------------------------------------------------------------------------
// NeuronModels::Base
//----------------------------------------------------------------------------
class Base
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::function<double(const vector<double> &pars, double)> DerivedParamFunc;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string GetSimCode() const{ return ""; }
    virtual std::string GetThresholdConditionCode() const{ return ""; }
    virtual std::string GetResetCode() const{ return ""; }
    virtual std::string GetSupportCode() const{ return ""; }

    virtual std::vector<std::string> GetParamNames() const{ return {}; }
    virtual std::vector<std::pair<std::string, DerivedParamFunc>> GetDerivedParams() const{ return {}; }
    virtual std::vector<std::pair<std::string, std::string>> GetInitVals() const{ return {}; }
    virtual std::vector<std::pair<std::string, std::string>> GetExtraGlobalParams() const{ return {}; }
};

//----------------------------------------------------------------------------
// NeuronModels::LegacyWrapper
//----------------------------------------------------------------------------
// Wrapper around neuron models stored in global
class LegacyWrapper : public Base
{
public:
    LegacyWrapper(int legacyTypeIndex) : m_LegacyTypeIndex(legacyTypeIndex)
    {
    }

    //----------------------------------------------------------------------------
    // Base virtuals
    //----------------------------------------------------------------------------
    virtual std::string GetSimCode() const;
    virtual std::string GetThresholdConditionCode() const;
    virtual std::string GetResetCode() const;
    virtual std::string GetSupportCode() const;

    virtual std::vector<std::string> GetParamNames() const;

    virtual std::vector<std::pair<std::string, DerivedParamFunc>> GetDerivedParams() const;

    virtual std::vector<std::pair<std::string, std::string>> GetInitVals() const;
    virtual std::vector<std::pair<std::string, std::string>> GetExtraGlobalParams() const;

private:
    //----------------------------------------------------------------------------
    // Static methods
    //----------------------------------------------------------------------------
    static std::vector<std::pair<std::string, std::string>> ZipStringVectors(const std::vector<std::string> &a,
                                                                             const std::vector<std::string> &b);
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const int m_LegacyTypeIndex;
};

//----------------------------------------------------------------------------
// NeuronModels::BaseSingleton
//----------------------------------------------------------------------------
// Simple boilerplate class which implements singleton
// functionality using curiously recurring template pattern
template<typename Type, unsigned int NumParamVals, unsigned int NumInitVals>
class BaseSingleton : public Base
{
public:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef NeuronModels::ValueBase<NumParamVals> ParamValues;
    typedef NeuronModels::ValueBase<NumInitVals> InitValues;

    //------------------------------------------------------------------------
    // Static methods
    //------------------------------------------------------------------------
    static const Type *GetInstance()
    {
        if(s_Instance == NULL)
        {
            s_Instance = new Type;
        }
        return s_Instance;
    }

private:
    //------------------------------------------------------------------------
    // Static members
    //------------------------------------------------------------------------
    static Type *s_Instance;
};

template<typename Type, unsigned int NumParamVals, unsigned int NumInitVals>
Type *BaseSingleton<Type, NumParamVals, NumInitVals>::s_Instance = NULL;

//----------------------------------------------------------------------------
// NeuronModels::Izhikevich
//----------------------------------------------------------------------------
class Izhikevich : public BaseSingleton<Izhikevich, 4, 2>
{
public:
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
class SpikeSource : public BaseSingleton<SpikeSource, 0, 0>
{
public:
    SET_THRESHOLD_CONDITION_CODE("0");
};
} // NeuronModels