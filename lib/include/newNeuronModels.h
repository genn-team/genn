#pragma once

// Standard includes
#include <array>
#include <string>
#include <tuple>
#include <vector>

// GeNN includes
#include "stringUtils.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_PARAM_VALUES(NUM_PARAMS) typedef NeuronModels::ValueBase<NUM_PARAMS> ParamValues
#define DECLARE_INIT_VALUES(NUM_INIT_VALUES) typedef NeuronModels::ValueBase<NUM_INIT_VALUES> InitValues

#define SET_SIM_CODE(SIM_CODE) virtual std::string GetSimCode() const{ return SIM_CODE; }
#define SET_THRESHOLD_CONDITION_CODE(THRESHOLD_CONDITION_CODE) virtual std::string GetThresholdConditionCode() const{ return THRESHOLD_CONDITION_CODE; }
#define SET_RESET_CODE(RESET_CODE) virtual std::string GetResetCode() const{ return RESET_CODE; }
#define SET_PARAM_NAMES(...) virtual std::vector<std::string> GetParamNames() const{ return __VA_ARGS__; }
#define SET_DERIVED_PARAM_NAMES(...) virtual std::vector<std::string> GetDerivedParamNames() const{ return __VA_ARGS__; }
#define SET_INIT_VALS(...) virtual std::vector<std::pair<std::string, std::string>> GetInitVals() const{ return __VA_ARGS__; }

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
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string GetSimCode() const{ return ""; }
    virtual std::string GetThresholdConditionCode() const{ return ""; }
    virtual std::string GetResetCode() const{ return ""; }

    virtual std::vector<std::string> GetParamNames() const{ return {}; }

    virtual std::vector<std::string> GetDerivedParamNames() const{ return {}; }

    virtual std::vector<std::pair<std::string, std::string>> GetInitVals() const{ return {}; }

    virtual double CalculateDerivedParam(int index, const vector<double> &pars, double dt = 1.0) const{ return -1; }
};

//----------------------------------------------------------------------------
// NeuronModels::BaseSingleton
//----------------------------------------------------------------------------
// Simple boilerplate class which implements singleton
// functionality using curiously recurring template pattern
template<typename Type>
class BaseSingleton : public Base
{
public:
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

template<typename Type>
Type *BaseSingleton<Type>::s_Instance = NULL;

//----------------------------------------------------------------------------
// NeuronModels::Izhikevich
//----------------------------------------------------------------------------
class Izhikevich : public BaseSingleton<Izhikevich>
{
public:
    DECLARE_PARAM_VALUES(4);
    DECLARE_INIT_VALUES(2);

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
class SpikeSource : public BaseSingleton<SpikeSource>
{
public:
    DECLARE_PARAM_VALUES(0);
    DECLARE_INIT_VALUES(0);

    SET_THRESHOLD_CONDITION_CODE("0");
};
} // NeuronModels