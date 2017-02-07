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
#define DECLARE_NEURON()                                           \
private:                                                                                              \
    static std::string s_SimCode;                                                                     \
    static std::string s_ThresholdConditionCode;                                                      \
    static std::string s_ResetCode; \
    static std::vector<std::string> s_ParamNames;\
    static std::vector<std::pair<std::string, std::string>> s_InitVals;\
public:                                                                                               \
    virtual const std::string &GetSimCode() const{ return s_SimCode; } \
    virtual const std::string &GetThresholdConditionCode() const{ return s_ThresholdConditionCode; } \
    virtual const std::string &GetResetCode() const{ return s_ResetCode; }\
    virtual const std::vector<std::string> &GetParamNames() const{ return s_ParamNames; }\
    virtual const std::vector<std::pair<std::string, std::string>> &GetInitVals() const{ return s_InitVals; } \

//----------------------------------------------------------------------------
// NeuronModels::ValueBase
//----------------------------------------------------------------------------
namespace NeuronModels
{
template<size_t NumValues>
class ValueBase
{
public:
    ValueBase(const std::array<double, NumValues> &values) : m_Values(values)
    {
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
    virtual const std::string &GetSimCode() const = 0;
    virtual const std::string &GetThresholdConditionCode() const = 0;
    virtual const std::string &GetResetCode() const = 0;

    virtual const std::vector<std::string> &GetParamNames() const = 0;

    virtual const std::vector<std::pair<std::string, std::string>> &GetInitVals() const = 0;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    PairStringKeyConstIter GetInitValNamesCBegin() const
    {
      return GetInitVals().cbegin();
    }

    PairStringKeyConstIter GetInitValNamesCEnd() const
    {
      return GetInitVals().cend();
    }
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
    DECLARE_NEURON();

    //--------------------------------------------------------------------------
    // ParamValues
    //--------------------------------------------------------------------------
    class ParamValues : public ValueBase<4>
    {
    public:
        ParamValues(double a, double b, double c, double d) : ValueBase<4>({a, b, c, d})
        {
        }
    };

    //--------------------------------------------------------------------------
    // InitValues
    //--------------------------------------------------------------------------
    class InitValues : public ValueBase<2>
    {
    public:
        InitValues(double v, double u) : ValueBase<2>({v, u})
        {
        }
    };
};
} // NeuronModels