#pragma once

// Standard includes
#include <array>
#include <tuple>
#include <vector>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define IMPLEMENT_NEURON(TYPE, NUM_PARAMs, NUM_INIT_VALUES)                                           \
private:                                                                                              \
    static const char *s_SimCode;                                                                     \
    static const char *s_ThresholdConditionCode;                                                      \
    static const char *s_ParamNames[NUM_PARAMS];                                                      \
    static const char *s_InitValueNames[NUM_INIT_VALUES];                                             \
    static const char *s_InitValueTypes[NUM_INIT_VALUES];                                             \
    static TYPE *s_Instance;                                                                          \
public:                                                                                               \
    virtual const char *GetSimCode() const { return TYPE::s_SimCode; }                                \
    virtual const char *GetThresholdConditionCode() const { return TYPE::s_ThresholdConditionCode; }  \
    virtual size_t GetNumParams() const { return NUM_PARAMs; }                                        \
    virtual const char *GetParamName(size_t i) const { return s_ParamNames[i]; }                      \
    virtual size_t GetNumInitValues() const { return NUM_INIT_VALUES; }                               \
    virtual const char *GetInitValueName(size_t i) const { return s_InitValueNames[i]; }              \
    virtual const char *GetInitValueType(size_t i) const { return s_InitValueTypes[i]; }              \
    static const TYPE *GetInstance()                                                                  \
    {                                                                                                 \
        if(s_Instance == NULL)                                                                        \
        {                                                                                             \
            s_Instance = new TYPE;                                                                    \
        }                                                                                             \
        return s_Instance;                                                                            \
    }


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
    void GetValues() const
    {
        return std::vector<double>(m_Values);
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
    virtual const char *GetSimCode() const = 0;
    virtual const char *GetThresholdConditionCode() const = 0;

    virtual size_t GetNumParams() const = 0;
    virtual const char *GetParamName(size_t i) const = 0;

    virtual size_t GetNumInitValues() const = 0;
    virtual const char *GetInitValueName(size_t i) const = 0;
    virtual const char *GetInitValueType(size_t i) const = 0;
};

//----------------------------------------------------------------------------
// NeuronModels::Izhikevich
//----------------------------------------------------------------------------
class Izhikevich : public Base
{
public:
    IMPLEMENT_NEURON(Izhikevich, 4, 2);

    //--------------------------------------------------------------------------
    // ParamValues
    //--------------------------------------------------------------------------
    class ParamValues : public ValueBase<4>
    {
    public:
        ParamValues(double a, double b, double c, double d) : ParamBase({a, b, c, d})
        {
        }
    };

    //--------------------------------------------------------------------------
    // InitValues
    //--------------------------------------------------------------------------
    class InitValues : public ValueBase<2>
    {
    public:
        InitValues(double v, double u) : ParamBase({v, u})
        {
        }
    };

private:
  Izhikevich()
  {
  }
};

//----------------------------------------------------------------------------
// NeuronModels::IzhikevichVS
//----------------------------------------------------------------------------
class IzhikevichV : public Base
{
public:
    IMPLEMENT_NEURON(IzhikevichV, 1, 6);

    //--------------------------------------------------------------------------
    // ParamValues
    //--------------------------------------------------------------------------
    class ParamValues : public ValueBase<1>
    {
    public:
        ParamValues(double iOffset) : ParamBase({iOffset})
        {
        }
    };

    //--------------------------------------------------------------------------
    // InitValues
    //--------------------------------------------------------------------------
    class InitValues : public ValueBase<6>
    {
    public:
        InitValues(double v, double u, double a, double b, double c, double d) : ParamBase({v, u, a, b, c, d})
        {
        }
    };

private:
  IzhikevichV()
  {
  }
};



} // NeuronModels