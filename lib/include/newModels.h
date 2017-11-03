#pragma once

// Standard includes
#include <array>
#include <functional>
#include <string>
#include <vector>

#include <cassert>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_MODEL(TYPE, NUM_PARAMS, NUM_VARS)              \
private:                                                       \
    static TYPE *s_Instance;                                   \
public:                                                        \
    static const TYPE *getInstance()                           \
    {                                                          \
        if(s_Instance == NULL)                                 \
        {                                                      \
            s_Instance = new TYPE;                             \
        }                                                      \
        return s_Instance;                                     \
    }                                                          \
    typedef NewModels::ValueBase<NUM_PARAMS> ParamValues;      \
    typedef NewModels::ValueBase<NUM_VARS> VarValues;          \


#define IMPLEMENT_MODEL(TYPE) TYPE *TYPE::s_Instance = NULL

#define SET_PARAM_NAMES(...) virtual StringVec getParamNames() const{ return __VA_ARGS__; }
#define SET_DERIVED_PARAMS(...) virtual DerivedParamVec getDerivedParams() const{ return __VA_ARGS__; }
#define SET_VARS(...) virtual StringPairVec getVars() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// NewModels::ValueBase
//----------------------------------------------------------------------------
namespace NewModels
{
//! Wrapper to ensure at compile time that correct number of values are
//! used when specifying the values of a model's parameters and initial state.
template<size_t NumValues>
class ValueBase
{
private:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::array<double, NumValues> ValueArray;

public:
    // **NOTE** other less terrifying forms of constructor won't complain at compile time about
    // number of parameters e.g. std::array<double, 4> can be initialized with <= 4 elements
    template<typename... T>
    ValueBase(T&&... vals) : m_Values(ValueArray{{std::forward<const double>(vals)...}})
    {
        static_assert(sizeof...(vals) == NumValues, "Wrong number of values");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets values as a vector of doubles
    std::vector<double> getValues() const
    {
        return std::vector<double>(m_Values.cbegin(), m_Values.cend());
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    double operator[](size_t pos) const
    {
        return m_Values[pos];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    ValueArray m_Values;
};

//----------------------------------------------------------------------------
// NewModels::ValueBase<0>
//----------------------------------------------------------------------------
//! Template specialisation of ValueBase to avoid compiler warnings
//! in the case when a model requires no parameters or state variables
template<>
class ValueBase<0>
{
public:
    // **NOTE** other less terrifying forms of constructor won't complain at compile time about
    // number of parameters e.g. std::array<double, 4> can be initialized with <= 4 elements
    template<typename... T>
    ValueBase(T&&... vals)
    {
        static_assert(sizeof...(vals) == 0, "Wrong number of values");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets values as a vector of doubles
    std::vector<double> getValues() const
    {
        return {};
    }
};

//----------------------------------------------------------------------------
// NewModels::Base
//----------------------------------------------------------------------------
//! Base class for all models
class Base
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::function<double(const std::vector<double> &, double)> DerivedParamFunc;
    typedef std::vector<std::string> StringVec;
    typedef std::vector<std::pair<std::string, std::string>> StringPairVec;
    typedef std::vector<std::pair<std::string, std::pair<std::string, double>>> NameTypeValVec;
    typedef std::vector<std::pair<std::string, DerivedParamFunc>> DerivedParamVec;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets names of of (independent) model parameters
    virtual StringVec getParamNames() const{ return {}; }

    //! Gets names of derived model parameters and the function objects to call to
    //! Calculate their value from a vector of model parameter values
    virtual DerivedParamVec getDerivedParams() const{ return {}; }

    //! Gets names and types (as strings) of model variables
    virtual StringPairVec getVars() const{ return {}; }
};

//----------------------------------------------------------------------------
// NewModels::LegacyWrapper
//----------------------------------------------------------------------------
//! Wrapper around old-style models stored in global arrays and referenced by index
template<typename ModelBase, typename LegacyModelType, const std::vector<LegacyModelType> &ModelArray>
class LegacyWrapper : public ModelBase
{
private:
    typedef typename ModelBase::DerivedParamFunc DerivedParamFunc;
    typedef typename ModelBase::StringVec StringVec;
    typedef typename ModelBase::StringPairVec StringPairVec;
    typedef typename ModelBase::DerivedParamVec DerivedParamVec;

public:
    LegacyWrapper(unsigned int legacyTypeIndex) : m_LegacyTypeIndex(legacyTypeIndex)
    {
    }

    //----------------------------------------------------------------------------
    // ModelBase virtuals
    //----------------------------------------------------------------------------
    //! Gets names of of (independent) model parameters
    virtual StringVec getParamNames() const
    {
        const auto &nm = ModelArray[m_LegacyTypeIndex];
        return nm.pNames;
    }

    //! Gets names of derived model parameters and the function objects to call to
    //! Calculate their value from a vector of model parameter values
    virtual DerivedParamVec getDerivedParams() const
    {
        const auto &m = ModelArray[m_LegacyTypeIndex];

        // Reserve vector to hold derived parameters
        DerivedParamVec derivedParams;
        derivedParams.reserve(m.dpNames.size());

        // Loop through derived parameters
        for(size_t p = 0; p < m.dpNames.size(); p++)
        {
            // Add pair consisting of parameter name and lambda function which calls
            // through to the DPS object associated with the legacy model
            derivedParams.push_back(std::make_pair(
              m.dpNames[p],
              [this, p](const std::vector<double> &pars, double dt)
              {
                  return ModelArray[m_LegacyTypeIndex].dps->calculateDerivedParameter(p, pars, dt);
              }
            ));
        }

        return derivedParams;
    }

    //! Gets names and types (as strings) of model variables
    virtual StringPairVec getVars() const
    {
        const auto &nm = ModelArray[m_LegacyTypeIndex];
        return zipStringVectors(nm.varNames, nm.varTypes);
    }

protected:
    //----------------------------------------------------------------------------
    // Static methods
    //----------------------------------------------------------------------------
    static StringPairVec zipStringVectors(const StringVec &a, const StringVec &b)
    {
        assert(a.size() == b.size());

        // Reserve vector to hold initial values
        StringPairVec zip;
        zip.reserve(a.size());

        // Build vector from legacy neuron model
        for(size_t v = 0; v < a.size(); v++)
        {
            zip.push_back(std::make_pair(a[v], b[v]));
        }

        return zip;
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    //! Index into the array of legacy models
    const unsigned int m_LegacyTypeIndex;
};
} // NewModels