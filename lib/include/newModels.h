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
    typedef NewModels::ValueBase<NUM_VARS> VarValues;


#define IMPLEMENT_MODEL(TYPE) TYPE *TYPE::s_Instance = NULL

#define SET_PARAM_NAMES(...) virtual StringVec getParamNames() const{ return __VA_ARGS__; }
#define SET_DERIVED_PARAMS(...) virtual DerivedParamVec getDerivedParams() const{ return __VA_ARGS__; }
#define SET_VARS(...) virtual StringPairVec getVars() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// NewModels::ValueBase
//----------------------------------------------------------------------------
namespace NewModels
{
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
    ValueBase(T&&... vals) : m_Values(ValueArray{std::forward<double>(vals)...})
    {
        static_assert(sizeof...(vals) == NumValues, "Wrong number of values");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::vector<double> getValues() const
    {
        return std::vector<double>(m_Values.cbegin(), m_Values.cend());
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
    std::vector<double> getValues() const
    {
        return {};
    }
};

//----------------------------------------------------------------------------
// NewModels::Base
//----------------------------------------------------------------------------
class Base
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::function<double(const std::vector<double> &, double)> DerivedParamFunc;
    typedef std::vector<std::string> StringVec;
    typedef std::vector<std::pair<std::string, std::string>> StringPairVec;
    typedef std::vector<std::pair<std::string, DerivedParamFunc>> DerivedParamVec;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual StringVec getParamNames() const{ return {}; }
    virtual DerivedParamVec getDerivedParams() const{ return {}; }
    virtual StringPairVec getVars() const{ return {}; }
};

//----------------------------------------------------------------------------
// NewModels::LegacyWrapper
//----------------------------------------------------------------------------
// Wrapper around old-style models stored in global arrays and referenced by index
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
    virtual StringVec getParamNames() const
    {
        const auto &nm = ModelArray[m_LegacyTypeIndex];
        return nm.pNames;
    }

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
            derivedParams.push_back(std::pair<std::string, DerivedParamFunc>(
              m.dpNames[p],
              [this, p](const std::vector<double> &pars, double dt)
              {
                  return ModelArray[m_LegacyTypeIndex].dps->calculateDerivedParameter(p, pars, dt);
              }
            ));
        }

        return derivedParams;
    }

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
            zip.push_back(std::pair<std::string, std::string>(a[v], b[v]));
        }

        return zip;
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const unsigned int m_LegacyTypeIndex;
};
} // NewModels