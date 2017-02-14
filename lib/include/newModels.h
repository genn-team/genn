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
#define DECLARE_MODEL(TYPE, NUM_PARAMS, NUM_INIT_VALUES)       \
public:                                                        \
    static const TYPE *GetInstance()                           \
    {                                                          \
        if(s_Instance == NULL)                                 \
        {                                                      \
            s_Instance = new TYPE;                             \
        }                                                      \
        return s_Instance;                                     \
    }                                                          \
    typedef NewModels::ValueBase<NUM_PARAMS> ParamValues;      \
    typedef NewModels::ValueBase<NUM_INIT_VALUES> InitValues;  \
private:                                                       \
    static TYPE *s_Instance;

#define IMPLEMENT_MODEL(TYPE) TYPE *TYPE::s_Instance = NULL

#define SET_PARAM_NAMES(...) virtual std::vector<std::string> GetParamNames() const{ return __VA_ARGS__; }
#define SET_DERIVED_PARAMS(...) virtual std::vector<std::pair<std::string, DerivedParamFunc>> GetDerivedParams() const{ return __VA_ARGS__; }
#define SET_INIT_VALS(...) virtual std::vector<std::pair<std::string, std::string>> GetInitVals() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// NewModels::ValueBase
//----------------------------------------------------------------------------
namespace NewModels
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
    std::vector<double> GetValues() const
    {
        return {};
    }
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
    typedef std::function<double(const std::vector<double> &, double)> DerivedParamFunc;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::vector<std::string> GetParamNames() const{ return {}; }
    virtual std::vector<std::pair<std::string, DerivedParamFunc>> GetDerivedParams() const{ return {}; }
    virtual std::vector<std::pair<std::string, std::string>> GetInitVals() const{ return {}; }
};

//----------------------------------------------------------------------------
// NeuronModels::LegacyWrapper
//----------------------------------------------------------------------------
// Wrapper around neuron models stored in global
template<typename ModelBase, typename LegacyModelType, const std::vector<LegacyModelType> &ModelArray>
class LegacyWrapper : public ModelBase
{
private:
    typedef typename ModelBase::DerivedParamFunc DerivedParamFunc;

public:
    LegacyWrapper(unsigned int legacyTypeIndex) : m_LegacyTypeIndex(legacyTypeIndex)
    {
    }

    //----------------------------------------------------------------------------
    // ModelBase virtuals
    //----------------------------------------------------------------------------
    std::vector<std::string>  GetParamNames() const
    {
        const auto &nm = ModelArray[m_LegacyTypeIndex];
        return nm.pNames;
    }

    std::vector<std::pair<std::string, DerivedParamFunc>> GetDerivedParams() const
    {
        const auto &m = ModelArray[m_LegacyTypeIndex];

        // Reserve vector to hold derived parameters
        std::vector<std::pair<std::string, DerivedParamFunc>> derivedParams;
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
    std::vector<std::pair<std::string, std::string>> GetInitVals() const
    {
        const auto &nm = ModelArray[m_LegacyTypeIndex];
        return ZipStringVectors(nm.varNames, nm.varTypes);
    }

protected:
    //----------------------------------------------------------------------------
    // Static methods
    //----------------------------------------------------------------------------
    static std::vector<std::pair<std::string, std::string>> ZipStringVectors(const std::vector<std::string> &a,
                                                                             const std::vector<std::string> &b)
    {
        assert(a.size() == b.size());

        // Reserve vector to hold initial values
        std::vector<std::pair<std::string, std::string>> zip;
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