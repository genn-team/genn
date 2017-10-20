#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// Standard C includes
#include <cassert>

// GeNN includes
#include "snippet.h"
#include "varInitSnippet.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_MODEL(TYPE, NUM_PARAMS, NUM_VARS)                       \
    DECLARE_SNIPPET(TYPE, NUM_PARAMS)                                   \
    typedef Snippet::ValueBase<NUM_VARS> VarValues;                     \
    typedef NewModels::VarInitContainerBase<NUM_VARS> VarInitialisers;  \

#define IMPLEMENT_MODEL(TYPE) IMPLEMENT_SNIPPET(TYPE)

#define SET_VARS(...) virtual StringPairVec getVars() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// NewModels::VarInit
//----------------------------------------------------------------------------
namespace NewModels
{
class VarInit
{
public:
    template<typename Snippet>
    static VarInit init(const typename Snippet::ParamValues &params)
    {
        return VarInit(Snippet::getInstance(), params.getValues());
    }


    VarInit(const VarInitSnippet::Base *snippet, const std::vector<double> &params)
        : m_Snippet(snippet), m_Params(params)
    {
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const VarInitSnippet::Base *m_Snippet;
    std::vector<double> m_Params;
};

//----------------------------------------------------------------------------
// NewModels::VarInitContainerBase
//----------------------------------------------------------------------------
//! Wrapper to ensure at compile time that correct number of value initialisers
//! are used when specifying the values of a model's initial state.
template<size_t NumVars>
class VarInitContainerBase
{
private:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::array<VarInit, NumVars> InitialiserArray;

public:
    // **NOTE** other less terrifying forms of constructor won't complain at compile time about
    // number of parameters e.g. std::array<VarInit, 4> can be initialized with <= 4 elements
    template<typename... T>
    VarInitContainerBase(T&&... initialisers) : m_Initialisers(InitialiserArray{{std::forward<const VarInit>(initialisers)...}})
    {
        static_assert(sizeof...(initialisers) == NumVars, "Wrong number of initialisers");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets initialisers as a vector of Values
    std::vector<VarInit> getInitialisers() const
    {
        return std::vector<VarInit>(m_Initialisers.cbegin(), m_Initialisers.cend());
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    const VarInit &operator[](size_t pos) const
    {
        return m_Initialisers[pos];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    InitialiserArray m_Initialisers;
};

//----------------------------------------------------------------------------
// NewModels::VarInitBase<0>
//----------------------------------------------------------------------------
//! Template specialisation of ValueInitBase to avoid compiler warnings
//! in the case when a model requires no variable initialisers
template<>
class VarInitContainerBase<0>
{
public:
    // **NOTE** other less terrifying forms of constructor won't complain at compile time about
    // number of parameters e.g. std::array<VarInit, 4> can be initialized with <= 4 elements
    template<typename... T>
    VarInitContainerBase(T&&... initialisers)
    {
        static_assert(sizeof...(initialisers) == 0, "Wrong number of initialisers");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets initialisers as a vector of Values
    std::vector<VarInit> getInitialisers() const
    {
        return {};
    }
};

//----------------------------------------------------------------------------
// NewModels::Base
//----------------------------------------------------------------------------
//! Base class for all models
class Base : public Snippet::Base
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::vector<std::pair<std::string, std::string>> StringPairVec;
    typedef std::vector<std::pair<std::string, std::pair<std::string, double>>> NameTypeValVec;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
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
            zip.push_back(std::pair<std::string, std::string>(a[v], b[v]));
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