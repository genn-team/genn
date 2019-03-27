#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// GeNN includes
#include "snippet.h"
#include "initVarSnippet.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_MODEL(TYPE, NUM_PARAMS, NUM_VARS)                       \
    DECLARE_SNIPPET(TYPE, NUM_PARAMS);                                  \
    typedef Models::VarInitContainerBase<NUM_VARS> VarValues;        \
    typedef Models::VarInitContainerBase<0> PreVarValues;            \
    typedef Models::VarInitContainerBase<0> PostVarValues

#define IMPLEMENT_MODEL(TYPE) IMPLEMENT_SNIPPET(TYPE)

#define SET_VARS(...) virtual StringPairVec getVars() const override{ return __VA_ARGS__; }
#define SET_EXTRA_GLOBAL_PARAMS(...) virtual StringPairVec getExtraGlobalParams() const override{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// Models::VarInit
//----------------------------------------------------------------------------
namespace Models
{
//! Class used to bind together everything required to initialise a variable:
//! 1. A pointer to a variable initialisation snippet
//! 2. The parameters required to control the variable initialisation snippet
class VarInit : public Snippet::Init<InitVarSnippet::Base>
{
public:
    VarInit(const InitVarSnippet::Base *snippet, const std::vector<double> &params)
        : Snippet::Init<InitVarSnippet::Base>(snippet, params)
    {
    }

    VarInit(double constant)
        : Snippet::Init<InitVarSnippet::Base>(InitVarSnippet::Constant::getInstance(), {constant})
    {
    }
};

//----------------------------------------------------------------------------
// Models::VarInitContainerBase
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
    typedef std::vector<VarInit> InitialiserArray;

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
    const std::vector<VarInit> &getInitialisers() const
    {
        return m_Initialisers;
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
// Models::VarInitContainerBase<0>
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

    VarInitContainerBase(const Snippet::ValueBase<0> &)
    {
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
// Models::Base
//----------------------------------------------------------------------------
//! Base class for all models - in addition to the parameters snippets have, models can have state variables
class Base : public Snippet::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Gets names and types (as strings) of model variables
    virtual StringPairVec getVars() const{ return {}; }

    //! Gets names and types (as strings) of additional
    //! per-population parameters for the weight update model.
    virtual StringPairVec getExtraGlobalParams() const{ return {}; }

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Find the index of a named variable
    size_t getVarIndex(const std::string &varName) const
    {
        return getStringPairVecIndex(varName, getVars());
    }

    //! Find the index of a named extra global parameter
    size_t getExtraGlobalParamIndex(const std::string &paramName) const
    {
        return getStringPairVecIndex(paramName, getExtraGlobalParams());
    }
};
} // Models
