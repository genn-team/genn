#pragma once

// Standard C++ includes
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

// Standard C includes
#include <cassert>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_SNIPPET(TYPE, NUM_PARAMS)               \
private:                                                \
    GENN_EXPORT static TYPE *s_Instance;                \
public:                                                 \
    static const TYPE *getInstance()                    \
    {                                                   \
        if(s_Instance == NULL)                          \
        {                                               \
            s_Instance = new TYPE;                      \
        }                                               \
        return s_Instance;                              \
    }                                                   \
    typedef Snippet::ValueBase<NUM_PARAMS> ParamValues  \


#define IMPLEMENT_SNIPPET(TYPE) TYPE *TYPE::s_Instance = NULL

#define SET_PARAM_NAMES(...) virtual StringVec getParamNames() const override{ return __VA_ARGS__; }
#define SET_DERIVED_PARAMS(...) virtual DerivedParamVec getDerivedParams() const override{ return __VA_ARGS__; }
#define SET_EXTRA_GLOBAL_PARAMS(...) virtual EGPVec getExtraGlobalParams() const override{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// Snippet::InitialiserContainerBase
//----------------------------------------------------------------------------
//! Wrapper to ensure at compile time that correct 
//! number of values are used when initialising models
namespace Snippet
{
template<typename V, size_t NumVars>
class InitialiserContainerBase
{
public:
    // **NOTE** other less terrifying forms of constructor won't complain at compile time about
    // number of parameters e.g. std::array<V, 4> can be initialized with <= 4 elements
    template<typename... T>
    InitialiserContainerBase(T&&... vals) : m_Values(std::vector<V>{{std::forward<const V>(vals)...}})
    {
        static_assert(sizeof...(vals) == NumVars, "Wrong number of values");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets values as a vector
    const std::vector<V> &getInitialisers() const
    {
        return m_Values;
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    const V &operator[](size_t pos) const
    {
        return m_Values[pos];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::vector<V> m_Values;
};

//----------------------------------------------------------------------------
// Snippet::InitialiserContainerBase<0>
//----------------------------------------------------------------------------
//! Template specialisation of InitialiserContainerBase to avoid compiler warnings
//! in the case when a model requires no parameters or state variables
template<typename V>
class InitialiserContainerBase<V, 0>
{
public:
    // **NOTE** other less terrifying forms of constructor won't complain at compile time about
    // number of parameters e.g. std::array<double, 4> can be initialized with <= 4 elements
    template<typename... T>
    InitialiserContainerBase(T&&... vals)
    {
        static_assert(sizeof...(vals) == 0, "Wrong number of values");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets values as a vector of doubles
    std::vector<V> getInitialisers() const
    {
        return {};
    }
};

//----------------------------------------------------------------------------
// Snippet::ValueBase
//----------------------------------------------------------------------------
template<size_t NumVars>
using ValueBase = InitialiserContainerBase<double, NumVars>;

//----------------------------------------------------------------------------
// Snippet::Base
//----------------------------------------------------------------------------
//! Base class for all code snippets
class GENN_EXPORT Base
{
public:
    virtual ~Base()
    {
    }

    //----------------------------------------------------------------------------
    // Structs
    //----------------------------------------------------------------------------
    //! An extra global parameter has a name and a type
    struct EGP
    {
        bool operator == (const EGP &other) const
        {
            return ((name == other.name) && (type == other.type));
        }

        std::string name;
        std::string type;
    };

    //! Additional input variables, row state variables and other things have a name, a type and an initial value
    struct ParamVal
    {
        ParamVal(const std::string &n, const std::string &t, const std::string &v) : name(n), type(t), value(v)
        {}
        ParamVal(const std::string &n, const std::string &t, double v) : ParamVal(n, t, Utils::writePreciseString(v))
        {}
        ParamVal() : ParamVal("", "", "0.0")
        {}

        bool operator == (const ParamVal &other) const
        {
            return ((name == other.name) && (type == other.type) && (value == other.value));
        }

        std::string name;
        std::string type;
        std::string value;
    };

    //! A derived parameter has a name and a function for obtaining its value
    struct DerivedParam
    {
        bool operator == (const DerivedParam &other) const
        {
            return (name == other.name);
        }

        std::string name;
        std::function<double(const std::vector<double> &, double)> func;
    };


    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::vector<std::string> StringVec;
    typedef std::vector<EGP> EGPVec;
    typedef std::vector<ParamVal> ParamValVec;
    typedef std::vector<DerivedParam> DerivedParamVec;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets names of of (independent) model parameters
    virtual StringVec getParamNames() const{ return {}; }

    //! Gets names of derived model parameters and the function objects to call to
    //! Calculate their value from a vector of model parameter values
    virtual DerivedParamVec getDerivedParams() const{ return {}; }

    //! Gets names and types (as strings) of additional
    //! per-population parameters for the snippet
    virtual EGPVec getExtraGlobalParams() const { return {}; }
    
    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Find the index of a named extra global parameter
    size_t getExtraGlobalParamIndex(const std::string &paramName) const
    {
        return getNamedVecIndex(paramName, getExtraGlobalParams());
    }

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    bool canBeMerged(const Base *other) const
    {
        // Return true if parameters names and derived parameter names match
        return ((getParamNames() == other->getParamNames()) 
                && (getDerivedParams() == other->getDerivedParams())
                && (getExtraGlobalParams() == other->getExtraGlobalParams()));
    }

    //------------------------------------------------------------------------
    // Protected static helpers
    //------------------------------------------------------------------------
    template<typename T>
    static size_t getNamedVecIndex(const std::string &name, const std::vector<T> &vec)
    {
        auto iter = std::find_if(vec.begin(), vec.end(),
            [name](const T &v){ return (v.name == name); });

        if(iter == vec.end()) {
            throw std::runtime_error("Cannot find variable '" + name + "'");
        }

        // Return 'distance' between first entry in vector and iterator i.e. index
        return distance(vec.begin(), iter);
    }
};

//----------------------------------------------------------------------------
// Snippet::Init
//----------------------------------------------------------------------------
//! Class used to bind together everything required to utilize a snippet
//! 1. A pointer to a variable initialisation snippet
//! 2. The parameters required to control the variable initialisation snippet
template<typename SnippetBase>
class Init
{
public:
    Init(const SnippetBase *snippet, const std::vector<double> &params)
        : m_Snippet(snippet), m_Params(params)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const SnippetBase *getSnippet() const{ return m_Snippet; }
    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<double> &getDerivedParams() const{ return m_DerivedParams; }

    void initDerivedParams(double dt)
    {
        auto derivedParams = m_Snippet->getDerivedParams();

        // Reserve vector to hold derived parameters
        m_DerivedParams.reserve(derivedParams.size());

        // Loop through derived parameters
        for(const auto &d : derivedParams) {
            m_DerivedParams.push_back(d.func(m_Params, dt));
        }
    }

    bool canBeMerged(const Init<SnippetBase> &other) const
    {
        return getSnippet()->canBeMerged(other.getSnippet());
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SnippetBase *m_Snippet;
    std::vector<double> m_Params;
    std::vector<double> m_DerivedParams;
};
}   // namespace Snippet
