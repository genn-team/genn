#pragma once

// Standard C++ includes
#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Standard C includes
#include <cassert>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_SNIPPET(TYPE)                           \
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
    }


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
// Snippet::ParamValues
//----------------------------------------------------------------------------
//! Container for parameter (and derived parameter) values
/*! We don't use std::unordered_map directly both to save typing and because it doesn't have a const [] operator */
class ParamValues
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::unordered_map<std::string, double> ParamMap;

    ParamValues()
    {
    }

    ParamValues(const ParamMap &values)
    :   m_Values(values)
    {}

    ParamValues(std::initializer_list<ParamMap::value_type> values)
    :   m_Values(values)
    {}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    double operator[](const std::string &name) const
    {
        return m_Values.at(name);
    }

    ParamMap &getValues() { return m_Values; }
    const ParamMap &getValues() const { return m_Values; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    ParamMap m_Values;
};

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
        std::function<double(const ParamValues &, double)> func;
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
    void updateHash(boost::uuids::detail::sha1 &hash) const
    {
        Utils::updateHash(getParamNames(), hash);
        Utils::updateHash(getDerivedParams(), hash);
        Utils::updateHash(getExtraGlobalParams(), hash);
    }

    //! Validate names of parameters etc
    void validate() const
    {
        Utils::validateParamNames(getParamNames());
        Utils::validateVecNames(getDerivedParams(), "Derived parameter");
        Utils::validateVecNames(getExtraGlobalParams(), "Derived parameter");
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
    Init(const SnippetBase *snippet, const ParamValues &params)
        : m_Snippet(snippet), m_Params(params)
    {
        // Validate names
        getSnippet()->validate();
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const SnippetBase *getSnippet() const{ return m_Snippet; }
    const ParamValues::ParamMap &getParams() const{ return m_Params.getValues(); }
    const ParamValues::ParamMap &getDerivedParams() const{ return m_DerivedParams; }

    void initDerivedParams(double dt)
    {
        auto derivedParams = m_Snippet->getDerivedParams();

        // Loop through derived parameters
        for(const auto &d : derivedParams) {
            m_DerivedParams.emplace(d.name, d.func(m_Params, dt));
        }
    }

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return getSnippet()->getHashDigest();
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SnippetBase *m_Snippet;
    ParamValues m_Params;
    ParamValues::ParamMap m_DerivedParams;
};

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
inline void updateHash(const Base::EGP &e, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(e.name, hash);
    Utils::updateHash(e.type, hash);
}

inline void updateHash(const Base::ParamVal &p, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(p.name, hash);
    Utils::updateHash(p.type, hash);
    Utils::updateHash(p.value, hash);
}

inline void updateHash(const Base::DerivedParam &d, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(d.name, hash);
}
}   // namespace Snippet
