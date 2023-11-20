#pragma once

// Standard C++ includes
#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Standard C includes
#include <cassert>

// GeNN includes
#include "gennExport.h"
#include "type.h"

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

#define SET_PARAMS(...) virtual ParamVec getParams() const override{ return __VA_ARGS__; }
#define SET_DERIVED_PARAMS(...) virtual DerivedParamVec getDerivedParams() const override{ return __VA_ARGS__; }
#define SET_EXTRA_GLOBAL_PARAMS(...) virtual EGPVec getExtraGlobalParams() const override{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// GeNN::Snippet::Base
//----------------------------------------------------------------------------
//! Base class for all code snippets
namespace GeNN
{
using ParamValues = std::unordered_map<std::string, Type::NumericValue>;

namespace Snippet
{
class GENN_EXPORT Base
{
public:
    virtual ~Base()
    {
    }

    //----------------------------------------------------------------------------
    // Structs
    //----------------------------------------------------------------------------
    //! A parameter has a name and a type
    struct GENN_EXPORT Param
    {
        Param(const std::string &n, const Type::ResolvedType &t) : name(n), type(t)
        {}
        Param(const std::string &n, const std::string &t) : name(n), type(t)
        {}
        Param(const char *n) : Param(n, "scalar")
        {}
        
        bool operator == (const Param &other) const
        {
            return (std::tie(name, type) == std::tie(other.name, other.type));
        }

        std::string name;
        Type::UnresolvedType type;
    };

    //! An extra global parameter has a name and a type
    struct GENN_EXPORT EGP
    {
        EGP(const std::string &n, const Type::ResolvedType &t) : name(n), type(t)
        {}
        EGP(const std::string &n, const std::string &t);
        
        bool operator == (const EGP &other) const
        {
            return (std::tie(name, type) == std::tie(other.name, other.type));
        }

        std::string name;
        Type::UnresolvedType type;
    };

    //! Additional input variables, row state variables and other things have a name, a type and an initial value
    struct GENN_EXPORT ParamVal
    {
        ParamVal(const std::string &n, const Type::ResolvedType &t, Type::NumericValue v)
        :   name(n), type(t), value(v)
        {}

        ParamVal(const std::string &n, const std::string &t, Type::NumericValue v)
        :   name(n), type(t), value(v)
        {}

        bool operator == (const ParamVal &other) const
        {
            // **THINK** why isn't value included?
            return (std::tie(name, type) == std::tie(other.name, other.type));
        }

        std::string name;
        Type::UnresolvedType type;
        Type::NumericValue value;
    };

    //! A derived parameter has a name and a function for obtaining its value
    struct GENN_EXPORT DerivedParam
    {
        bool operator == (const DerivedParam &other) const
        {
            return (name == other.name);
        }

        std::string name;
        std::function<Type::NumericValue(const std::unordered_map<std::string, Type::NumericValue>&, double)> func;
    };

    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::vector<Param> ParamVec;
    typedef std::vector<EGP> EGPVec;
    typedef std::vector<ParamVal> ParamValVec;
    typedef std::vector<DerivedParam> DerivedParamVec;
    typedef std::function<unsigned int(unsigned int, unsigned int, const std::unordered_map<std::string, Type::NumericValue> &)> CalcMaxLengthFunc;
    typedef std::function<std::vector<unsigned int>(const std::unordered_map<std::string, Type::NumericValue> &)> CalcKernelSizeFunc;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets names and types of (independent) model parameters
    virtual ParamVec getParams() const{ return {}; }

    //! Gets names of derived model parameters and the function objects to call to
    //! Calculate their value from a vector of model parameter values
    virtual DerivedParamVec getDerivedParams() const{ return {}; }

    //! Gets names and types (as strings) of additional
    //! per-population parameters for the snippet
    virtual EGPVec getExtraGlobalParams() const { return {}; }
    
    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Find the named parameter
    std::optional<Param> getParam(const std::string &paramName) const
    {
        return getNamed(paramName, getParams());
    }
    //! Find the named extra global parameter
    std::optional<EGP> getExtraGlobalParam(const std::string &paramName) const
    {
        return getNamed(paramName, getExtraGlobalParams());
    }

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void updateHash(boost::uuids::detail::sha1 &hash) const;

    //! Validate names of parameters etc
    void validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues, const std::string &description) const;

    //------------------------------------------------------------------------
    // Protected static helpers
    //------------------------------------------------------------------------
    template<typename T>
    static std::optional<T> getNamed(const std::string &name, const std::vector<T> &vec)
    {
        auto iter = std::find_if(vec.begin(), vec.end(),
            [name](const T &v){ return (v.name == name); });
        if(iter == vec.cend()) {
            return std::nullopt;
        }
        else {
            return *iter;
        }
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
    Init(const SnippetBase *snippet, const std::unordered_map<std::string, Type::NumericValue> &params)
        : m_Snippet(snippet), m_Params(params)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const SnippetBase *getSnippet() const{ return m_Snippet; }
    const std::unordered_map<std::string, Type::NumericValue> &getParams() const{ return m_Params; }
    const std::unordered_map<std::string, Type::NumericValue> &getDerivedParams() const{ return m_DerivedParams; }

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return getSnippet()->getHashDigest();
    }

    void finalise(double dt)
    {
        auto derivedParams = m_Snippet->getDerivedParams();

        // Loop through derived parameters
        for(const auto &d : derivedParams) {
            m_DerivedParams.emplace(d.name, d.func(m_Params, dt));
        }
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SnippetBase *m_Snippet;
    std::unordered_map<std::string, Type::NumericValue> m_Params;
    std::unordered_map<std::string, Type::NumericValue> m_DerivedParams;
};

//----------------------------------------------------------------------------
// GeNN::Snippet::DynamicParameterContainer
//----------------------------------------------------------------------------
class DynamicParameterContainer
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void set(const std::string &name, bool value);
    bool get(const std::string &name) const;

    void updateHash(boost::uuids::detail::sha1 &hash) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const Base *m_Snippet;
    std::unordered_set<std::string> m_Dynamic;
};

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
GENN_EXPORT void updateHash(const Base::Param &p, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::EGP &e, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::ParamVal &p, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::DerivedParam &d, boost::uuids::detail::sha1 &hash);
}   // namespace GeNN::Snippet
}   // namespace GeNN