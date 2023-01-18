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

// Forward declarations
namespace GeNN
{
namespace Type
{
class NumericBase;
}
}

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
// GeNN::Snippet::Base
//----------------------------------------------------------------------------
//! Base class for all code snippets
namespace GeNN::Snippet
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
    //! An extra global parameter has a name and a type
    struct EGP
    {
        EGP(const std::string &n, const Type::NumericBase *t);
        EGP(const std::string &n, const std::string &t);
        
        bool operator == (const EGP &other) const;

        const std::string name;
        const Type::NumericBase *type;
    };

    //! Additional input variables, row state variables and other things have a name, a type and an initial value
    struct ParamVal
    {
        ParamVal(const std::string &n, const std::string &t, const std::string &v) : name(n), type(t), value(v)
        {}
        ParamVal(const std::string &n, const std::string &t, double v) : ParamVal(n, t, Utils::writePreciseString(v))
        {}

        bool operator == (const ParamVal &other) const
        {
            return ((name == other.name) && (type == other.type) && (value == other.value));
        }

        const std::string name;
        const std::string type;
        const std::string value;
    };

    //! A derived parameter has a name and a function for obtaining its value
    struct DerivedParam
    {
        bool operator == (const DerivedParam &other) const
        {
            return (name == other.name);
        }

        const std::string name;
        const std::function<double(const std::unordered_map<std::string, double>&, double)> func;
    };


    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::vector<std::string> StringVec;
    typedef std::vector<EGP> EGPVec;
    typedef std::vector<ParamVal> ParamValVec;
    typedef std::vector<DerivedParam> DerivedParamVec;
    typedef std::function<unsigned int(unsigned int, unsigned int, const std::unordered_map<std::string, double> &)> CalcMaxLengthFunc;
    typedef std::function<std::vector<unsigned int>(const std::unordered_map<std::string, double> &)> CalcKernelSizeFunc;

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
    void updateHash(boost::uuids::detail::sha1 &hash) const;

    //! Validate names of parameters etc
    void validate(const std::unordered_map<std::string, double> &paramValues, const std::string &description) const;

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
    Init(const SnippetBase *snippet, const std::unordered_map<std::string, double> &params)
        : m_Snippet(snippet), m_Params(params)
    {
        // Validate names
        getSnippet()->validate(params);
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const SnippetBase *getSnippet() const{ return m_Snippet; }
    const std::unordered_map<std::string, double> &getParams() const{ return m_Params; }
    const std::unordered_map<std::string, double> &getDerivedParams() const{ return m_DerivedParams; }

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
    std::unordered_map<std::string, double> m_Params;
    std::unordered_map<std::string, double> m_DerivedParams;
};

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
GENN_EXPORT void updateHash(const Base::EGP &e, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::ParamVal &p, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::DerivedParam &d, boost::uuids::detail::sha1 &hash);
}   // namespace GeNN::Snippet
