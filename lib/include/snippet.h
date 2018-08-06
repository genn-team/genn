#pragma once

// Standard C++ includes
#include <functional>
#include <string>
#include <vector>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_SNIPPET(TYPE, NUM_PARAMS)               \
private:                                                \
    static TYPE *s_Instance;                            \
public:                                                 \
    static const TYPE *getInstance()                    \
    {                                                   \
        if(s_Instance == NULL)                          \
        {                                               \
            s_Instance = new TYPE;                      \
        }                                               \
        return s_Instance;                              \
    }                                                   \
    typedef Snippet::ValueBase<NUM_PARAMS> ParamValues; \


#define IMPLEMENT_SNIPPET(TYPE) TYPE *TYPE::s_Instance = NULL

#define SET_PARAM_NAMES(...) virtual StringVec getParamNames() const{ return __VA_ARGS__; }
#define SET_DERIVED_PARAMS(...) virtual DerivedParamVec getDerivedParams() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// Snippet::ValueBase
//----------------------------------------------------------------------------
//! Wrapper to ensure at compile time that correct number of values are
//! used when specifying the values of a model's parameters and initial state.
namespace Snippet
{
template<size_t NumVars>
class ValueBase
{
public:
    // **NOTE** other less terrifying forms of constructor won't complain at compile time about
    // number of parameters e.g. std::array<double, 4> can be initialized with <= 4 elements
    template<typename... T>
    ValueBase(T&&... vals) : m_Values(std::vector<double>{{std::forward<const double>(vals)...}})
    {
        static_assert(sizeof...(vals) == NumVars, "Wrong number of values");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Gets values as a vector of doubles
    const std::vector<double> &getValues() const
    {
        return m_Values;
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
    std::vector<double> m_Values;
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
// Snippet::Base
//----------------------------------------------------------------------------
//! Base class for all code snippets
class Base
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::function<double(const std::vector<double> &, double)> DerivedParamFunc;
    typedef std::vector<std::string> StringVec;
    typedef std::vector<std::pair<std::string, DerivedParamFunc>> DerivedParamVec;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Virtual destructor
    virtual ~Base() {}

    //! Gets names of of (independent) model parameters
    virtual StringVec getParamNames() const{ return {}; }

    //! Gets names of derived model parameters and the function objects to call to
    //! Calculate their value from a vector of model parameter values
    virtual DerivedParamVec getDerivedParams() const{ return {}; }
};
}   // namespace Snippet
