#pragma once

// GeNN includes
#include "snippet.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_CODE(CODE) virtual std::string getCode() const{ return CODE; }

//----------------------------------------------------------------------------
// InitVarSnippet::Base
//----------------------------------------------------------------------------
//! Base class for all value initialisation snippets
namespace InitVarSnippet
{
class Base : public Snippet::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string getCode() const{ return ""; }
};

//----------------------------------------------------------------------------
// InitVarSnippet::Uninitialised
//----------------------------------------------------------------------------
class Uninitialised : public Base
{
public:
    DECLARE_SNIPPET(InitVarSnippet::Uninitialised, 0);
};

//----------------------------------------------------------------------------
// InitVarSnippet::Constant
//----------------------------------------------------------------------------
//! Initialises variable to a constant value
/*! This snippet takes 1 parameter:
 *
    - \c value - The value to intialise the variable to

    \note This snippet type is seldom used directly - NewModels::VarInit
    has an implicit constructor that, internally, creates one of these snippets*/
class Constant : public Base
{
public:
    DECLARE_SNIPPET(InitVarSnippet::Constant, 1);

    SET_CODE("$(value) = $(constant);");

    SET_PARAM_NAMES({"constant"});
};

//----------------------------------------------------------------------------
// InitVarSnippet::Uniform
//----------------------------------------------------------------------------
//! Initialises variable by sampling from the uniform distribution
/*! This snippet takes 2 parameters:
 *
    - \c min - The minimum value
    - \c max - The maximum value */
class Uniform : public Base
{
public:
    DECLARE_SNIPPET(InitVarSnippet::Uniform, 2);

    SET_CODE(
        "const scalar scale = $(max) - $(min);\n"
        "$(value) = $(min) + ($(gennrand_uniform) * scale);");

    SET_PARAM_NAMES({"min", "max"});
};

//----------------------------------------------------------------------------
// InitVarSnippet::Normal
//----------------------------------------------------------------------------
//! Initialises variable by sampling from the normal distribution
/*! This snippet takes 2 parameters:
 *
    - \c mean - The mean
    - \c sd - The standard distribution*/
class Normal : public Base
{
public:
    DECLARE_SNIPPET(InitVarSnippet::Normal, 2);

    SET_CODE("$(value) = $(mean) + ($(gennrand_normal) * $(sd));");

    SET_PARAM_NAMES({"mean", "sd"});
};

//----------------------------------------------------------------------------
// InitVarSnippet::Exponential
//----------------------------------------------------------------------------
//! Initialises variable by sampling from the exponential distribution
/*! This snippet takes 1 parameter:
 *
    - \c lambda - mean event rate (events per unit time/distance)*/
class Exponential : public Base
{
public:
    DECLARE_SNIPPET(InitVarSnippet::Exponential, 1);

    SET_CODE("$(value) = $(lambda) * $(gennrand_exponential);");

    SET_PARAM_NAMES({"lambda"});
};
}   // namespace InitVarSnippet