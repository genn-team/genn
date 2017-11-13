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
class Constant : public Base
{
public:
    DECLARE_SNIPPET(InitVarSnippet::Constant, 1);

    SET_CODE("$(set_value, $(value));");

    SET_PARAM_NAMES({"value"});
};

//----------------------------------------------------------------------------
// InitVarSnippet::Uniform
//----------------------------------------------------------------------------
class Uniform : public Base
{
public:
    DECLARE_SNIPPET(InitVarSnippet::Uniform, 2);

    SET_CODE(
        "const scalar scale = $(max) - $(min);\n"
        "const scalar value = $(min) + ($(gennrand_uniform) * scale);\n"
        "$(set_value, value);");

    SET_PARAM_NAMES({"min", "max"});
};

//----------------------------------------------------------------------------
// InitVarSnippet::Normal
//----------------------------------------------------------------------------
class Normal : public Base
{
public:
    DECLARE_SNIPPET(InitVarSnippet::Normal, 2);

    SET_CODE(
        "const scalar value = $(mean) + ($(gennrand_normal) * $(sd));\n"
        "$(set_value, value);");

    SET_PARAM_NAMES({"mean", "sd"});
};
}   // namespace InitVarSnippet