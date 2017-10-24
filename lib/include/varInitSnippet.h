#pragma once

// GeNN includes
#include "snippet.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_CODE(CODE) virtual std::string getCode() const{ return CODE; }

//----------------------------------------------------------------------------
// VarInitSnippet::Base
//----------------------------------------------------------------------------
//! Base class for all value initialisation snippets
namespace VarInitSnippet
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
// VarInitSnippet::Uninitialised
//----------------------------------------------------------------------------
class Uninitialised : public Base
{
public:
    DECLARE_SNIPPET(VarInitSnippet::Uninitialised, 0);
};

//----------------------------------------------------------------------------
// VarInitSnippet::Constant
//----------------------------------------------------------------------------
class Constant : public Base
{
public:
    DECLARE_SNIPPET(VarInitSnippet::Constant, 1);

    SET_CODE("$(set_value, $(value));");

    SET_PARAM_NAMES({"value"});
};

//----------------------------------------------------------------------------
// VarInitSnippet::Uniform
//----------------------------------------------------------------------------
class Uniform : public Base
{
public:
    DECLARE_SNIPPET(VarInitSnippet::Uniform, 2);

    SET_CODE(
        "const scalar scale = $(max) - $(min);\n"
        "const scalar value = $(min) + ($(gennrand_uniform) * scale);\n"
        "$(set_value, value);");

    SET_PARAM_NAMES({"min", "max"});
};
}   // namespace VarInitSnippet