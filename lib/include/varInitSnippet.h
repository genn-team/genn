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
// VarInitSnippet::Constant
//----------------------------------------------------------------------------
class Constant : public Base
{
public:
    DECLARE_SNIPPET(VarInitSnippet::Constant, 1);

    SET_CODE("$(set_value, $(value))");

    SET_PARAM_NAMES({"value"});
};
}   // namespace VarInitSnippet