#pragma once

// Standard C++ includes
#include <vector>

// Standard C includes
#include <cmath>

// GeNN includes
#include "snippet.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_ROW_BUILD_CODE(CODE) virtual std::string getRowBuildCode() const{ return CODE; }

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::Base
//----------------------------------------------------------------------------
//! Base class for all sparse connectivity initialisation snippets
namespace InitSparseConnectivitySnippet
{
class Base : public Snippet::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string getRowBuildCode() const{ return ""; }
};

//----------------------------------------------------------------------------
// Typedefines
//----------------------------------------------------------------------------
typedef Snippet::Init<Base> Init;

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::Uninitialised
//----------------------------------------------------------------------------
//! Used to mark connectivity as uninitialised - no initialisation code will be run
class Uninitialised : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::Uninitialised, 0);
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::OneToOne
//----------------------------------------------------------------------------
//! Initialises variable to a constant value
class OneToOne : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::OneToOne, 0);

    SET_ROW_BUILD_CODE(
        "$(addSynapse, $(i));\n"
        "$(stop);\n");
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::FixedProbability
//----------------------------------------------------------------------------
//! Initialises variable by sampling from the uniform distribution
class FixedProbability : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::FixedProbability, 1);

    SET_ROW_BUILD_CODE(
        "const scalar u = $(gennrand_uniform);\n"
        "$(j) += (1 + (int)(log(u) * $(probLogRecip)));\n"
        "if($(isPostNeuronValid($(j)) {\n"
        "   $(addSynapse, $(j));\n"
        "}\n"
        "else {\n"
        "   $(stop);\n"
        "}\n");

    SET_PARAM_NAMES({"prob"});
    SET_DERIVED_PARAMS({{"probLogRecip", [](const std::vector<double> &pars, double){ return 1.0 / log(1.0 - pars[0]); }}});
};

}   // namespace InitVarSnippet