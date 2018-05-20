#pragma once

// Standard C++ includes
#include <functional>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// GeNN includes
#include "binomial.h"
#include "snippet.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_ROW_BUILD_CODE(CODE) virtual std::string getRowBuildCode() const{ return CODE; }
#define SET_CALC_MAX_ROW_LENGTH_FUNC(FUNC) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const{ return FUNC; }
#define SET_CALC_MAX_COL_LENGTH_FUNC(FUNC) virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const{ return FUNC; }

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
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::function<unsigned int(unsigned int, unsigned int, const std::vector<double> &)> CalcMaxLengthFunc;
    
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string getRowBuildCode() const{ return ""; }
    virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const{ return CalcMaxLengthFunc(); }
    virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const{ return CalcMaxLengthFunc(); }
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
        "$(endRow);\n");
    
    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &)
        {
            assert(numPre == numPost);
            return 1;
        });
    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &)
        {
            assert(numPre == numPost);
            return 1;
        });
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
        "$(prevJ) += (1 + (int)(log(u) * $(probLogRecip)));\n"
        "if($(isPostNeuronValid, $(prevJ))) {\n"
        "   $(addSynapse, $(prevJ));\n"
        "}\n"
        "else {\n"
        "   $(endRow);\n"
        "}\n");

    SET_PARAM_NAMES({"prob"});
    SET_DERIVED_PARAMS({{"probLogRecip", [](const std::vector<double> &pars, double){ return 1.0 / log(1.0 - pars[0]); }}});
    
    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            return binomialInverseCDF(quantile, numPost, pars[0]);
        });
    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPos times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            return binomialInverseCDF(quantile, numPre, pars[0]);
        });
};

}   // namespace InitVarSnippet