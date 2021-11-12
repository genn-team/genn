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
#define SET_DIAGONAL_BUILD_CODE(CODE) virtual std::string getDiagonalBuildCode() const override{ return CODE; }
#define SET_DIAGONAL_BUILD_STATE_VARS(...) virtual ParamValVec getDiagonalBuildStateVars() const override{ return __VA_ARGS__; }

#define SET_CALC_MAX_ROW_LENGTH_FUNC(FUNC) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return FUNC; }
#define SET_CALC_KERNEL_SIZE_FUNC(...) virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const override{ return __VA_ARGS__; }

#define SET_MAX_ROW_LENGTH(MAX_ROW_LENGTH) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return [](unsigned int, unsigned int, const std::vector<double> &){ return MAX_ROW_LENGTH; }; }

//----------------------------------------------------------------------------
// InitToeplitzConnectivitySnippet::Base
//----------------------------------------------------------------------------
//! Base class for all toeplitz connectivity initialisation snippets
namespace InitToeplitzConnectivitySnippet
{
class GENN_EXPORT Base : public Snippet::Base
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::function<unsigned int(unsigned int, unsigned int, const std::vector<double> &)> CalcMaxLengthFunc;
    typedef std::function<std::vector<unsigned int>(const std::vector<double> &)> CalcKernelSizeFunc;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string getDiagonalBuildCode() const{ return ""; }
    virtual ParamValVec getDiagonalBuildStateVars() const { return {}; }

    //! Get function to calculate the maximum row length of this connector based on the parameters and the size of the pre and postsynaptic population
    virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const{ return CalcMaxLengthFunc(); }

    //! Get function to calculate kernel size required for this conenctor based on its parameters
    virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const{ return CalcKernelSizeFunc(); }

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Update hash from snippet
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Validate names of parameters etc
    void validate() const;
};

//----------------------------------------------------------------------------
// Init
//----------------------------------------------------------------------------
class Init : public Snippet::Init<InitToeplitzConnectivitySnippet::Base>
{
public:
    Init(const Base *snippet, const std::vector<double> &params)
        : Snippet::Init<Base>(snippet, params)
    {
    }
};

//----------------------------------------------------------------------------
// InitToeplitzConnectivitySnippet::Uninitialised
//----------------------------------------------------------------------------
//! Used to mark connectivity as uninitialised - no initialisation code will be run
class Uninitialised : public Base
{
public:
    DECLARE_SNIPPET(InitToeplitzConnectivitySnippet::Uninitialised, 0);
};

//----------------------------------------------------------------------------
// InitToeplitzConnectivitySnippet::Conv2D
//----------------------------------------------------------------------------
//! Initialises convolutional connectivity
//! Row build state variables are used to convert presynaptic neuron index to rows, columns and channels and, 
//! from these, to calculate the range of postsynaptic rows, columns and channels connections will be made within.
/*! This sparse connectivity snippet does not support multiple threads per neuron */
class Conv2D : public Base
{
public:
    DECLARE_SNIPPET(Conv2D, 10);

    SET_PARAM_NAMES({"conv_kh", "conv_kw",
                     "conv_sh", "conv_sw",
                     "conv_ih", "conv_iw", "conv_ic",
                     "conv_oh", "conv_ow", "conv_oc"});
    SET_DERIVED_PARAMS({{"conv_bw", [](const std::vector<double> &pars, double){ return (((int)pars[5] + (int)pars[1] - 1) - (int)pars[8]) / 2; }},
                        {"conv_bh", [](const std::vector<double> &pars, double){ return (((int)pars[4] + (int)pars[0] - 1) - (int)pars[7]) / 2; }}});

    SET_DIAGONAL_BUILD_STATE_VARS({{"kernRow", "int", "($(id_diag) / (int)$(conv_oc)) / (int)$(conv_kw)"},
                                   {"kernCol", "int", "($(id_diag) / (int)$(conv_oc)) % (int)$(conv_kw)"},
                                   {"kernOutChan", "int", "$(id_diag) % (int)$(conv_oc)"},
                                   {"flipKernRow", "int", "(int)$(conv_kh) - $(kernRow) - 1"},
                                   {"flipKernCol", "int", "(int)$(conv_kw) - $(kernCol) - 1"},
                                   {"kernelInd", "int", "($(flipKernRow) * (int)$(conv_kw) * (int)$(conv_ic) * (int)$(conv_oc)) + ($(flipKernCol) * (int)$(conv_ic) * (int)$(conv_oc)) + $(kernOutChan);"}});

    SET_DIAGONAL_BUILD_CODE(
        "const int preRow = ($(id_pre) / (int)$(conv_ic)) / (int)$(conv_iw)\n"
        "const int preCol = ($(id_pre) / (int)$(conv_ic)) % (int)$(conv_iw)\n"
        "const int preChan = ($(id_pre) % (int)$(conv_ic);\n"
        "// If we haven't gone off edge of output\n"
        "const int postRow = $(preRow) + $(kernRow) - (int)$(conv_bh);\n"
        "const int postCol = $(preCol) + $(kernCol) - (int)$(conv_bw);\n"
        "if(postRow >= 0 && postCol >= 0 && postRow < (int)$(conv_oh) && postCol < (int)$(conv_ow)) {\n"
        "    // Calculate postsynaptic index\n"
        "    const int postInd = ((postRow * (int)$(conv_ow) * (int)$(conv_oc)) +\n"
        "                         (postCol * (int)$(conv_oc)) +\n"
        "                         $(kernOutChan));\n"
        "    $(addSynapse, postInd,  $(flipKernRow), $(flipKernCol), $(preChan), $(kernOutChan));\n"
        "}\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::vector<double> &pars)
        {
            const unsigned int conv_kh = (unsigned int)pars[0];
            const unsigned int conv_kw = (unsigned int)pars[1];
            const unsigned int conv_sh = (unsigned int)pars[2];
            const unsigned int conv_sw = (unsigned int)pars[3];
            const unsigned int conv_oc = (unsigned int)pars[9];
            return (conv_kh / conv_sh) * (conv_kw / conv_sw) * conv_oc;
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const std::vector<double> &pars)->std::vector<unsigned int>
        {
            return {(unsigned int)pars[0], (unsigned int)pars[1],
                    (unsigned int)pars[6], (unsigned int)pars[9]};
        });
};
}   // namespace InitSparseConnectivitySnippet
