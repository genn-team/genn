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
#define SET_DIAGONAL_STATE_VARS(...) virtual ParamValVec getDiagonalStateVars() const override{ return __VA_ARGS__; }
#define SET_ROW_STATE_VARS(...) virtual ParamValVec getRowStateVars() const override{ return __VA_ARGS__; }

#define SET_CALC_MAX_ROW_LENGTH_FUNC(FUNC) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return FUNC; }
#define SET_CALC_MAX_COL_LENGTH_FUNC(FUNC) virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const override{ return FUNC; }
#define SET_CALC_KERNEL_SIZE_FUNC(...) virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const override{ return __VA_ARGS__; }

#define SET_MAX_ROW_LENGTH(MAX_ROW_LENGTH) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return [](unsigned int, unsigned int, const std::vector<double> &){ return MAX_ROW_LENGTH; }; }
#define SET_MAX_COL_LENGTH(MAX_COL_LENGTH) virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const override{ return [](unsigned int, unsigned int, const std::vector<double> &){ return MAX_COL_LENGTH; }; }

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
    virtual ParamValVec getDiagonalStateVars() const { return {}; }
    virtual ParamValVec getRowStateVars() const { return {}; }

    //! Get function to calculate the maximum row length of this connector based on the parameters and the size of the pre and postsynaptic population
    virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const{ return CalcMaxLengthFunc(); }

    //! Get function to calculate the maximum column length of this connector based on the parameters and the size of the pre and postsynaptic population
    virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const{ return CalcMaxLengthFunc(); }

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
    DECLARE_SNIPPET(Conv2D, 12);

    SET_PARAM_NAMES({"conv_kh", "conv_kw",
                     "conv_sh", "conv_sw",
                     "conv_padh", "conv_padw",
                     "conv_ih", "conv_iw", "conv_ic",
                     "conv_oh", "conv_ow", "conv_oc"});


    SET_DIAGONAL_STATE_VARS({{"kernRow", "int", "($(id_diag) / (int)$(conv_oc)) / (int)$(conv_kw)"},
                             {"kernCol", "int", "($(id_diag) / (int)$(conv_oc)) % (int)$(conv_kw)"},
                             {"kernOutChan", "int", "$(id_diag) % (int)$(conv_oc)"},
                             {"flipKernRow", "int", "(int)$(conv_kh) - $(kernRow) - 1"},
                             {"flipKernCol", "int", "(int)$(conv_kw) - $(kernCol) - 1"},
                             {"kernelInd", "int", "($(flipKernRow) * (int)$(conv_kw) * (int)$(conv_ic) * (int)$(conv_oc)) + ($(flipKernCol) * (int)$(conv_ic) * (int)$(conv_oc)) + $(kernOutChan);"}});

    SET_ROW_STATE_VARS({{"preRow", "int", "($(id_pre) / (int)$(conv_ic)) / (int)$(conv_iw)" },
                        {"preCol", "int", "($(id_pre) / (int)$(conv_ic)) % (int)$(conv_iw)" },
                        {"preChan", "int", "($(id_pre) % (int)$(conv_ic); }})"}});

    SET_DIAGONAL_BUILD_CODE(
        "// If we haven't gone off edge of output\n"
        "const int postRow = preRow + $(kernRow) - ConvB;\n"
        "const int postCol = preCol + $(kernCol) - ConvB;\n"
        "if(postRow >= 0 && postCol >= 0 && postRow < ConvO && postCol < ConvO) {\n"
        "    const float kernelVal = d_kernel[kernelInd + (preChan * ConvOC)];\n"
        "    // Calculate postsynaptic index\n"
        "    const int postInd = ((postRow * ConvO * ConvOC) +\n"
        "                            (postCol * ConvOC) +\n"
        "                            kernOutChan);\n"
        "    $(addSynapse, postInd,  $(flipKernRow), $(flipKernCol), $(preChan), $(kernOutChan));\n"
        "}\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::vector<double> &pars)
        {
            const unsigned int conv_kh = (unsigned int)pars[0];
            const unsigned int conv_kw = (unsigned int)pars[1];
            const unsigned int conv_sh = (unsigned int)pars[2];
            const unsigned int conv_sw = (unsigned int)pars[3];
            const unsigned int conv_oc = (unsigned int)pars[11];
            return (conv_kh / conv_sh) * (conv_kw / conv_sw) * conv_oc;
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const std::vector<double> &pars)->std::vector<unsigned int>
        {
            return {(unsigned int)pars[0], (unsigned int)pars[1],
                    (unsigned int)pars[8], (unsigned int)pars[11]};
        });
};
}   // namespace InitSparseConnectivitySnippet
