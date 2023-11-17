#pragma once

// Standard C++ includes
#include <functional>
#include <unordered_map>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// GeNN includes
#include "binomial.h"
#include "snippet.h"

// GeNN transpiler includes
#include "transpiler/token.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_DIAGONAL_BUILD_CODE(CODE) virtual std::string getDiagonalBuildCode() const override{ return CODE; }

#define SET_CALC_MAX_ROW_LENGTH_FUNC(FUNC) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return FUNC; }
#define SET_CALC_KERNEL_SIZE_FUNC(...) virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const override{ return __VA_ARGS__; }

#define SET_MAX_ROW_LENGTH(MAX_ROW_LENGTH) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return [](unsigned int, unsigned int, const std::unordered_map<std::string, double> &){ return MAX_ROW_LENGTH; }; }

//----------------------------------------------------------------------------
// GeNN:::InitToeplitzConnectivitySnippet::Base
//----------------------------------------------------------------------------
//! Base class for all toeplitz connectivity initialisation snippets
namespace GeNN::InitToeplitzConnectivitySnippet
{
class GENN_EXPORT Base : public Snippet::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string getDiagonalBuildCode() const{ return ""; }

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
    void validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues) const;
};

//----------------------------------------------------------------------------
// Init
//----------------------------------------------------------------------------
class GENN_EXPORT Init : public Snippet::Init<InitToeplitzConnectivitySnippet::Base>
{
public:
    Init(const Base *snippet, const std::unordered_map<std::string, Type::NumericValue> &params);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool isRNGRequired() const;
    
    const std::vector<Transpiler::Token> &getDiagonalBuildCodeTokens() const{ return m_DiagonalBuildCodeTokens; }
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<Transpiler::Token> m_DiagonalBuildCodeTokens;

};

//----------------------------------------------------------------------------
// GeNN::InitToeplitzConnectivitySnippet::Uninitialised
//----------------------------------------------------------------------------
//! Used to mark connectivity as uninitialised - no initialisation code will be run
class Uninitialised : public Base
{
public:
    DECLARE_SNIPPET(InitToeplitzConnectivitySnippet::Uninitialised);
};

//----------------------------------------------------------------------------
// GeNN::InitToeplitzConnectivitySnippet::Conv2D
//----------------------------------------------------------------------------
//! Initialises convolutional connectivity
//! Row build state variables are used to convert presynaptic neuron index to rows, columns and channels and, 
//! from these, to calculate the range of postsynaptic rows, columns and channels connections will be made within.
class Conv2D : public Base
{
public:
    DECLARE_SNIPPET(Conv2D);

    SET_PARAM_NAMES({"conv_kh", "conv_kw",
                     "conv_ih", "conv_iw", "conv_ic",
                     "conv_oh", "conv_ow", "conv_oc"});
    SET_DERIVED_PARAMS({{"conv_bw", [](const auto &pars, double){ return (((int)pars.at("conv_iw") + (int)pars.at("conv_kw") - 1) - (int)pars.at("conv_ow")) / 2; }},
                        {"conv_bh", [](const auto &pars, double){ return (((int)pars.at("conv_ih") + (int)pars.at("conv_kh") - 1) - (int)pars.at("conv_oh")) / 2; }}});

    SET_DIAGONAL_BUILD_CODE(
        "const int kernRow = (id_diag / (int)conv_oc) / (int)conv_kw;\n"
        "const int kernCol = (id_diag / (int)conv_oc) % (int)conv_kw;\n"
        "const int kernOutChan = id_diag % (int)conv_oc;\n"
        "const int flipKernRow = (int)conv_kh - kernRow - 1;\n"
        "const int flipKernCol = (int)conv_kw - kernCol - 1;\n"
        "for_each_synapse {\n"
        "    const int preRow = (id_pre / (int)conv_ic) / (int)conv_iw;\n"
        "    const int preCol = (id_pre / (int)conv_ic) % (int)conv_iw;\n"
        "    const int preChan = id_pre % (int)conv_ic;\n"
        "    // If we haven't gone off edge of output\n"
        "    const int postRow = preRow + kernRow - (int)conv_bh;\n"
        "    const int postCol = preCol + kernCol - (int)conv_bw;\n"
        "    if(postRow >= 0 && postCol >= 0 && postRow < (int)conv_oh && postCol < (int)conv_ow) {\n"
        "        // Calculate postsynaptic index\n"
        "        const int postInd = ((postRow * (int)conv_ow * (int)conv_oc) +\n"
        "                             (postCol * (int)conv_oc) +\n"
        "                              kernOutChan);\n"
        "        addSynapse(postInd,  flipKernRow, flipKernCol, preChan, kernOutChan);\n"
        "     }\n"
        "}\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::unordered_map<std::string, double> &pars)
        {
            return ((unsigned int)pars.at("conv_kh") * (unsigned int)pars.at("conv_kw") * (unsigned int)pars.at("conv_oc"));
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const std::unordered_map<std::string, double> &pars)->std::vector<unsigned int>
        {
            return {(unsigned int)pars.at("conv_kh"), (unsigned int)pars.at("conv_kw"),
                    (unsigned int)pars.at("conv_ic"), (unsigned int)pars.at("conv_oc")};
        });
};

//----------------------------------------------------------------------------
// GeNN::InitToeplitzConnectivitySnippet::AvgPoolConv2D
//----------------------------------------------------------------------------
//! Initialises convolutional connectivity preceded by averaging pooling
//! Row build state variables are used to convert presynaptic neuron index to rows, columns and channels and, 
//! from these, to calculate the range of postsynaptic rows, columns and channels connections will be made within.
class AvgPoolConv2D : public Base
{
public:
    DECLARE_SNIPPET(AvgPoolConv2D);

    SET_PARAM_NAMES({"conv_kh", "conv_kw",
                     "pool_kh", "pool_kw",
                     "pool_sh", "pool_sw",
                     "pool_ih", "pool_iw", "pool_ic",
                     "conv_oh", "conv_ow", "conv_oc"});
    SET_DERIVED_PARAMS({{"conv_bw", [](const auto &pars, double){ return (int(ceil((pars.at("pool_iw") - pars.at("pool_kw") + 1.0) / pars.at("pool_sw"))) + (int)pars.at("conv_kw") - 1 - (int)pars.at("conv_ow")) / 2; }},
                        {"conv_bh", [](const auto &pars, double){ return (int(ceil((pars.at("pool_ih") - pars.at("pool_kh") + 1.0) / pars.at("pool_sh"))) + (int)pars.at("conv_h") - 1 - (int)pars.at("conv_oh")) / 2; }}});

    SET_DIAGONAL_BUILD_CODE(
        "const int kernRow = (id_diag / (int)conv_oc) / (int)conv_kw;\n"
        "const int kernCol = (id_diag / (int)conv_oc) % (int)conv_kw;\n"
        "const int kernOutChan = id_diag % (int)conv_oc;\n"
        "const int flipKernRow = (int)conv_kh - kernRow - 1;\n"
        "const int flipKernCol = (int)conv_kw - kernCol - 1;\n"
        "for_each_synapse {\n"
        "    // Convert spike ID into row, column and channel going INTO pool\n"
        "    const int prePoolInRow = (id_pre / (int)pool_ic) / (int)pool_iw;\n"
        "    const int prePoolInCol = (id_pre / (int)pool_ic) % (int)pool_iw;\n"
        "    const int preChan = id_pre % (int)pool_ic;\n"
        "    // Calculate row and column going OUT of pool\n"
        "    const int poolPreOutRow = prePoolInRow / (int)pool_sh;\n"
        "    const int poolStrideRow = poolPreOutRow * (int)pool_sh;\n"
        "    const int poolPreOutCol = prePoolInCol / (int)pool_sw;\n"
        "    const int poolStrideCol = poolPreOutCol * (int)pool_sw;\n"
        "    if(prePoolInRow < (poolStrideRow + (int)pool_kh) && prePoolInCol < (poolStrideCol + (int)pool_kw)) {\n"
        "       // If we haven't gone off edge of output\n"
        "       const int postRow = poolPreOutRow + kernRow - (int)conv_bh;\n"
        "       const int postCol = poolPreOutCol + kernCol - (int)conv_bw;\n"
        "       if(postRow >= 0 && postCol >= 0 && postRow < (int)conv_oh && postCol < (int)conv_ow) {\n"
        "           // Calculate postsynaptic index\n"
        "           const int postInd = ((postRow * (int)conv_ow * (int)conv_oc) +\n"
        "                                 (postCol * (int)conv_oc) +\n"
        "                                 kernOutChan);\n"
        "           addSynapse(postInd,  flipKernRow, flipKernCol, preChan, kernOutChan);\n"
        "       }\n"
        "    }\n"
        "}\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::unordered_map<std::string, double> &pars)
        {
            return ((unsigned int)pars.at("conv_kh") * (unsigned int)pars.at("conv_kw") * (unsigned int)pars.at("conv_oc"));
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const std::unordered_map<std::string, double> &pars)->std::vector<unsigned int>
        {
            return {(unsigned int)pars.at("conv_kh"), (unsigned int)pars.at("conv_kw"),
                    (unsigned int)pars.at("pool_ic"), (unsigned int)pars.at("conv_oc")};
        });
};
}   // namespace GeNN::InitToeplitzConnectivitySnippet
