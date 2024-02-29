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

// GeNN transpiler includes
#include "transpiler/token.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_DIAGONAL_BUILD_CODE(CODE) virtual std::string getDiagonalBuildCode() const override{ return CODE; }

#define SET_CALC_MAX_ROW_LENGTH_FUNC(FUNC) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return FUNC; }
#define SET_CALC_KERNEL_SIZE_FUNC(...) virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const override{ return __VA_ARGS__; }

#define SET_MAX_ROW_LENGTH(MAX_ROW_LENGTH) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return [](unsigned int, unsigned int, const std::map<std::string, Type::NumericValue> &){ return MAX_ROW_LENGTH; }; }

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
    void validate(const std::map<std::string, Type::NumericValue> &paramValues) const;
};

//----------------------------------------------------------------------------
// Init
//----------------------------------------------------------------------------
class GENN_EXPORT Init : public Snippet::Init<InitToeplitzConnectivitySnippet::Base>
{
public:
    Init(const Base *snippet, const std::map<std::string, Type::NumericValue> &params);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool isRNGRequired() const;
    
    const auto &getDiagonalBuildCodeTokens() const{ return m_DiagonalBuildCodeTokens; }
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

    SET_PARAMS({{"conv_kh", "int"}, {"conv_kw", "int"},
                {"conv_ih", "int"}, {"conv_iw", "int"}, {"conv_ic", "int"},
                {"conv_oh", "int"}, {"conv_ow", "int"}, {"conv_oc", "int"}});
    SET_DERIVED_PARAMS({{"conv_bw", [](const ParamValues &pars, double){ return ((pars.at("conv_iw").cast<int>() + pars.at("conv_kw").cast<int>() - 1) - pars.at("conv_ow").cast<int>()) / 2; }},
                        {"conv_bh", [](const ParamValues &pars, double){ return ((pars.at("conv_ih").cast<int>() + pars.at("conv_kh").cast<int>() - 1) - pars.at("conv_oh").cast<int>()) / 2; }}});

    SET_DIAGONAL_BUILD_CODE(
        "const int kernRow = (id_diag / conv_oc) / conv_kw;\n"
        "const int kernCol = (id_diag / conv_oc) % conv_kw;\n"
        "const int kernOutChan = id_diag % conv_oc;\n"
        "const int flipKernRow = conv_kh - kernRow - 1;\n"
        "const int flipKernCol = conv_kw - kernCol - 1;\n"
        "for_each_synapse {\n"
        "    const int preRow = (id_pre / conv_ic) / conv_iw;\n"
        "    const int preCol = (id_pre / conv_ic) % conv_iw;\n"
        "    const int preChan = id_pre % conv_ic;\n"
        "    // If we haven't gone off edge of output\n"
        "    const int postRow = preRow + kernRow - conv_bh;\n"
        "    const int postCol = preCol + kernCol - conv_bw;\n"
        "    if(postRow >= 0 && postCol >= 0 && postRow < conv_oh && postCol < conv_ow) {\n"
        "        // Calculate postsynaptic index\n"
        "        const int postInd = ((postRow * conv_ow * conv_oc) +\n"
        "                             (postCol * conv_oc) +\n"
        "                              kernOutChan);\n"
        "        addSynapse(postInd,  flipKernRow, flipKernCol, preChan, kernOutChan);\n"
        "     }\n"
        "}\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const ParamValues &pars)
        {
            return (pars.at("conv_kh").cast<unsigned int>() * pars.at("conv_kw").cast<unsigned int>() * pars.at("conv_oc").cast<unsigned int>());
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const ParamValues &pars)->std::vector<unsigned int>
        {
            return {pars.at("conv_kh").cast<unsigned int>(), pars.at("conv_kw").cast<unsigned int>(),
                    pars.at("conv_ic").cast<unsigned int>(), pars.at("conv_oc").cast<unsigned int>()};
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

    SET_PARAMS({{"conv_kh", "int"}, {"conv_kw", "int"},
                {"pool_kh", "int"}, {"pool_kw", "int"},
                {"pool_sh", "int"}, {"pool_sw", "int"},
                {"pool_ih", "int"}, {"pool_iw", "int"}, {"pool_ic", "int"},
                {"conv_oh", "int"}, {"conv_ow", "int"}, {"conv_oc", "int"}});
    SET_DERIVED_PARAMS({{"conv_bw", [](const ParamValues &pars, double){ return (int(ceil((pars.at("pool_iw").cast<double>() - pars.at("pool_kw").cast<double>() + 1.0) / pars.at("pool_sw").cast<double>())) + pars.at("conv_kw").cast<int>() - 1 - pars.at("conv_ow").cast<int>()) / 2; }},
                        {"conv_bh", [](const ParamValues &pars, double){ return (int(ceil((pars.at("pool_ih").cast<double>() - pars.at("pool_kh").cast<double>() + 1.0) / pars.at("pool_sh").cast<double>())) + pars.at("conv_kh").cast<int>() - 1 - pars.at("conv_oh").cast<int>()) / 2; }}});

    SET_DIAGONAL_BUILD_CODE(
        "const int kernRow = (id_diag / conv_oc) / conv_kw;\n"
        "const int kernCol = (id_diag / conv_oc) % conv_kw;\n"
        "const int kernOutChan = id_diag % conv_oc;\n"
        "const int flipKernRow = conv_kh - kernRow - 1;\n"
        "const int flipKernCol = conv_kw - kernCol - 1;\n"
        "for_each_synapse {\n"
        "    // Convert spike ID into row, column and channel going INTO pool\n"
        "    const int prePoolInRow = (id_pre / pool_ic) / pool_iw;\n"
        "    const int prePoolInCol = (id_pre / pool_ic) % pool_iw;\n"
        "    const int preChan = id_pre % pool_ic;\n"
        "    // Calculate row and column going OUT of pool\n"
        "    const int poolPreOutRow = prePoolInRow / pool_sh;\n"
        "    const int poolStrideRow = poolPreOutRow * pool_sh;\n"
        "    const int poolPreOutCol = prePoolInCol / pool_sw;\n"
        "    const int poolStrideCol = poolPreOutCol * pool_sw;\n"
        "    if(prePoolInRow < (poolStrideRow + pool_kh) && prePoolInCol < (poolStrideCol + pool_kw)) {\n"
        "       // If we haven't gone off edge of output\n"
        "       const int postRow = poolPreOutRow + kernRow - conv_bh;\n"
        "       const int postCol = poolPreOutCol + kernCol - conv_bw;\n"
        "       if(postRow >= 0 && postCol >= 0 && postRow < conv_oh && postCol < conv_ow) {\n"
        "           // Calculate postsynaptic index\n"
        "           const int postInd = ((postRow * conv_ow * conv_oc) +\n"
        "                                 (postCol * conv_oc) +\n"
        "                                 kernOutChan);\n"
        "           addSynapse(postInd,  flipKernRow, flipKernCol, preChan, kernOutChan);\n"
        "       }\n"
        "    }\n"
        "}\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const ParamValues &pars)
        {
            return (pars.at("conv_kh").cast<int>() * pars.at("conv_kw").cast<int>() * pars.at("conv_oc").cast<int>());
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const ParamValues &pars)->std::vector<unsigned int>
        {
            return {pars.at("conv_kh").cast<unsigned int>(), pars.at("conv_kw").cast<unsigned int>(),
                    pars.at("pool_ic").cast<unsigned int>(), pars.at("conv_oc").cast<unsigned int>()};
        });
};
}   // namespace GeNN::InitToeplitzConnectivitySnippet
