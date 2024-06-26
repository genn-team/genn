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
#define SET_ROW_BUILD_CODE(CODE) virtual std::string getRowBuildCode() const override{ return CODE; }
#define SET_COL_BUILD_CODE(CODE) virtual std::string getColBuildCode() const override{ return CODE; }

#define SET_HOST_INIT_CODE(CODE) virtual std::string getHostInitCode() const override{ return CODE; }

#define SET_CALC_MAX_ROW_LENGTH_FUNC(FUNC) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return FUNC; }
#define SET_CALC_MAX_COL_LENGTH_FUNC(FUNC) virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const override{ return FUNC; }
#define SET_CALC_KERNEL_SIZE_FUNC(...) virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const override{ return __VA_ARGS__; }

#define SET_MAX_ROW_LENGTH(MAX_ROW_LENGTH) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return [](unsigned int, unsigned int, const std::map<std::string, Type::NumericValue> &){ return MAX_ROW_LENGTH; }; }
#define SET_MAX_COL_LENGTH(MAX_COL_LENGTH) virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const override{ return [](unsigned int, unsigned int, const std::map<std::string, Type::NumericValue> &){ return MAX_COL_LENGTH; }; }

//----------------------------------------------------------------------------
// GeNN::InitSparseConnectivitySnippet::Base
//----------------------------------------------------------------------------
//! Base class for all sparse connectivity initialisation snippets
namespace GeNN::InitSparseConnectivitySnippet
{
class GENN_EXPORT Base : public Snippet::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string getRowBuildCode() const{ return ""; }
    virtual std::string getColBuildCode() const { return ""; }
    virtual std::string getHostInitCode() const{ return ""; }

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
    void validate(const std::map<std::string, Type::NumericValue> &paramValues) const;
};

//----------------------------------------------------------------------------
// Init
//----------------------------------------------------------------------------
class GENN_EXPORT Init : public Snippet::Init<Base>
{
public:
    Init(const Base *snippet, const std::map<std::string, Type::NumericValue> &params);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool isRNGRequired() const;
    bool isHostRNGRequired() const;
    
    const auto &getRowBuildCodeTokens() const{ return m_RowBuildCodeTokens; }
    const auto &getColBuildCodeTokens() const{ return m_ColBuildCodeTokens; }
    const auto &getHostInitCodeTokens() const{ return m_HostInitCodeTokens; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<Transpiler::Token> m_RowBuildCodeTokens;
    std::vector<Transpiler::Token> m_ColBuildCodeTokens;
    std::vector<Transpiler::Token> m_HostInitCodeTokens;
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::Uninitialised
//----------------------------------------------------------------------------
//! Used to mark connectivity as uninitialised - no initialisation code will be run
class Uninitialised : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::Uninitialised);
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::OneToOne
//----------------------------------------------------------------------------
//! Initialises connectivity to a 'one-to-one' diagonal matrix
/*! This snippet has no parameters */
class OneToOne : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::OneToOne);

    SET_ROW_BUILD_CODE(
        "addSynapse(id_pre);\n");

    SET_MAX_ROW_LENGTH(1);
    SET_MAX_COL_LENGTH(1);
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::FixedProbabilityBase
//----------------------------------------------------------------------------
//! Base class for snippets which initialise connectivity with a fixed probability
//! of a synapse existing between a pair of pre and postsynaptic neurons.
class FixedProbabilityBase : public Base
{
public:
    virtual std::string getRowBuildCode() const override = 0;

    SET_PARAMS({"prob"});
    SET_DERIVED_PARAMS({{"probLogRecip", [](const ParamValues &pars, double){ return 1.0 / log(1.0 - pars.at("prob").cast<double>()); }}});

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const ParamValues &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            return binomialInverseCDF(quantile, numPost, pars.at("prob").cast<double>());
        });
    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const ParamValues &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPos times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            return binomialInverseCDF(quantile, numPre, pars.at("prob").cast<double>());
        });
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::FixedProbability
//----------------------------------------------------------------------------
//! Initialises connectivity with a fixed probability of a synapse existing
//! between a pair of pre and postsynaptic neurons.
/*! Whether a synapse exists between a pair of pre and a postsynaptic
    neurons can be modelled using a Bernoulli distribution. While this COULD
    be sampled directly by repeatedly drawing from the uniform distribution,
    this is inefficient. Instead we sample from the geometric distribution
    which describes "the probability distribution of the number of Bernoulli
    trials needed to get one success" -- essentially the distribution of the
    'gaps' between synapses. We do this using the "inversion method"
    described by Devroye (1986) -- essentially inverting the CDF of the
    equivalent continuous distribution (in this case the exponential distribution)
    This snippet takes 1 parameter:

    - \c prob - probability of connection in [0, 1]*/
class FixedProbability : public FixedProbabilityBase
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::FixedProbability);

    SET_ROW_BUILD_CODE(
        "int prevJ = -1;\n"
        "while(true) {\n"
        "   const scalar u = gennrand_uniform();\n"
        "   prevJ += (1 + (int)(log(u) * probLogRecip));\n"
        "   if(prevJ < num_post) {\n"
        "       addSynapse(prevJ + id_post_begin);\n"
        "   }\n"
        "   else {\n"
        "       break;\n"
        "   }\n"
        "}\n");
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::FixedProbabilityNoAutapse
//----------------------------------------------------------------------------
//! Initialises connectivity with a fixed probability of a synapse existing
//! between a pair of pre and postsynaptic neurons. This version ensures there
//! are no autapses - connections between neurons with the same id
//! so should be used for recurrent connections.
/*! Whether a synapse exists between a pair of pre and a postsynaptic
    neurons can be modelled using a Bernoulli distribution. While this COULD
    br sampling directly by repeatedly drawing from the uniform distribution, 
    this is innefficient. Instead we sample from the gemetric distribution 
    which describes "the probability distribution of the number of Bernoulli 
    trials needed to get one success" -- essentially the distribution of the 
    'gaps' between synapses. We do this using the "inversion method"
    described by Devroye (1986) -- essentially inverting the CDF of the
    equivalent continuous distribution (in this case the exponential distribution)
    This snippet takes 1 parameter: 

    - \c prob - probability of connection in [0, 1]*/
class FixedProbabilityNoAutapse : public FixedProbabilityBase
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::FixedProbabilityNoAutapse);

    SET_ROW_BUILD_CODE(
        "int prevJ = -1;\n"
        "while(true) {\n"
        "   int nextJ;\n"
        "   do {\n"
        "       const scalar u = gennrand_uniform();\n"
        "       nextJ = prevJ + (1 + (int)(log(u) * probLogRecip));\n"
        "   } while(nextJ == id_pre);\n"
        "   prevJ = nextJ;\n"
        "   if(prevJ < num_post) {\n"
        "       addSynapse(prevJ + id_post_begin);\n"
        "   }\n"
        "   else {\n"
        "       break;\n"
        "   }\n"
        "}\n");
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::FixedNumberPostWithReplacement
//----------------------------------------------------------------------------
//! Initialises connectivity with a fixed number of random synapses per row.
/*! The postsynaptic targets of the synapses can be initialised in parallel by sampling from the discrete
    uniform distribution. However, to sample connections in ascending order, we sample from the 1st order statistic
    of the uniform distribution -- Beta[1, Npost] -- essentially the next smallest value. In this special case
    this is equivalent to the exponential distribution which can be sampled in constant time using the inversion method.
    This snippet takes 1 parameter:

    - \c num - number of postsynaptic neurons to connect each presynaptic neuron to.*/
class FixedNumberPostWithReplacement : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::FixedNumberPostWithReplacement);

    SET_ROW_BUILD_CODE(
        "scalar x = 0.0;\n"
        "for(unsigned int c = num; c != 0; c--) {\n"
        "   const scalar u = gennrand_uniform();\n"
        "   x += (1.0 - x) * (1.0 - pow(u, 1.0 / (scalar)c));\n"
        "   unsigned int postIdx = (unsigned int)(x * num_post);\n"
        "   postIdx = (postIdx < num_post) ? postIdx : (num_post - 1);\n"
        "   addSynapse(postIdx + id_post_begin);\n"
        "}\n");

    SET_PARAMS({{"num", "unsigned int"}});

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const ParamValues &pars)
        {
            return pars.at("num").cast<unsigned int>();
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const ParamValues &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPost times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            // In each row the number of connections that end up in a column are distributed
            // binomially with n=numConnections and p=1.0 / numPost. As there are numPre rows the total number
            // of connections that end up in each column are distributed binomially with n=numConnections * numPre and p=1.0 / numPost
            return binomialInverseCDF(quantile, pars.at("num").cast<unsigned int>() * numPre, 1.0 / (double)numPost);
        });
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement
//----------------------------------------------------------------------------
//! Initialises connectivity with a total number of random synapses.
//! The first stage in using this connectivity is to determine how many of the total synapses end up in each row.
//! This can be determined by sampling from the multinomial distribution. However, this operation cannot be
//! efficiently parallelised so must be performed on the host and the result passed as an extra global parameter array.
/*! Once the length of each row is determined, the postsynaptic targets of the synapses can be initialised in parallel
    by sampling from the discrete uniform distribution. However, to sample connections in ascending order, we sample
    from the 1st order statistic of the uniform distribution -- Beta[1, Npost] -- essentially the next smallest value.
    In this special case this is equivalent to the exponential distribution which can be sampled in constant time using the inversion method.
    This snippet takes 1 parameter:

    - \c num - total number of synapses to distribute throughout synaptic matrix.*/
class FixedNumberTotalWithReplacement : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement);

    SET_ROW_BUILD_CODE(
        "scalar x = 0.0;\n"
        "for(unsigned int c = preCalcRowLength[(id_pre * num_threads) + id_thread]; c != 0; c--) {\n"
        "   const scalar u = gennrand_uniform();\n"
        "   x += (1.0 - x) * (1.0 - pow(u, 1.0 / (scalar)c));\n"
        "   unsigned int postIdx = (unsigned int)(x * num_post);\n"
        "   postIdx = (postIdx < num_post) ? postIdx : (num_post - 1);\n"
        "   addSynapse(postIdx + id_post_begin);\n"
        "}\n");

    SET_PARAMS({{"num", "unsigned int"}});
    SET_EXTRA_GLOBAL_PARAMS({{"preCalcRowLength", "uint16_t*"}})

    SET_HOST_INIT_CODE(
        "// Allocate pre-calculated row length array\n"
        "allocatepreCalcRowLength(num_pre * num_threads);\n"
        "// Calculate row lengths\n"
        "const size_t numPostPerThread = (num_post + num_threads - 1) / num_threads;\n"
        "const size_t leftOverNeurons = num_post % numPostPerThread;\n"
        "size_t remainingConnections = num;\n"
        "size_t matrixSize = (size_t)num_pre * (size_t)num_post;\n"
        "uint16_t *subRowLengths = preCalcRowLength;\n"
        "// Loop through rows\n"
        "for(size_t i = 0; i < num_pre; i++) {\n"
        "    const bool lastPre = (i == (num_pre - 1));\n"
        "    // Loop through subrows\n"
        "    for(size_t j = 0; j < num_threads; j++) {\n"
        "        const bool lastSubRow = (j == (num_threads - 1));\n"
        "        // If this isn't the last sub-row of the matrix\n"
        "        if(!lastPre || ! lastSubRow) {\n"
        "            // Get length of this subrow\n"
        "            const unsigned int numSubRowNeurons = (leftOverNeurons != 0 && lastSubRow) ? leftOverNeurons : numPostPerThread;\n"
        "            // Calculate probability\n"
        "            const double probability = (double)numSubRowNeurons / (double)matrixSize;\n"
        "            // Sample row length;\n"
        "            const size_t subRowLength = gennrand_binomial(remainingConnections, probability);\n"
        "            // Update counters\n"
        "            remainingConnections -= subRowLength;\n"
        "            matrixSize -= numSubRowNeurons;\n"
        "            // Add row length to array\n"
        "            assert(subRowLength < 0xFFFF);\n"
        "            *subRowLengths++ = (uint16_t)subRowLength;\n"
        "        }\n"
        "    }\n"
        "}\n"
        "// Insert remaining connections into last sub-row\n"
        "*subRowLengths = (uint16_t)remainingConnections;\n"
        "// Push populated row length array\n"
        "pushpreCalcRowLength(num_pre * num_threads);\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const ParamValues &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, pars.at("num").cast<unsigned int>(), (double)numPost / ((double)numPre * (double)numPost));
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const ParamValues &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPost times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPre)/(numPre*numPost)
            // probability of being selected and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, pars.at("num").cast<unsigned int>(), (double)numPre / ((double)numPre * (double)numPost));
        });
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::FixedNumberPreWithReplacement
//----------------------------------------------------------------------------
//! Initialises connectivity with a fixed number of random synapses per column.
/*! No need for ordering here so fine to sample directly from uniform distribution 
    This snippet takes 1 parameter:

    - \c num - number of presynaptic neurons to connect each postsynaptic neuron to.*/
class FixedNumberPreWithReplacement : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::FixedNumberPreWithReplacement);

    SET_COL_BUILD_CODE(
        "for(unsigned int c = num; c != 0; c--) {\n"
        "   const unsigned int idPre = gennrand() % num_pre;\n"
        "   addSynapse(idPre + id_pre_begin);\n"
        "}\n");
 
    SET_PARAMS({{"num", "unsigned int"}});

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const ParamValues &pars)
        {
            // Calculate suitable quantile for 0.9999 chance when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // In each column the number of connections that end up in a row are distributed
            // binomially with n=numConnections and p=1.0 / numPre. As there are numPost columns the total number
            // of connections that end up in each row are distributed binomially with n=numConnections * numPost and p=1.0 / numPre
            return binomialInverseCDF(quantile, pars.at("num").cast<unsigned int>() * numPost, 1.0 / (double)numPre);
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int, unsigned int, const ParamValues &pars)
        {
            return pars.at("num").cast<unsigned int>();
        });
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::Conv2D
//----------------------------------------------------------------------------
//! Initialises 2D convolutional connectivity
//! Row build state variables are used to convert presynaptic neuron index to rows, columns and channels and, 
//! from these, to calculate the range of postsynaptic rows, columns and channels connections will be made within.
/*! This sparse connectivity snippet does not support multiple threads per neuron 
    This snippet takes 12 parameter:

    - \c conv_kh - height of 2D convolution kernel.
    - \c conv_kw - width of 2D convolution kernel.
    - \c conv_sh - height of convolution stride
    - \c conv_sw - width of convolution stride
    - \c conv_padh - width of padding around input
    - \c conv_padw - height of padding around input
    - \c conv_ih - width of input to this convolution
    - \c conv_iw - height of input to this convolution
    - \c conv_ic - number of input channels to this convolution
    - \c conv_oh - width of output from this convolution
    - \c conv_ow - height of output from this convolution
    - \c conv_oc - number of output channels from this convolution

    \note
    ``conv_ih * conv_iw * conv_ic`` should equal the number of neurons in the presynaptic
    neuron population and ``conv_oh * conv_ow * conv_oc`` should equal the number of 
    neurons in the postsynaptic neuron population.*/
class Conv2D : public Base
{
public:
    DECLARE_SNIPPET(Conv2D);

    SET_PARAMS({{"conv_kh", "int"}, {"conv_kw", "int"},
                {"conv_sh", "int"}, {"conv_sw", "int"},
                {"conv_padh", "int"}, {"conv_padw", "int"},
                {"conv_ih", "int"}, {"conv_iw", "int"}, {"conv_ic", "int"},
                {"conv_oh", "int"}, {"conv_ow", "int"}, {"conv_oc", "int"}});

    SET_ROW_BUILD_CODE(
        "const int inRow = (id_pre / conv_ic) / conv_iw;\n"
        "const int inCol = (id_pre / conv_ic) % conv_iw;\n"
        "const int inChan = id_pre % conv_ic;\n"
        "const int maxOutRow = min(conv_oh, max(0, 1 + ((inRow + conv_padh) / conv_sh)));\n"
        "const int minOutCol = min(conv_ow, max(0, 1 + (int)floor((inCol + conv_padw - conv_kw) / (scalar)conv_sw)));\n"
        "const int maxOutCol = min(conv_ow, max(0, 1 + ((inCol + conv_padw) / conv_sw)));\n"
        "int outRow = min(conv_oh, max(0, 1 + (int)floor((inRow + conv_padh - conv_kh) / (scalar)conv_sh)));\n"
        "for(;outRow < maxOutRow; outRow++) {\n"
        "   const int strideRow = (outRow * conv_sh) - conv_padh;\n"
        "   const int kernRow = inRow - strideRow;\n"
        "   for(int outCol = minOutCol; outCol < maxOutCol; outCol++) {\n"
        "       const int strideCol = (outCol * conv_sw) - conv_padw;\n"
        "       const int kernCol = inCol - strideCol;\n"
        "       for(int outChan = 0; outChan < conv_oc; outChan++) {\n"
        "           const int idPost = ((outRow * conv_ow * conv_oc) +\n"
        "                                (outCol * conv_oc) +\n"
        "                                outChan);\n"
        "            addSynapse(idPost, kernRow, kernCol, inChan, outChan);\n"
        "       }\n"
        "   }\n"
        "}\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const ParamValues &pars)
        {
            return ((unsigned int)std::ceil(pars.at("conv_kh").cast<double>() / pars.at("conv_sh").cast<double>()) 
                    * (unsigned int)std::ceil(pars.at("conv_kw").cast<double>() / pars.at("conv_sw").cast<double>()) 
                    * pars.at("conv_oc").cast<unsigned int>());
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const ParamValues &pars)->std::vector<unsigned int>
        {
            return {pars.at("conv_kh").cast<unsigned int>(), pars.at("conv_kw").cast<unsigned int>(),
                    pars.at("conv_ic").cast<unsigned int>(), pars.at("conv_oc").cast<unsigned int>()};
        });
};
}   // namespace GeNN::InitSparseConnectivitySnippet
