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

#define SET_MAX_ROW_LENGTH(MAX_ROW_LENGTH) virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override{ return [](unsigned int, unsigned int, const std::unordered_map<std::string, double> &){ return MAX_ROW_LENGTH; }; }
#define SET_MAX_COL_LENGTH(MAX_COL_LENGTH) virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const override{ return [](unsigned int, unsigned int, const std::unordered_map<std::string, double> &){ return MAX_COL_LENGTH; }; }

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
    void validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues) const;
};

//----------------------------------------------------------------------------
// Init
//----------------------------------------------------------------------------
class GENN_EXPORT Init : public Snippet::Init<Base>
{
public:
    Init(const Base *snippet, const std::unordered_map<std::string, Type::NumericValue> &params);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool isRNGRequired() const;
    bool isHostRNGRequired() const;
    
    const std::vector<Transpiler::Token> &getRowBuildCodeTokens() const{ return m_RowBuildCodeTokens; }
    const std::vector<Transpiler::Token> &getColBuildCodeTokens() const{ return m_ColBuildCodeTokens; }
    const std::vector<Transpiler::Token> &getHostInitCodeTokens() const{ return m_HostInitCodeTokens; }

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

    SET_PARAM_NAMES({"prob"});
    SET_DERIVED_PARAMS({{"probLogRecip", [](const auto &pars, double){ return 1.0 / log(1.0 - pars.at("prob")); }}});

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::unordered_map<std::string, double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            return binomialInverseCDF(quantile, numPost, pars.at("prob"));
        });
    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::unordered_map<std::string, double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPos times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            return binomialInverseCDF(quantile, numPre, pars.at("prob"));
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
    equivalent continuous distribution (in this case the exponential distribution)*/
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
    equivalent continuous distribution (in this case the exponential distribution)*/
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
    this is equivalent to the exponential distribution which can be sampled in constant time using the inversion method.*/
class FixedNumberPostWithReplacement : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::FixedNumberPostWithReplacement);

    SET_ROW_BUILD_CODE(
        "scalar x = 0.0;\n"
        "for(unsigned int c = (unsigned int)rowLength; c != 0; c--) {\n"
        "   const scalar u = gennrand_uniform();\n"
        "   x += (1.0 - x) * (1.0 - pow(u, 1.0 / (scalar)c));\n"
        "   unsigned int postIdx = (unsigned int)(x * num_post);\n"
        "   postIdx = (postIdx < num_post) ? postIdx : (num_post - 1);\n"
        "   addSynapse(postIdx + id_post_begin);\n"
        "}\n");

    SET_PARAM_NAMES({"rowLength"});

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::unordered_map<std::string, double> &pars)
        {
            return (unsigned int)pars.at("rowLength");
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::unordered_map<std::string, double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPost times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            // In each row the number of connections that end up in a column are distributed
            // binomially with n=numConnections and p=1.0 / numPost. As there are numPre rows the total number
            // of connections that end up in each column are distributed binomially with n=numConnections * numPre and p=1.0 / numPost
            return binomialInverseCDF(quantile, (unsigned int)pars.at("rowLength") * numPre, 1.0 / (double)numPost);
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
    In this special case this is equivalent to the exponential distribution which can be sampled in constant time using the inversion method.*/
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

    SET_PARAM_NAMES({"total"});
    SET_EXTRA_GLOBAL_PARAMS({{"preCalcRowLength", "uint16_t*"}})

    SET_HOST_INIT_CODE(
        "// Allocate pre-calculated row length array\n"
        "allocatepreCalcRowLength(num_pre * num_threads);\n"
        "// Calculate row lengths\n"
        "const size_t numPostPerThread = (num_post + num_threads - 1) / num_threads;\n"
        "const size_t leftOverNeurons = num_post % numPostPerThread;\n"
        "size_t remainingConnections = total;\n"
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
        [](unsigned int numPre, unsigned int numPost, const std::unordered_map<std::string, double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, (unsigned int)pars.at("total"), (double)numPost / ((double)numPre * (double)numPost));
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::unordered_map<std::string, double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPost times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPre)/(numPre*numPost)
            // probability of being selected and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, (unsigned int)pars.at("total"), (double)numPre / ((double)numPre * (double)numPost));
        });
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::FixedNumberPreWithReplacement
//----------------------------------------------------------------------------
//! Initialises connectivity with a fixed number of random synapses per column.
/*! No need for ordering here so fine to sample directly from uniform distribution */
class FixedNumberPreWithReplacement : public Base
{
public:
    DECLARE_SNIPPET(InitSparseConnectivitySnippet::FixedNumberPreWithReplacement);

    SET_COL_BUILD_CODE(
        "for(unsigned int c = colLength; c != 0; c--) {\n"
        "   const unsigned int idPre = (unsigned int)ceil(gennrand_uniform() * num_pre) - 1;\n"
        "   addSynapse(idPre + id_pre_begin);\n"
        "}\n");
 
    SET_PARAM_NAMES({"colLength"});

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::unordered_map<std::string, double> &pars)
        {
            // Calculate suitable quantile for 0.9999 chance when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // In each column the number of connections that end up in a row are distributed
            // binomially with n=numConnections and p=1.0 / numPre. As there are numPost columns the total number
            // of connections that end up in each row are distributed binomially with n=numConnections * numPost and p=1.0 / numPre
            return binomialInverseCDF(quantile, (unsigned int)pars.at("colLength") * numPost, 1.0 / (double)numPre);
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::unordered_map<std::string, double> &pars)
        {
            return (unsigned int)pars.at("colLength");
        });
};

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::Conv2D
//----------------------------------------------------------------------------
//! Initialises convolutional connectivity
//! Row build state variables are used to convert presynaptic neuron index to rows, columns and channels and, 
//! from these, to calculate the range of postsynaptic rows, columns and channels connections will be made within.
/*! This sparse connectivity snippet does not support multiple threads per neuron */
class Conv2D : public Base
{
public:
    DECLARE_SNIPPET(Conv2D);

    SET_PARAM_NAMES({"conv_kh", "conv_kw",
                     "conv_sh", "conv_sw",
                     "conv_padh", "conv_padw",
                     "conv_ih", "conv_iw", "conv_ic",
                     "conv_oh", "conv_ow", "conv_oc"});

    SET_ROW_BUILD_CODE(
        "const int inRow = (id_pre / (int)conv_ic) / (int)conv_iw;\n"
        "const int inCol = (id_pre / (int)conv_ic) % (int)conv_iw;\n"
        "const int inChan = id_pre % (int)conv_ic;\n"
        "const int maxOutRow = min((int)conv_oh, max(0, 1 + ((inRow + (int)conv_padh) / (int)conv_sh)));\n"
        "const int minOutCol = min((int)conv_ow, max(0, 1 + (int)floor((inCol + conv_padw - conv_kw) / conv_sw)));\n"
        "const int maxOutCol = min((int)conv_ow, max(0, 1 + ((inCol + (int)conv_padw) / (int)conv_sw)));\n"
        "int outRow = min((int)conv_oh, max(0, 1 + (int)floor((inRow + conv_padh - conv_kh) / conv_sh)));\n"
        "for(;outRow < maxOutRow; outRow++) {\n"
        "   const int strideRow = (outRow * (int)conv_sh) - (int)conv_padh;\n"
        "   const int kernRow = inRow - strideRow;\n"
        "   for(int outCol = minOutCol; outCol < maxOutCol; outCol++) {\n"
        "       const int strideCol = (outCol * (int)conv_sw) - (int)conv_padw;\n"
        "       const int kernCol = inCol - strideCol;\n"
        "       for(unsigned int outChan = 0; outChan < (unsigned int)conv_oc; outChan++) {\n"
        "           const int idPost = ((outRow * (int)conv_ow * (int)conv_oc) +\n"
        "                                (outCol * (int)conv_oc) +\n"
        "                                outChan);\n"
        "            addSynapse(idPost, kernRow, kernCol, inChan, outChan);\n"
        "       }\n"
        "   }\n"
        "}\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::unordered_map<std::string, double> &pars)
        {
            return ((unsigned int)std::ceil(pars.at("conv_kh") / pars.at("conv_sh")) 
                    * (unsigned int)std::ceil(pars.at("conv_kw") / pars.at("conv_sw")) 
                    * (unsigned int)pars.at("conv_oc"));
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const std::unordered_map<std::string, double> &pars)->std::vector<unsigned int>
        {
            return {(unsigned int)pars.at("conv_kh"), (unsigned int)pars.at("conv_kw"),
                    (unsigned int)pars.at("conv_ic"), (unsigned int)pars.at("conv_oc")};
        });
};
}   // namespace GeNN::InitSparseConnectivitySnippet
