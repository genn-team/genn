#pragma once

// Standard includes
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>
#include <vector>

// GeNN includes
#include "variableMode.h"

using namespace std;

// Forward declarations
class NNmodel;
class SynapseGroup;

namespace NeuronModels
{
    class Base;
}

namespace NewModels
{
    class VarInit;
}

//--------------------------------------------------------------------------
// GenericFunction
//--------------------------------------------------------------------------
//! Immutable structure for specifying the name and number of
//! arguments of a generic funcion e.g. gennrand_uniform
struct GenericFunction
{
    //! Generic name used to refer to function in user code
    const std::string genericName;

    //! Number of function arguments
    const unsigned int numArguments;
};

//--------------------------------------------------------------------------
// FunctionTemplate
//--------------------------------------------------------------------------
//! Immutable structure for specifying how to implement
//! a generic function e.g. gennrand_uniform
/*! **NOTE** for the sake of easy initialisation first two parameters of GenericFunction are repeated (C++17 fixes) */
struct FunctionTemplate
{
    // **HACK** while GCC and CLang automatically generate this fine/don't require it, VS2013 seems to need it
    FunctionTemplate operator = (const FunctionTemplate &o)
    {
        return FunctionTemplate{o.genericName, o.numArguments, o.doublePrecisionTemplate, o.singlePrecisionTemplate};
    }

    //! Generic name used to refer to function in user code
    const std::string genericName;

    //! Number of function arguments
    const unsigned int numArguments;

    //! The function template (for use with ::functionSubstitute) used when model uses double precision
    const std::string doublePrecisionTemplate;

    //! The function template (for use with ::functionSubstitute) used when model uses single precision
    const std::string singlePrecisionTemplate;
};

//--------------------------------------------------------------------------
// PairKeyConstIter
//--------------------------------------------------------------------------
//! Custom iterator for iterating through the keys of containers containing pairs
template<typename BaseIter>
class PairKeyConstIter : public BaseIter
{
private:
    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    typedef typename BaseIter::value_type::first_type KeyType;

public:
    PairKeyConstIter() {}
    PairKeyConstIter(BaseIter iter) : BaseIter(iter) {}

    //--------------------------------------------------------------------------
    // Operators
    //--------------------------------------------------------------------------
    const KeyType *operator -> () const
    {
        return (const KeyType *) &(BaseIter::operator -> ( )->first);
    }

    const KeyType &operator * () const
    {
        return BaseIter::operator * ( ).first;
    }
};

//! Helper function for creating a PairKeyConstIter from an iterator
template<typename BaseIter>
inline PairKeyConstIter<BaseIter> GetPairKeyConstIter(BaseIter iter)
{
    return iter;
}

//--------------------------------------------------------------------------
//! \brief CUDA implementations of standard functions
//--------------------------------------------------------------------------
const std::vector<FunctionTemplate> cudaFunctions = {
    {"gennrand_uniform", 0, "curand_uniform_double($(rng))", "curand_uniform($(rng))"},
    {"gennrand_normal", 0, "curand_normal_double($(rng))", "curand_normal($(rng))"},
    {"gennrand_exponential", 0, "exponentialDistFloat($(rng))", "exponentialDistDouble($(rng))"},
    {"gennrand_log_normal", 2, "curand_log_normal_double($(rng), $(0), $(1))", "curand_log_normal_float($(rng), $(0), $(1))"},
};

//--------------------------------------------------------------------------
//! \brief CPU implementations of standard functions
//--------------------------------------------------------------------------
const std::vector<FunctionTemplate> cpuFunctions = {
    {"gennrand_uniform", 0, "standardUniformDistribution($(rng))", "standardUniformDistribution($(rng))"},
    {"gennrand_normal", 0, "standardNormalDistribution($(rng))", "standardNormalDistribution($(rng))"},
    {"gennrand_exponential", 0, "standardExponentialDistribution($(rng))", "standardExponentialDistribution($(rng))"},
    {"gennrand_log_normal", 2, "std::lognormal_distribution<double>($(0), $(1))($(rng))", "std::lognormal_distribution<float>($(0), $(1))($(rng))"},
};

//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------
void substitute(string &s, const string &trg, const string &rep);


//--------------------------------------------------------------------------
//! \brief Does the code string contain any functions requiring random number generator
//--------------------------------------------------------------------------
bool isRNGRequired(const std::string &code);

//--------------------------------------------------------------------------
//! \brief Does the model with the vectors of variable initialisers and modes require an RNG for the specified init location i.e. host or device
//--------------------------------------------------------------------------
bool isInitRNGRequired(const std::vector<NewModels::VarInit> &varInitialisers, const std::vector<VarMode> &varModes,
                       VarInit initLocation);

//--------------------------------------------------------------------------
/*! \brief This function substitutes function calls in the form:
 *
 *  $(functionName, parameter1, param2Function(0.12, "string"))
 *
 * with replacement templates in the form:
 *
 *  actualFunction(CONSTANT, $(0), $(1))
 *
 */
//--------------------------------------------------------------------------
void functionSubstitute(std::string &code, const std::string &funcName,
                        unsigned int numParams, const std::string &replaceFuncTemplate);

//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets.
//--------------------------------------------------------------------------
template<typename NameIter>
inline void name_substitutions(string &code, const string &prefix, NameIter namesBegin, NameIter namesEnd, const string &postfix= "", const string &ext = "")
{
    for (NameIter n = namesBegin; n != namesEnd; n++) {
        substitute(code,
                   "$(" + *n + ext + ")",
                   prefix + *n + postfix);
    }
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets.
//--------------------------------------------------------------------------
inline void name_substitutions(string &code, const string &prefix, const vector<string> &names, const string &postfix= "", const string &ext = "")
{
    name_substitutions(code, prefix, names.cbegin(), names.cend(), postfix, ext);
}


template<class T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
void writePreciseString(std::ostream &os, T value)
{
    // Cache previous precision
    const std::streamsize previousPrecision = os.precision();

    // Set scientific formatting
    os << std::scientific;

    // Set precision to what is required to fully represent T
    os << std::setprecision(std::numeric_limits<T>::max_digits10);

    // Write value to stream
    os << value;

    // Reset to default formatting
    // **YUCK** GCC 4.8.X doesn't seem to include std::defaultfloat
    os.unsetf(std::ios_base::floatfield);
    //os << std::defaultfloat;

    // Restore previous precision
    os << std::setprecision(previousPrecision);
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets.
//--------------------------------------------------------------------------
template<typename NameIter>
inline void value_substitutions(string &code, NameIter namesBegin, NameIter namesEnd, const vector<double> &values, const string &ext = "")
{
    NameIter n = namesBegin;
    auto v = values.cbegin();
    for (;n != namesEnd && v != values.cend(); n++, v++) {
        stringstream stream;
        writePreciseString(stream, *v);
        substitute(code,
                   "$(" + *n + ext + ")",
                   "(" + stream.str() + ")");
    }
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets.
//--------------------------------------------------------------------------
inline void value_substitutions(string &code, const vector<string> &names, const vector<double> &values, const string &ext = "")
{
    value_substitutions(code, names.cbegin(), names.cend(), values, ext);
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of function substitutions in code snipped
//--------------------------------------------------------------------------
void functionSubstitutions(std::string &code, const std::string &ftype,
                           const std::vector<FunctionTemplate> functions);

//--------------------------------------------------------------------------
/*! \brief This function implements a parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it).
 */
//--------------------------------------------------------------------------
string ensureFtype(const string &oldcode, const string &type);


//--------------------------------------------------------------------------
/*! \brief This function checks for unknown variable definitions and returns a gennError if any are found
 */
//--------------------------------------------------------------------------
void checkUnreplacedVariables(const string &code, const string &codeName);

//--------------------------------------------------------------------------
/*! \brief This function returns the 32-bit hash of a string - because these are used across MPI nodes which may have different libstdc++ it would be risky to use std::hash
 */
//--------------------------------------------------------------------------
uint32_t hashString(const std::string &string);

//-------------------------------------------------------------------------
/*!
  \brief Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.
*/
//-------------------------------------------------------------------------
void neuron_substitutions_in_synaptic_code(
    string &wCode,              //!< the code string to work on
    const SynapseGroup *sg,     //!< the synapse group connecting the pre and postsynaptic neuron populations whose parameters might need to be substituted
    const string &preIdx,       //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx,      //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix);   //!< device prefix, "dd_" for GPU, nothing for CPU