#pragma once

// Standard includes
#include <string>
#include <sstream>
#include <vector>

using namespace std;

// Forward declarations
class NNmodel;

namespace NeuronModels
{
    class Base;
}

//--------------------------------------------------------------------------
// PairKeyConstIter
//--------------------------------------------------------------------------
// Custom iterator for iterating through the keys of containers containing pairs
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

template<typename BaseIter>
inline PairKeyConstIter<BaseIter> GetPairKeyConstIter(BaseIter iter)
{
  return iter;
}

//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------

void substitute(string &s, const string &trg, const string &rep);


//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets.
//--------------------------------------------------------------------------
template<typename NameIter>
inline void name_substitutions(string &code, const string &prefix, NameIter namesBegin, NameIter namesEnd, const string &postfix= "")
{
    for (NameIter n = namesBegin; n != namesEnd; n++) {
        substitute(code, "$(" + *n + ")", prefix + *n + postfix);
    }
}

inline void name_substitutions(string &code, const string &prefix, const vector<string> &names, const string &postfix= "")
{
    name_substitutions(code, prefix, names.cbegin(), names.cend(), postfix);
}


//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets.
//--------------------------------------------------------------------------
template<typename NameIter>
inline void value_substitutions(string &code, NameIter namesBegin, NameIter namesEnd, const vector<double> &values)
{
    NameIter n = namesBegin;
    auto v = values.cbegin();
    for (;n != namesEnd && v != values.cend(); n++, v++) {
        substitute(code, "$(" + *n + ")", "(" + to_string(*v) + ")");
    }
}

inline void value_substitutions(string &code, const vector<string> &names, const vector<double> &values)
{
    value_substitutions(code, names.cbegin(), names.cend(), values);
}


//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets where the variables have an extension in their names (e.g. "_pre").
//--------------------------------------------------------------------------
template<typename NameIter>
inline void extended_name_substitutions(string &code, const string &prefix, NameIter namesBegin, NameIter namesEnd, const string &ext, const string &postfix= "")
{
    for (NameIter n = namesBegin; n != namesEnd; n++) {
        substitute(code, "$(" + *n + ext + ")", prefix + *n + postfix);
    }
}

inline void extended_name_substitutions(string &code, const string &prefix, const vector<string> &names, const string &ext, const string &postfix= "")
{
    extended_name_substitutions(code, prefix, names.cbegin(), names.cend(), ext, postfix);
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets where the parameters have an extension in their names (e.g. "_pre").
//--------------------------------------------------------------------------
template<typename NameIter>
inline void extended_value_substitutions(string &code, NameIter namesBegin, NameIter namesEnd,
                                         const string &ext, const vector<double> &values)
{
    NameIter n = namesBegin;
    auto v = values.cbegin();
    for (;n != namesEnd && v != values.cend(); n++, v++) {
        substitute(code, "$(" + *n + ext + ")", "(" + to_string(*v) + ")");
    }
}

inline void extended_value_substitutions(string &code, const vector<string> &names, const string &ext, const vector<double> &values)
{
    extended_value_substitutions(code, names.cbegin(), names.cend(), ext, values);
}
//--------------------------------------------------------------------------
/*! \brief This function converts code to contain only explicit single precision (float) function calls (C99 standard)
 */
//--------------------------------------------------------------------------

void ensureMathFunctionFtype(string &code, const string &type);


//--------------------------------------------------------------------------
/*! \brief This function is part of the parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it).
 */
//--------------------------------------------------------------------------

void doFinal(string &code, unsigned int i, const string &type, unsigned int &state);


//--------------------------------------------------------------------------
/*! \brief This function implements a parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it).
 */
//--------------------------------------------------------------------------

string ensureFtype(string oldcode, string type);


//--------------------------------------------------------------------------
/*! \brief This function checks for unknown variable definitions and returns a gennError if any are found
 */
//--------------------------------------------------------------------------

void checkUnreplacedVariables(string code, string codeName);


//-------------------------------------------------------------------------
/*!
  \brief Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.
*/
//-------------------------------------------------------------------------

void neuron_substitutions_in_synaptic_code(
    string &wCode, //!< the code string to work on
    const NNmodel &model, //!< the neuronal network model to generate code for
    unsigned int src, //!< the number of the src neuron population
    unsigned int trg, //!< the number of the target neuron population
    const NeuronModels::Base *preModel, //!< the model used by the pre-synaptic neuron
    const NeuronModels::Base *postModel, //!< the model used by the post-synaptic neuron
    const string &offsetPre, //!< delay slot offset expression for pre-synaptic vars
    const string &offsetPost, //!< delay slot offset expression for post-synaptic vars
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix //!< device prefix, "dd_" for GPU, nothing for CPU
                                           );
