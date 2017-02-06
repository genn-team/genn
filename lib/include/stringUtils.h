
#ifndef STRINGUTILS_H
#define STRINGUTILS_H

#include "modelSpec.h"

#include <string>
#include <sstream>
#include <vector>

using namespace std;


//--------------------------------------------------------------------------
/*! \brief template functions for conversion of various types to C++ strings
 */
//--------------------------------------------------------------------------

template<class T> std::string toString(T t)
{
    std::stringstream s;
    s << std::showpoint << t;
    return s.str();
}

#define tS(X) toString(X) //!< Macro providing the abbreviated syntax tS() instead of toString().


//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------

void substitute(string &s, const string &trg, const string &rep);


//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets.
//--------------------------------------------------------------------------

void name_substitutions(string &code, const string &prefix, const vector<string> &names, const string &postfix= "");


//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets.
//--------------------------------------------------------------------------

void value_substitutions(string &code, const vector<string> &names, const vector<double> &values);


//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets where the variables have an extension in their names (e.g. "_pre").
//--------------------------------------------------------------------------

void extended_name_substitutions(string &code, const string &prefix, const vector<string> &names, const string &ext, const string &postfix= "");


//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets where the parameters have an extension in their names (e.g. "_pre").
//--------------------------------------------------------------------------

void extended_value_substitutions(string &code, const vector<string> &names, const string &ext, const vector<double> &values);


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
    unsigned int nt_pre, //!< the neuron type of the pre-synaptic neuron
    unsigned int nt_post, //!< the neuron type of the post-synaptic neuron
    const string &offsetPre, //!< delay slot offset expression for pre-synaptic vars
    const string &offsetPost, //!< delay slot offset expression for post-synaptic vars
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix //!< device prefix, "dd_" for GPU, nothing for CPU
                                           );

#endif // STRINGUTILS_H
