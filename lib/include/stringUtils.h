
#ifndef STRINGUTILS_H
#define STRINGUTILS_H

#include <string>
#include <vector>

using namespace std;


//--------------------------------------------------------------------------
/*! \brief template function for conversion of various types to C++ strings
 */
//--------------------------------------------------------------------------

template<class T> std::string toString(T t);
template<> std::string toString(float t);
template<> std::string toString(double t);

#define tS(X) toString(X) //!< Macro providing the abbreviated syntax tS() instead of toString().


//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------

void substitute(string &s, const string trg, const string rep);


//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets.
//--------------------------------------------------------------------------

void name_substitutions(string &code, string prefix, vector<string> &names, string postfix= "");


//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets.
//--------------------------------------------------------------------------

void value_substitutions(string &code, vector<string> &names, vector<double> &values);


//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets where the variables have an extension in their names (e.g. "_pre").
//--------------------------------------------------------------------------

void extended_name_substitutions(string &code, string prefix, vector<string> &names, string ext, string postfix= "");


//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets where the parameters have an extension in their names (e.g. "_pre").
//--------------------------------------------------------------------------

void extended_value_substitutions(string &code, vector<string> &names, string ext, vector<double> &values);


//--------------------------------------------------------------------------
/*! \brief This function converts code to contain only explicit single precision (float) function calls (C99 standard)
 */
//--------------------------------------------------------------------------

void ensureMathFunctionFtype(string &code, string type);


//--------------------------------------------------------------------------
/*! \brief This function is part of the parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it). 
 */
//--------------------------------------------------------------------------

void doFinal(string &code, unsigned int i, string type, unsigned int &state);


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

#endif // STRINGUTILS_H
