
#ifndef STRINGUTILS_CC
#define STRINGUTILS_CC

#include "stringUtils.h"
#include "utils.h"

#if !defined(__GNUC__) || (__GNUC__ >= 4 && __GNUC_MINOR__ >= 9)
#include <regex>
#endif


//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------

void substitute(string &s, const string &trg, const string &rep)
{
    size_t found= s.find(trg);
    while (found != string::npos) {
        s.replace(found,trg.length(),rep);
        found= s.find(trg);
    }
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets.
//--------------------------------------------------------------------------

void name_substitutions(string &code, const string &prefix, const vector<string> &names, const string &postfix)
{
    for (int k = 0, l = names.size(); k < l; k++) {
        substitute(code, tS("$(") + names[k] + tS(")"), prefix+names[k]+postfix);
    }
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets.
//--------------------------------------------------------------------------

void value_substitutions(string &code, const vector<string> &names, const vector<double> &values)
{
    for (int k = 0, l = names.size(); k < l; k++) {
        substitute(code, tS("$(") + names[k] + tS(")"), tS("(")+tS(values[k])+ tS(")"));
    }
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of name substitutions for variables in code snippets where the variables have an extension in their names (e.g. "_pre").
//--------------------------------------------------------------------------

void extended_name_substitutions(string &code, const string &prefix, const vector<string> &names, const string &ext, const string &postfix)
{
    for (int k = 0, l = names.size(); k < l; k++) {
        substitute(code, tS("$(") + names[k] + ext + tS(")"), prefix+names[k]+postfix);
    }
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of value substitutions for parameters in code snippets where the parameters have an extension in their names (e.g. "_pre").
//--------------------------------------------------------------------------

void extended_value_substitutions(string &code, const vector<string> &names, const string &ext, const vector<double> &values)
{
    for (int k = 0, l = names.size(); k < l; k++) {
        substitute(code, tS("$(") + names[k] + ext + tS(")"), tS("(")+tS(values[k])+ tS(")"));
    }
}

const string digits= string("0123456789");
const string op= string("+-*/(<>= ,;")+string("\n")+string("\t");

const int __mathFN = 56;
const char *__dnames[__mathFN]= {
    "cos",
    "sin",
    "tan",
    "acos",
    "asin",
    "atan",
    "atan2",
    "cosh",
    "sinh",
    "tanh",
    "acosh",
    "asinh",
    "atanh",
    "exp",
    "frexp",
    "ldexp",
    "log",
    "log10",
    "modf",
    "exp2",
    "expm1",
    "ilogb",
    "log1p",
    "log2",
    "logb",
    "scalbn",
    "scalbln",
    "pow",
    "sqrt",
    "cbrt",
    "hypot",
    "erf",
    "erfc",
    "tgamma",
    "lgamma",
    "ceil",
    "floor",
    "fmod",
    "trunc",
    "round",
    "lround",
    "llround",
    "rint",
    "lrint",
    "nearbyint",
    "remainder",
    "remquo",
    "copysign",
    "nan",
    "nextafter",
    "nexttoward",
    "fdim",
    "fmax",
    "fmin",
    "fabs",
    "fma"
};

const char *__fnames[__mathFN]= {
    "cosf",
    "sinf",
    "tanf",
    "acosf",
    "asinf",
    "atanf",
    "atan2f",
    "coshf",
    "sinhf",
    "tanhf",
    "acoshf",
    "asinhf",
    "atanhf",
    "expf",
    "frexpf",
    "ldexpf",
    "logf",
    "log10f",
    "modff",
    "exp2f",
    "expm1f",
    "ilogbf",
    "log1pf",
    "log2f",
    "logbf",
    "scalbnf",
    "scalblnf",
    "powf",
    "sqrtf",
    "cbrtf",
    "hypotf",
    "erff",
    "erfcf",
    "tgammaf",
    "lgammaf",
    "ceilf",
    "floorf",
    "fmodf",
    "truncf",
    "roundf",
    "lroundf",
    "llroundf",
    "rintf",
    "lrintf",
    "nearbyintf",
    "remainderf",
    "remquof",
    "copysignf",
    "nanf",
    "nextafterf",
    "nexttowardf",
    "fdimf",
    "fmaxf",
    "fminf",
    "fabsf",
    "fmaf"
};


//--------------------------------------------------------------------------
/*! \brief This function converts code to contain only explicit single precision (float) function calls (C99 standard)
 */
//--------------------------------------------------------------------------

void ensureMathFunctionFtype(string &code, const string &type)
{
    if (type == string("double")) {
        for (int i= 0; i < __mathFN; i++) {
            substitute(code, string(__fnames[i])+string("("), string(__dnames[i])+string("("));
        }
    }
    else {
        for (int i= 0; i < __mathFN; i++) {
            substitute(code, string(__dnames[i])+string("("), string(__fnames[i])+string("("));
        }
    }
}


//--------------------------------------------------------------------------
/*! \brief This function is part of the parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it). 
 */
//--------------------------------------------------------------------------

void doFinal(string &code, unsigned int i, const string &type, unsigned int &state)
{
    if (code[i] == 'f') {
        if (type == "double") {
            code.erase(i,1);
        }
    }
    else {
        if (type == "float") {
            code.insert(i,1,'f');
        }
    }
    if (i < code.size()-1) {
        if (op.find(code[i]) == string::npos) {
            state= 0;
        }
        else {
            state= 1;
        }
    }
}


//--------------------------------------------------------------------------
/*! \brief This function implements a parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it). 
 */
//--------------------------------------------------------------------------

string ensureFtype(string oldcode, string type) 
{
//    cerr << "entering ensure" << endl;
//    cerr << oldcode << endl;
    string code= oldcode;
    unsigned int i= 0;
    unsigned int state= 1; // allowed to start with a number straight away.
    while (i < code.size()) {
        switch (state)
        {
        case 0: // looking for a valid lead-in
            if (op.find(code[i]) != string::npos) {
                state= 1;
                break;
            }
            break;
        case 1: // looking for start of number
            if (digits.find(code[i]) != string::npos) {
                state= 2; // found the beginning of a number starting with a digit
                break;
            }
            if (code[i] == '.') {
                state= 3; // number starting with a dot
                break;
            }
            if (op.find(code[i]) == string::npos) {
                state= 0;
                break;
            }
            break;
        case 2: // in a number, looking for more digits, '.', 'e', 'E', or end of number
            if (code[i] == '.') {
                state= 3; // number now also contained a dot
                break;
            }
            if ((code[i] == 'e') || (code[i] == 'E')) {
                state= 4;
                break;
            }
            if (digits.find(code[i]) == string::npos) {// the number looks like an integer ...
                if (op.find(code[i]) != string::npos) state= 1;
                else state= 0;
                break;
            }
            break;
        case 3: // we have had '.' now looking for digits or 'e', 'E'
            if ((code[i] == 'e') || (code[i] == 'E')) {
                state= 4;
                break;
            }
            if (digits.find(code[i]) == string::npos) {
                doFinal(code, i, type, state);
                break;
            }
            break;
        case 4: // we have had '.' and 'e', 'E', digits only now
            if (digits.find(code[i]) != string::npos) {
                state= 6;
                break;
            }
            if ((code[i] != '+') && (code[i] != '-')) {
                if (op.find(code[i]) != string::npos) state= 1;
                else state= 0;
                break;
            }
            else {
                state= 5;
                break;
            }
        case 5: // now one or more digits or else ...
            if (digits.find(code[i]) != string::npos) {
                state= 6;
                break;
            }
            else {
                if (op.find(code[i]) != string::npos) state= 1;
                else state= 0;
                break;
            }
        case 6: // any non-digit character will trigger action
            if (digits.find(code[i]) == string::npos) {
                doFinal(code, i, type, state);
                break;
            }
            break;
        }
        i++;
    }
    if ((state == 3) || (state == 6)) {
        if (type == "float") {
            code= code+string("f");
        }
    }
    ensureMathFunctionFtype(code, type);
    return code;
}


#if !defined(__GNUC__) || (__GNUC__ >= 4 && __GNUC_MINOR__ >= 9)

//--------------------------------------------------------------------------
/*! \brief This function checks for unknown variable definitions and returns a gennError if any are found
 */
//--------------------------------------------------------------------------

void checkUnreplacedVariables(string code, string codeName) 
{
    regex rgx("\\$\\([\\w]+\\)");
    string vars= "";
    for (sregex_iterator it(code.begin(), code.end(), rgx), end; it != end; it++) {
        vars+= it->str().substr(2,it->str().size()-3) + ", ";
    }
    if (vars.size() > 0) {
        vars= vars.substr(0, vars.size()-2);
        if (vars.find(",") != string::npos) vars= "variables "+vars+" were ";
        else vars= "variable "+vars+" was ";
        gennError("The "+vars+"undefined in code "+codeName+".");
    }
}
#else
void checkUnreplacedVariables(string code, string codeName) 
{
}
#endif


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
    )
{
    // presynaptic neuron variables, parameters, and global parameters
    if (model.neuronType[src] == POISSONNEURON) substitute(wCode, tS("$(V_pre)"), tS(model.neuronPara[src][2]));
    substitute(wCode, tS("$(sT_pre)"), devPrefix+ tS("sT") + model.neuronName[src] + tS("[") + offsetPre + preIdx + tS("]"));
    for (int j = 0; j < nModels[nt_pre].varNames.size(); j++) {
        if (model.neuronVarNeedQueue[src][j]) {
            substitute(wCode, tS("$(") + nModels[nt_pre].varNames[j] + tS("_pre)"),
                       devPrefix + nModels[nt_pre].varNames[j] + model.neuronName[src] + tS("[") + offsetPre + preIdx + tS("]"));
        }
        else {
            substitute(wCode, tS("$(") + nModels[nt_pre].varNames[j] + tS("_pre)"),
                       devPrefix + nModels[nt_pre].varNames[j] + model.neuronName[src] + tS("[") + preIdx + tS("]"));
        }
    }
    extended_value_substitutions(wCode, nModels[nt_pre].pNames, tS("_pre"), model.neuronPara[src]);
    extended_value_substitutions(wCode, nModels[nt_pre].dpNames, tS("_pre"), model.dnp[src]);
    extended_name_substitutions(wCode, devPrefix, nModels[nt_pre].extraGlobalNeuronKernelParameters, tS("_pre"), model.neuronName[src]);
    
    // postsynaptic neuron variables, parameters, and global parameters
    substitute(wCode, tS("$(sT_post)"), devPrefix+tS("sT") + model.neuronName[trg] + tS("[") + offsetPost + postIdx + tS("]"));
    for (int j = 0; j < nModels[nt_post].varNames.size(); j++) {
        if (model.neuronVarNeedQueue[trg][j]) {
            substitute(wCode, tS("$(") + nModels[nt_post].varNames[j] + tS("_post)"),
                       devPrefix + nModels[nt_post].varNames[j] + model.neuronName[trg] + tS("[") + offsetPost + postIdx + tS("]"));
        }
        else {
            substitute(wCode, tS("$(") + nModels[nt_post].varNames[j] + tS("_post)"),
                       devPrefix + nModels[nt_post].varNames[j] + model.neuronName[trg] + tS("[") + postIdx + tS("]"));
        }
    }
    extended_value_substitutions(wCode, nModels[nt_post].pNames, tS("_post"), model.neuronPara[trg]);
    extended_value_substitutions(wCode, nModels[nt_post].dpNames, tS("_post"), model.dnp[trg]);
    extended_name_substitutions(wCode, devPrefix, nModels[nt_post].extraGlobalNeuronKernelParameters, tS("_post"), model.neuronName[trg]);
}

#endif // STRINGUTILS_CC
