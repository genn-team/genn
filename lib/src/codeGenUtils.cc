#include "codeGenUtils.h"

// Standard includes
#if !defined(__GNUC__) || (__GNUC__ >= 4 && __GNUC_MINOR__ >= 9)
#include <regex>
#endif

// GeNN includes
#include "modelSpec.h"
#include "utils.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
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
}    // Anonymous namespace

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
/*! \brief This function implements a parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it). 
 */
//--------------------------------------------------------------------------

string ensureFtype(const string &oldcode, const string &type)
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

void checkUnreplacedVariables(const string &code, const string &codeName)
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
void checkUnreplacedVariables(const string &, const string &)
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
    const string &offsetPre, //!< delay slot offset expression for pre-synaptic vars
    const string &offsetPost, //!< delay slot offset expression for post-synaptic vars
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix //!< device prefix, "dd_" for GPU, nothing for CPU
    )
{
    // presynaptic neuron variables, parameters, and global parameters
    if (model.neuronModel[src]->IsPoisson()) {
        substitute(wCode, "$(V_pre)", to_string(model.neuronPara[src][2]));
    }
    substitute(wCode, "$(sT_pre)", devPrefix+ "sT" + model.neuronName[src] + "[" + offsetPre + preIdx + "]");
    auto preVars = model.neuronModel[src]->GetVars();
    for (size_t j = 0; j < preVars.size(); j++) {
        if (model.neuronVarNeedQueue[src][j]) {
            substitute(wCode, "$(" + preVars[j].first + "_pre)",
                       devPrefix + preVars[j].first + model.neuronName[src] + "[" + offsetPre + preIdx + "]");
        }
        else {
            substitute(wCode, "$(" + preVars[j].first + "_pre)",
                       devPrefix + preVars[j].first + model.neuronName[src] + "[" + preIdx + "]");
        }
    }
    extended_value_substitutions(wCode, model.neuronModel[src]->GetParamNames(), "_pre", model.neuronPara[src]);

    auto preDerivedParams = model.neuronModel[src]->GetDerivedParams();
    auto preDerivedParamNameBegin = GetPairKeyConstIter(preDerivedParams.cbegin());
    auto preDerivedParamNameEnd = GetPairKeyConstIter(preDerivedParams.cend());
    extended_value_substitutions(wCode, preDerivedParamNameBegin, preDerivedParamNameEnd, "_pre", model.dnp[src]);

    auto preExtraGlobalParams = model.neuronModel[src]->GetExtraGlobalParams();
    auto preExtraGlobalParamsNameBegin = GetPairKeyConstIter(preExtraGlobalParams.cbegin());
    auto preExtraGlobalParamsNameEnd = GetPairKeyConstIter(preExtraGlobalParams.cend());
    extended_name_substitutions(wCode, devPrefix, preExtraGlobalParamsNameBegin, preExtraGlobalParamsNameEnd, "_pre", model.neuronName[src]);
    
    // postsynaptic neuron variables, parameters, and global parameters
    substitute(wCode, "$(sT_post)", devPrefix + "sT" + model.neuronName[trg] + "[" + offsetPost + postIdx + "]");
    auto postVars = model.neuronModel[trg]->GetVars();
    for (size_t j = 0; j < postVars.size(); j++) {
        if (model.neuronVarNeedQueue[trg][j]) {
            substitute(wCode, "$(" + postVars[j].first + "_post)",
                       devPrefix + postVars[j].first + model.neuronName[trg] + "[" + offsetPost + postIdx + "]");
        }
        else {
            substitute(wCode, "$(" + postVars[j].first + "_post)",
                       devPrefix + postVars[j].first + model.neuronName[trg] + "[" + postIdx + "]");
        }
    }
    extended_value_substitutions(wCode, model.neuronModel[trg]->GetParamNames(), "_post", model.neuronPara[trg]);

    auto postDerivedParams = model.neuronModel[trg]->GetDerivedParams();
    auto postDerivedParamNameBegin= GetPairKeyConstIter(postDerivedParams.cbegin());
    auto postDerivedParamNameEnd = GetPairKeyConstIter(postDerivedParams.cend());
    extended_value_substitutions(wCode, postDerivedParamNameBegin, postDerivedParamNameEnd, "_post", model.dnp[trg]);

    auto postExtraGlobalParams = model.neuronModel[trg]->GetExtraGlobalParams();
    auto postExtraGlobalParamsNameBegin = GetPairKeyConstIter(postExtraGlobalParams.cbegin());
    auto postExtraGlobalParamsNameEnd = GetPairKeyConstIter(postExtraGlobalParams.cend());
    extended_name_substitutions(wCode, devPrefix, postExtraGlobalParamsNameBegin, postExtraGlobalParamsNameEnd, "_post", model.neuronName[trg]);
}