#include "codeGenUtils.h"

// Is C++ regex library operational?
// We assume it is for:
// 1) Non GCC compilers
// 2) GCC 5.X.X and future
// 3) Any future (4.10.X?) releases
// 4) 4.9.1 and subsequent patch releases (GCC fully implemented regex in 4.9.0
// BUT bug 61227 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61227 prevented \w from working)
#if !defined(__GNUC__) || \
    __GNUC__ > 4 || \
    (__GNUC__ == 4 && (__GNUC_MINOR__ > 9 || \
                      (__GNUC_MINOR__ == 9 && __GNUC_PATCHLEVEL__ >= 1)))
    #define REGEX_OPERATIONAL
#endif

// Standard includes
#ifdef REGEX_OPERATIONAL
#include <regex>
#endif

// GeNN includes
#include "modelSpec.h"
#include "standardSubstitutions.h"
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
void functionSubstitutions(std::string &code, const std::string &funcName,
                           unsigned int numParams, const std::string &replaceFuncTemplate)
{
    // Cache length of function name (including leading '$(')
    const size_t funcNameLength = funcName.size() + 2;

    // Reserve vector to hold parameters
    std::vector<std::string> parameters;
    parameters.reserve(numParams);

    // String to hold parameter currently being parsed
    std::string parameter = "";

    // Find first occurance of function with leading '$(' in code
    size_t found = code.find("$(" + funcName);

    // While functions are found
    while (found != std::string::npos) {
        // Loop through subsequent characerters of code
        unsigned int bracketDepth = 0;
        for(size_t i = found + funcNameLength; i < code.size(); i++) {
            // If this character is a comma at function bracket depth
            if(code[i] == ',' && bracketDepth == 0) {
                // If there was no parameter was read before comma, check that this is the first parameter
                if(parameter.empty()) {
                    assert(parameters.empty());
                }
                // Otherwise, add parameter to array
                else if(parameter.length() > 0) {
                    parameters.push_back(parameter);
                    parameter = "";
                }
            }
            // Otherwise
            else {
                // If this is an open bracket, increase bracket depth
                if(code[i] == '(') {
                    bracketDepth++;
                }
                // Otherwise, it's a close bracket
                else if(code[i] == ')') {
                    // If we are at a deeper bracket depth than function, decrease bracket depth
                    if(bracketDepth > 0) {
                        bracketDepth--;
                    }
                    // Otherwise
                    else {
                        // If there was no parameter was read before comma, check that this is the first parameter
                        if(parameter.empty()) {
                            assert(parameters.empty());
                        }
                        // Otherwise, add parameter to array
                        else if(parameter.length() > 0) {
                            parameters.push_back(parameter);
                            parameter = "";
                        }

                        // Check parameters match
                        assert(parameters.size() == numParams);

                        // Substitute parsed parameters into template
                        std::string replaceFunc = replaceFuncTemplate;
                        for(unsigned int p = 0; p < numParams; p++) {
                            substitute(replaceFunc, "$(" + std::to_string(p) + ")", parameters[p]);
                        }

                        // Clear parameters now they have been substituted
                        // into the final string to replace in to code
                        parameters.clear();

                        // Replace this into code
                        code.replace(found, i - found + 1, replaceFunc);
                        break;
                    }
                }

                // If this isn't a space at function bracket depth, add to parameter string
                if(bracketDepth > 0 || !std::isspace(code[i])) {
                    parameter += code[i];
                }
            }
        }

        // Find next function to replace
        found = code.find("$(" + funcName);
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


#ifdef REGEX_OPERATIONAL

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
    const SynapseGroup *sg,
     const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix //!< device prefix, "dd_" for GPU, nothing for CPU
    )
{

    // presynaptic neuron variables, parameters, and global parameters
    const auto *srcNeuronModel = sg->getSrcNeuronGroup()->getNeuronModel();
    if (srcNeuronModel->isPoisson()) {
        substitute(wCode, "$(V_pre)", to_string(sg->getSrcNeuronGroup()->getParams()[2]));
    }
    substitute(wCode, "$(sT_pre)", devPrefix+ "sT" + sg->getSrcNeuronGroup()->getName() + "[" + sg->getOffsetPre() + preIdx + "]");
    for(const auto &v : srcNeuronModel->getVars()) {
        if (sg->getSrcNeuronGroup()->isVarQueueRequired(v.first)) {
            substitute(wCode, "$(" + v.first + "_pre)",
                       devPrefix + v.first + sg->getSrcNeuronGroup()->getName() + "[" + sg->getOffsetPre() + preIdx + "]");
        }
        else {
            substitute(wCode, "$(" + v.first + "_pre)",
                       devPrefix + v.first + sg->getSrcNeuronGroup()->getName() + "[" + preIdx + "]");
        }
    }
    value_substitutions(wCode, srcNeuronModel->getParamNames(), sg->getSrcNeuronGroup()->getParams(), "_pre");

    DerivedParamNameIterCtx preDerivedParams(srcNeuronModel->getDerivedParams());
    value_substitutions(wCode, preDerivedParams.nameBegin, preDerivedParams.nameEnd, sg->getSrcNeuronGroup()->getDerivedParams(), "_pre");

    ExtraGlobalParamNameIterCtx preExtraGlobalParams(srcNeuronModel->getExtraGlobalParams());
    name_substitutions(wCode, "", preExtraGlobalParams.nameBegin, preExtraGlobalParams.nameEnd, sg->getSrcNeuronGroup()->getName(), "_pre");
    
    // postsynaptic neuron variables, parameters, and global parameters
    const auto *trgNeuronModel = sg->getTrgNeuronGroup()->getNeuronModel();
    substitute(wCode, "$(sT_post)", devPrefix + "sT" + sg->getTrgNeuronGroup()->getName() + "[" + sg->getOffsetPost(devPrefix) + postIdx + "]");
    for(const auto &v : trgNeuronModel->getVars()) {
        if (sg->getTrgNeuronGroup()->isVarQueueRequired(v.first)) {
            substitute(wCode, "$(" + v.first + "_post)",
                       devPrefix + v.first + sg->getTrgNeuronGroup()->getName() + "[" + sg->getOffsetPost(devPrefix) + postIdx + "]");
        }
        else {
            substitute(wCode, "$(" + v.first + "_post)",
                       devPrefix + v.first + sg->getTrgNeuronGroup()->getName() + "[" + postIdx + "]");
        }
    }
    value_substitutions(wCode, trgNeuronModel->getParamNames(), sg->getTrgNeuronGroup()->getParams(), "_post");

    DerivedParamNameIterCtx postDerivedParams(trgNeuronModel->getDerivedParams());
    value_substitutions(wCode, postDerivedParams.nameBegin, postDerivedParams.nameEnd, sg->getTrgNeuronGroup()->getDerivedParams(), "_post");

    ExtraGlobalParamNameIterCtx postExtraGlobalParams(trgNeuronModel->getExtraGlobalParams());
    name_substitutions(wCode, "", postExtraGlobalParams.nameBegin, postExtraGlobalParams.nameEnd, sg->getTrgNeuronGroup()->getName(), "_post");
}
