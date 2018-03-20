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

// Standard C++ includes
#ifdef REGEX_OPERATIONAL
#include <regex>
#endif

// Standard C includes
#include <cstring>

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

enum MathsFunc
{
    MathsFuncDouble,
    MathsFuncSingle,
    MathsFuncMax,
};

const char *mathsFuncs[][MathsFuncMax] = {
    {"cos", "cosf"},
    {"sin", "sinf"},
    {"tan", "tanf"},
    {"acos", "acosf"},
    {"asin", "asinf"},
    {"atan", "atanf"},
    {"atan2", "atan2f"},
    {"cosh", "coshf"},
    {"sinh", "sinhf"},
    {"tanh", "tanhf"},
    {"acosh", "acoshf"},
    {"asinh", "asinhf"},
    {"atanh", "atanhf"},
    {"exp", "expf"},
    {"frexp", "frexpf"},
    {"ldexp", "ldexpf"},
    {"log", "logf"},
    {"log10", "log10f"},
    {"modf", "modff"},
    {"exp2", "exp2f"},
    {"expm1", "expm1f"},
    {"ilogb", "ilogbf"},
    {"log1p", "log1pf"},
    {"log2", "log2f"},
    {"logb", "logbf"},
    {"scalbn", "scalbnf"},
    {"scalbln", "scalblnf"},
    {"pow", "powf"},
    {"sqrt", "sqrtf"},
    {"cbrt", "cbrtf"},
    {"hypot", "hypotf"},
    {"erf", "erff"},
    {"erfc", "erfcf"},
    {"tgamma", "tgammaf"},
    {"lgamma", "lgammaf"},
    {"ceil", "ceilf"},
    {"floor", "floorf"},
    {"fmod", "fmodf"},
    {"trunc", "truncf"},
    {"round", "roundf"},
    {"lround", "lroundf"},
    {"llround", "llroundf"},
    {"rint", "rintf"},
    {"lrint", "lrintf"},
    {"nearbyint", "nearbyintf"},
    {"remainder", "remainderf"},
    {"remquo", "remquof"},
    {"copysign", "copysignf"},
    {"nan", "nanf"},
    {"nextafter", "nextafterf"},
    {"nexttoward", "nexttowardf"},
    {"fdim", "fdimf"},
    {"fmax", "fmaxf"},
    {"fmin", "fminf"},
    {"fabs", "fabsf"},
    {"fma", "fmaf"}
};

GenericFunction randomFuncs[] = {
    {"gennrand_uniform", 0},
    {"gennrand_normal", 0},
    {"gennrand_exponential", 0},
    {"gennrand_log_normal", 2}
};

//--------------------------------------------------------------------------
/*! \brief This function converts code to contain only explicit single precision (float) function calls (C99 standard)
 */
//--------------------------------------------------------------------------
void ensureMathFunctionFtype(string &code, const string &type)
{
    // If type is double, substitute any single precision maths functions for double precision version
    if (type == "double") {
        for(const auto &m : mathsFuncs) {
            substitute(code, string(m[MathsFuncSingle])+string("("), string(m[MathsFuncDouble])+string("("));
        }
    }
    // Otherwise, substitute any double precision maths functions for single precision version
    else {
        for(const auto &m : mathsFuncs) {
            substitute(code, string(m[MathsFuncDouble])+string("("), string(m[MathsFuncSingle])+string("("));
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

void neuronSubstitutionsInSynapticCode(
    string &wCode, //!< the code string to work on
    const NeuronGroup *ng,
    const string &offset,
    const string &idx,
    const string &varSuffix,
    const string &devPrefix) //!< device prefix, "dd_" for GPU, nothing for CPU
{
    // presynaptic neuron variables, parameters, and global parameters
    const auto *neuronModel = ng->getNeuronModel();
    substitute(wCode, "$(sT" + varSuffix + ")", devPrefix+ "sT" + ng->getName() + "[" + offset + idx + "]");
    for(const auto &v : neuronModel->getVars()) {
        if (ng->isVarQueueRequired(v.first)) {
            substitute(wCode, "$(" + v.first + varSuffix + ")",
                       devPrefix + v.first + ng->getName() + "[" + offset + idx + "]");
        }
        else {
            substitute(wCode, "$(" + v.first + varSuffix + ")",
                       devPrefix + v.first + ng->getName() + "[" + idx + "]");
        }
    }
    value_substitutions(wCode, neuronModel->getParamNames(), ng->getParams(), varSuffix);

    DerivedParamNameIterCtx preDerivedParams(neuronModel->getDerivedParams());
    value_substitutions(wCode, preDerivedParams.nameBegin, preDerivedParams.nameEnd, ng->getDerivedParams(), varSuffix);

    ExtraGlobalParamNameIterCtx preExtraGlobalParams(neuronModel->getExtraGlobalParams());
    name_substitutions(wCode, "", preExtraGlobalParams.nameBegin, preExtraGlobalParams.nameEnd, ng->getName(), varSuffix);
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
//! \brief Does the code string contain any functions requiring random number generator
//--------------------------------------------------------------------------
bool isRNGRequired(const std::string &code)
{
    // Loop through random functions
    for(const auto &r : randomFuncs) {
        // If this function takes no arguments, return true if
        // generic function name enclosed in $() markers is found
        if(r.numArguments == 0) {
            if(code.find("$(" + r.genericName + ")") != std::string::npos) {
                return true;
            }
        }
        // Otherwise, return true if generic function name
        // prefixed by $( and suffixed with comma is found
        else if(code.find("$(" + r.genericName + ",") != std::string::npos) {
            return true;
        }
    }
    return false;

}

//--------------------------------------------------------------------------
//! \brief Does the model with the vectors of variable initialisers and modes require an RNG for the specified init mode
//--------------------------------------------------------------------------
#ifndef CPU_ONLY
bool isInitRNGRequired(const std::vector<NewModels::VarInit> &varInitialisers, const std::vector<VarMode> &varModes,
                       VarInit initLocation)
{
    // Loop through variables
    for(unsigned int v = 0; v < varInitialisers.size(); v++) {
        const auto &varInit = varInitialisers[v];
        const auto varMode = varModes[v];

        // If initialisation snippet requires RNG and var should be initialised on this location, return true
        if(::isRNGRequired(varInit.getSnippet()->getCode()) && (varMode & initLocation)) {

            return true;
        }
    }

    return false;
}
#else
bool isInitRNGRequired(const std::vector<NewModels::VarInit> &varInitialisers, const std::vector<VarMode> &,
                       VarInit initLocation)
{
    // Loop through variables
    for(unsigned int v = 0; v < varInitialisers.size(); v++) {
        const auto &varInit = varInitialisers[v];

        // If initialisation snippet requires RNG and var init mode is set to host
        if(::isRNGRequired(varInit.getSnippet()->getCode()) && (initLocation == VarInit::HOST)) {
            return true;
        }
    }

    return false;
}
#endif
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
                        unsigned int numParams, const std::string &replaceFuncTemplate)
{
    // If there are no parameters, just replace the function name (wrapped in '$()')
    // with the template (which will, inherantly, not have any parameters)
    if(numParams == 0) {
        substitute(code, "$(" + funcName + ")", replaceFuncTemplate);
    }
    // Otherwise
    else {
        // Reserve vector to hold parameters
        std::vector<std::string> params;
        params.reserve(numParams);

        // String to hold parameter currently being parsed
        std::string currentParam = "";

        // Function will start with opening GeNN wrapper, name and comma before first argument
        // **NOTE** need to match up to comma so longer function names with same prefix aren't matched
        const std::string funcStart = "$(" + funcName + ",";

        // Find first occurance of start of function
        size_t found = code.find(funcStart);

        // While functions are found
        while (found != std::string::npos) {
            // Loop through subsequent characerters of code
            unsigned int bracketDepth = 0;
            for(size_t i = found + funcStart.length(); i < code.size(); i++) {
                // If this character is a comma at function bracket depth
                if(code[i] == ',' && bracketDepth == 0) {
                    assert(!currentParam.empty());

                    // Add parameter to array
                    params.push_back(currentParam);
                    currentParam = "";
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
                            assert(!currentParam.empty());

                            // Add parameter to array
                            params.push_back(currentParam);
                            currentParam = "";

                            // Check parameters match
                            assert(params.size() == numParams);

                            // Substitute parsed parameters into function template
                            std::string replaceFunc = replaceFuncTemplate;
                            for(unsigned int p = 0; p < numParams; p++) {
                                substitute(replaceFunc, "$(" + std::to_string(p) + ")", params[p]);
                            }

                            // Clear parameters now they have been substituted
                            // into the final string to replace in to code
                            params.clear();

                            // Replace this into code
                            code.replace(found, i - found + 1, replaceFunc);
                            break;
                        }
                    }

                    // If this isn't a space at function bracket depth,
                    // add to parameter string
                    if(bracketDepth > 0 || !::isspace(code[i])) {
                        currentParam += code[i];
                    }
                }
            }

            // Find start of next function to replace
            found = code.find(funcStart);
        }
    }
}

//--------------------------------------------------------------------------
//! \brief This function performs a list of function substitutions in code snipped
//--------------------------------------------------------------------------
void functionSubstitutions(std::string &code, const std::string &ftype,
                           const std::vector<FunctionTemplate> functions)
{
    // Substitute generic GeNN random functions for desired destination type
    for(const auto &f : functions) {
        const std::string &funcTemplate = (ftype == "double") ? f.doublePrecisionTemplate : f.singlePrecisionTemplate;
        functionSubstitute(code, f.genericName, f.numArguments, funcTemplate);
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

//--------------------------------------------------------------------------
/*! \brief This function returns the 32-bit hash of a string - because these are used across MPI nodes which may have different libstdc++ it would be risky to use std::hash
 */
//--------------------------------------------------------------------------
//! https://stackoverflow.com/questions/19411742/what-is-the-default-hash-function-used-in-c-stdunordered-map
//! suggests that libstdc++ uses MurmurHash2 so this seems as good a bet as any
//! MurmurHash2, by Austin Appleby
//! It has a few limitations -
//! 1. It will not work incrementally.
//! 2. It will not produce the same results on little-endian and big-endian
//!    machines.
uint32_t hashString(const std::string &string)
{
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.

    const uint32_t m = 0x5bd1e995;
    const unsigned int r = 24;

    // Get string length
    size_t len = string.length();

    // Initialize the hash to a 'random' value

    uint32_t h = 0xc70f6907 ^ len;

    // Mix 4 bytes at a time into the hash
    const char *data = string.c_str();
    while(len >= 4)
    {
        // **NOTE** one of the assumptions of the original MurmurHash2 was that
        // "We can read a 4-byte value from any address without crashing".
        // Bad experiance tells me this may not be the case on ARM so use memcpy
        uint32_t k;
        memcpy(&k, data, 4);

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        data += 4;
        len -= 4;
    }

    // Handle the last few bytes of the input array
    switch(len)
    {
        case 3: h ^= data[2] << 16;
        case 2: h ^= data[1] << 8;
        case 1: h ^= data[0];
                h *= m;
    };

    // Do a few final mixes of the hash to ensure the last few
    // bytes are well-incorporated.

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}
//-------------------------------------------------------------------------
/*!
  \brief Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.
*/
//-------------------------------------------------------------------------
void preNeuronSubstitutionsInSynapticCode(
    string &wCode, //!< the code string to work on
    const SynapseGroup *sg,
    const string &preIdx,
    const string &devPrefix) //!< device prefix, "dd_" for GPU, nothing for CPU
{
    // presynaptic neuron variables, parameters, and global parameters
    const auto *srcNeuronModel = sg->getSrcNeuronGroup()->getNeuronModel();
    if (srcNeuronModel->isPoisson()) {
        substitute(wCode, "$(V_pre)", to_string(sg->getSrcNeuronGroup()->getParams()[2]));
    }
    neuronSubstitutionsInSynapticCode(wCode, sg->getSrcNeuronGroup(), sg->getOffsetPre(), preIdx, "_pre", devPrefix);
}

void postNeuronSubstitutionsInSynapticCode(
    string &wCode, //!< the code string to work on
    const SynapseGroup *sg,
    const string &postIdx,
    const string &devPrefix) //!< device prefix, "dd_" for GPU, nothing for CPU
{
    // postsynaptic neuron variables, parameters, and global parameters
    neuronSubstitutionsInSynapticCode(wCode, sg->getTrgNeuronGroup(), sg->getOffsetPost(devPrefix), postIdx, "_post", devPrefix);
}

void neuron_substitutions_in_synaptic_code(
    string &wCode, //!< the code string to work on
    const SynapseGroup *sg,
    const string &preIdx, //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const string &postIdx, //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    const string &devPrefix) //!< device prefix, "dd_" for GPU, nothing for CPU
{
    preNeuronSubstitutionsInSynapticCode(wCode, sg, preIdx, devPrefix);
    postNeuronSubstitutionsInSynapticCode(wCode, sg, postIdx, devPrefix);

}
