#include "code_generator/codeGenUtils.h"

// Is C++ regex library operational?
// We assume it is for:
// 1) Compilers that don't define __GNUCC__
// 2) Clang
// 3) GCC 5.X.Y and future
// 4) Any future (4.10.Y?) GCC 4.X.Y releases
// 5) GCC 4.9.1 and subsequent patch releases (GCC fully implemented regex in 4.9.0
// BUT bug 61227 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61227 prevented \w from working until 4.9.1)
#if !defined(__GNUC__) || \
    __clang__ || \
    __GNUC__ > 4 || \
    (__GNUC__ == 4 && (__GNUC_MINOR__ > 9 || \
                      (__GNUC_MINOR__ == 9 && __GNUC_PATCHLEVEL__ >= 1)))
    #include <regex>
#else
    #error "GeNN now requires a functioning std::regex implementation - please upgrade your version of GCC to at least 4.9.1"
#endif

// Standard C includes
#include <cstring>

// GeNN includes
#include "modelSpec.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"
#include "code_generator/substitutions.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const std::string digits="0123456789";
const std::string op= std::string("+-*/(<>= ,;")+std::string("\n")+std::string("\t");

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

//--------------------------------------------------------------------------
/*! \brief This function converts code to contain only explicit single precision (float) function calls (C99 standard)
 */
//--------------------------------------------------------------------------
void ensureMathFunctionFtype(std::string &code, const std::string &type)
{
    using namespace CodeGenerator;

    // If type is double, substitute any single precision maths functions for double precision version
    if (type == "double") {
        for(const auto &m : mathsFuncs) {
            regexFuncSubstitute(code, m[MathsFuncSingle], m[MathsFuncDouble]);
        }
    }
    // Otherwise, substitute any double precision maths functions for single precision version
    else {
        for(const auto &m : mathsFuncs) {
            regexFuncSubstitute(code, m[MathsFuncDouble], m[MathsFuncSingle]);
        }
    }
}

//--------------------------------------------------------------------------
/*! \brief This function is part of the parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it).
 */
//--------------------------------------------------------------------------
void doFinal(std::string &code, unsigned int i, const std::string &type, unsigned int &state)
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
        if (op.find(code[i]) == std::string::npos) {
            state= 0;
        }
        else {
            state= 1;
        }
    }
}

bool regexSubstitute(std::string &s, const std::regex &regex, const std::string &format)
{
    // **NOTE** the following code performs the same function as std::regex_replace
    // but has a return value indicating whether any replacements are made
    // see http://en.cppreference.com/w/cpp/regex/regex_replace

    // Create regex iterator to iterate over matches found in code
    std::sregex_iterator matchesBegin(s.cbegin(), s.cend(), regex);
    std::sregex_iterator matchesEnd;

    // If there are no matches, leave s unmodified and return false
    if(matchesBegin == matchesEnd) {
        return false;
    }
    // Otherwise
    else {
        // Loop through matches
        std::string output;
        for(std::sregex_iterator m = matchesBegin;;) {
            // Copy the non-matched subsequence (m->prefix()) onto output
            std::copy(m->prefix().first, m->prefix().second, std::back_inserter(output));

            // Then replaces the matched subsequence with the formatted replacement string
            m->format(std::back_inserter(output), format);

            // If there are no subsequent matches
            if(std::next(m) == matchesEnd) {
                // Copy the remaining non-matched characters onto output
                std::copy(m->suffix().first, m->suffix().second, std::back_inserter(output));
                break;
            }
            // Otherwise go onto next match
            else {
                m++;
            }
        }

        // Set reference to newly processed version and return true
        s = output;
        return true;
    }
}
}    // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------
void substitute(std::string &s, const std::string &trg, const std::string &rep)
{
    size_t found= s.find(trg);
    while (found != std::string::npos) {
        s.replace(found,trg.length(),rep);
        found= s.find(trg);
    }
}

//--------------------------------------------------------------------------
//! \brief Tool for substituting variable  names in the neuron code strings or other templates using regular expressions
//--------------------------------------------------------------------------
bool regexVarSubstitute(std::string &s, const std::string &trg, const std::string &rep)
{
    // Build a regex to match variable name with at least one
    // character that can't be in a variable name on either side (or an end/beginning of string)
    // **NOTE** the suffix is non-capturing so two instances of variables separated by a single character are matched e.g. a*a
    std::regex regex("(^|[^0-9a-zA-Z_])" + trg + "(?=$|[^a-zA-Z0-9_])");

    // Create format string to replace in text
    // **NOTE** preceding character is captured as C++ regex doesn't support lookbehind so this needs to be replaced in
    const std::string format = "$1" + rep;

    return regexSubstitute(s, regex, format);
}

//--------------------------------------------------------------------------
//! \brief Tool for substituting function  names in the neuron code strings or other templates using regular expressions
//--------------------------------------------------------------------------
bool regexFuncSubstitute(std::string &s, const std::string &trg, const std::string &rep)
{
    // Build a regex to match function name with at least one
    // character that can't be part of the function name on the left and a bracket on the right (with optional whitespace)
    // **NOTE** the suffix is non-capturing so two instances of functions separated by a single character are matched e.g. sin(cos(x));
    std::regex regex("(^|[^0-9a-zA-Z_])" + trg + "(?=\\s*\\()");

    // Create format string to replace in text
    // **NOTE** preceding character is captured as C++ regex doesn't support lookbehind so this needs to be replaced in
    const std::string format = "$1" + rep;

    return regexSubstitute(s, regex, format);
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

void genMergedGroupSpikeCountReset(CodeStream &os, const NeuronGroupMerged &n)
{
    if(n.getArchetype().isDelayRequired()) { // with delay
        if(n.getArchetype().isSpikeEventRequired()) {
            os << "group.spkCntEvnt[*group.spkQuePtr] = 0;" << std::endl;
        }
        if(n.getArchetype().isTrueSpikeRequired()) {
            os << "group.spkCnt[*group.spkQuePtr] = 0;" << std::endl;
        }
        else {
            os << "group.spkCnt[0] = 0;" << std::endl;
        }
    }
    else { // no delay
        if(n.getArchetype().isSpikeEventRequired()) {
            os << "group.spkCntEvnt[0] = 0;" << std::endl;
        }
        os << "group.spkCnt[0] = 0;" << std::endl;
    }
}

void genScalarEGPPush(CodeStream &os, const MergedEGPMap &mergedEGPs, const std::string &suffix, const BackendBase &backend)
{
    // Loop through all merged EGPs
    for(const auto &e : mergedEGPs) {
        // Loop through range with correct suffix
        const auto groupEGPs = e.second.equal_range(suffix);
        for (auto g = groupEGPs.first; g != groupEGPs.second; ++g) {
            // If EGP is scalar, generate code to copy
            if(!Utils::isTypePointer(g->second.type)) {
                backend.genMergedExtraGlobalParamPush(os, suffix, g->second.mergedGroupIndex, 
                                                      std::to_string(g->second.groupIndex), 
                                                      g->second.fieldName, e.first);
            }

        }
    }
}

//--------------------------------------------------------------------------
/*! \brief This function implements a parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it). 
 */
//--------------------------------------------------------------------------

std::string ensureFtype(const std::string &oldcode, const std::string &type)
{
//    cerr << "entering ensure" << endl;
//    cerr << oldcode << endl;
    std::string code= oldcode;
    unsigned int i= 0;
    unsigned int state= 1; // allowed to start with a number straight away.
    while (i < code.size()) {
        switch (state)
        {
        case 0: // looking for a valid lead-in
            if (op.find(code[i]) != std::string::npos) {
                state= 1;
                break;
            }
            break;
        case 1: // looking for start of number
            if (digits.find(code[i]) != std::string::npos) {
                state= 2; // found the beginning of a number starting with a digit
                break;
            }
            if (code[i] == '.') {
                state= 3; // number starting with a dot
                break;
            }
            if (op.find(code[i]) == std::string::npos) {
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
            if (digits.find(code[i]) == std::string::npos) {// the number looks like an integer ...
                if (op.find(code[i]) != std::string::npos) state= 1;
                else state= 0;
                break;
            }
            break;
        case 3: // we have had '.' now looking for digits or 'e', 'E'
            if ((code[i] == 'e') || (code[i] == 'E')) {
                state= 4;
                break;
            }
            if (digits.find(code[i]) == std::string::npos) {
                doFinal(code, i, type, state);
                break;
            }
            break;
        case 4: // we have had '.' and 'e', 'E', digits only now
            if (digits.find(code[i]) != std::string::npos) {
                state= 6;
                break;
            }
            if ((code[i] != '+') && (code[i] != '-')) {
                if (op.find(code[i]) != std::string::npos) state= 1;
                else state= 0;
                break;
            }
            else {
                state= 5;
                break;
            }
        case 5: // now one or more digits or else ...
            if (digits.find(code[i]) != std::string::npos) {
                state= 6;
                break;
            }
            else {
                if (op.find(code[i]) != std::string::npos) state= 1;
                else state= 0;
                break;
            }
        case 6: // any non-digit character will trigger action
            if (digits.find(code[i]) == std::string::npos) {
                doFinal(code, i, type, state);
                break;
            }
            break;
        }
        i++;
    }
    if ((state == 3) || (state == 6)) {
        if (type == "float") {
            code= code+"f";
        }
    }
    ensureMathFunctionFtype(code, type);
    return code;
}

//--------------------------------------------------------------------------
/*! \brief This function checks for unknown variable definitions and returns a gennError if any are found
 */
//--------------------------------------------------------------------------

void checkUnreplacedVariables(const std::string &code, const std::string &codeName)
{
    std::regex rgx("\\$\\([\\w]+\\)");
    std::string vars= "";
    for (std::sregex_iterator it(code.begin(), code.end(), rgx), end; it != end; it++) {
        vars+= it->str().substr(2,it->str().size()-3) + ", ";
    }
    if (vars.size() > 0) {
        vars= vars.substr(0, vars.size()-2);

        vars = (vars.find(",") != std::string::npos) ? "variables " + vars + " were " : "variable " + vars + " was ";
       
        throw std::runtime_error("The "+vars+"undefined in code "+codeName+".");
    }
}

//-------------------------------------------------------------------------
/*!
  \brief Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.
*/
//-------------------------------------------------------------------------

void neuronSubstitutionsInSynapticCode(
    CodeGenerator::Substitutions &substitutions,
    const NeuronGroupInternal *ng,
    const std::string &offset,
    const std::string &delayOffset,
    const std::string &idx,
    const std::string &sourceSuffix,
    const std::string &destSuffix,
    const std::string &varPrefix,
    const std::string &varSuffix)
{
    using namespace CodeGenerator;

    // presynaptic neuron variables, parameters, and global parameters
    const auto *neuronModel = ng->getNeuronModel();
    substitutions.addVarSubstitution("sT" + sourceSuffix,
                                     "(" + delayOffset + varPrefix + "group.sT" + destSuffix + "[" + offset + idx + "]" + varSuffix + ")");
    for(const auto &v : neuronModel->getVars()) {
        const std::string varIdx = ng->isVarQueueRequired(v.name) ? offset + idx : idx;

        substitutions.addVarSubstitution(v.name + sourceSuffix,
                                         varPrefix + "group." + v.name + destSuffix + "[" + varIdx + "]" + varSuffix);
    }
    substitutions.addParamValueSubstitution(neuronModel->getParamNames(), ng->getParams(), sourceSuffix);
    substitutions.addVarValueSubstitution(neuronModel->getDerivedParams(), ng->getDerivedParams(), sourceSuffix);
    substitutions.addVarNameSubstitution(neuronModel->getExtraGlobalParams(), sourceSuffix, "group.", destSuffix);
}

void neuronSubstitutionsInSynapticCode(
    Substitutions &substitutions,        //!< the code string to work on
    const SynapseGroupInternal &sg,      //!< the synapse group connecting the pre and postsynaptic neuron populations whose parameters might need to be substituted
    const std::string &preIdx,           //!< index of the pre-synaptic neuron to be accessed for _pre variables; differs for different Span)
    const std::string &postIdx,          //!< index of the post-synaptic neuron to be accessed for _post variables; differs for different Span)
    double dt,                           //!< simulation timestep (ms)
    const std::string &preVarPrefix,     //!< prefix to be used for presynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &preVarSuffix,     //!< suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarPrefix,    //!< prefix to be used for postsynaptic variable accesses - typically combined with suffix to wrap in function call such as __ldg(&XXX)
    const std::string &postVarSuffix)    //!< suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as __ldg(&XXX)
{
    const std::string axonalDelayOffset = Utils::writePreciseString(dt * (double)(sg.getDelaySteps() + 1u)) + " + ";
    const std::string preOffset = sg.getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
    neuronSubstitutionsInSynapticCode(substitutions, sg.getSrcNeuronGroup(), preOffset, axonalDelayOffset, preIdx, "_pre", "Pre", preVarPrefix, preVarSuffix);
    
    const std::string backPropDelayMs = Utils::writePreciseString(dt * (double)(sg.getBackPropDelaySteps() + 1u)) + " + ";
    const std::string postOffset = sg.getTrgNeuronGroup()->isDelayRequired() ? "postReadDelayOffset + " : "";
    neuronSubstitutionsInSynapticCode(substitutions, sg.getTrgNeuronGroup(), postOffset, backPropDelayMs, postIdx, "_post", "Post", postVarPrefix, postVarSuffix);
}
}   // namespace CodeGenerator
