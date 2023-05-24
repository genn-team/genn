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
    MathsFuncCPP,
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
void ensureMathFunctionFtype(std::string &code)
{
    // Replace any outstanding explicit single-precision maths functions  
    // with C++ versions where overloads should work the same
    for(const auto &m : mathsFuncs) {
        GeNN::CodeGenerator::regexFuncSubstitute(code, m[MathsFuncSingle], m[MathsFuncCPP]);
    }
}
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
//--------------------------------------------------------------------------
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

std::string trimWhitespace(const std::string& str)
{
    const std::string whitespace = " \t\r\n";
    
    // If string is all whitespace, return empty
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos) {
        return ""; 
    }

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}
}    // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
void substitute(std::string &s, const std::string &trg, const std::string &rep)
{
    size_t found= s.find(trg);
    while (found != std::string::npos) {
        s.replace(found,trg.length(),rep);
        found= s.find(trg);
    }
}
//----------------------------------------------------------------------------
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

//----------------------------------------------------------------------------
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
//----------------------------------------------------------------------------
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
        size_t idx = code.find(funcStart);

        // While functions are found
        while (idx != std::string::npos) {
            // Loop through characters following funcStart
            unsigned int bracketDepth = 0;
            const size_t startIdx = idx;
            for(idx = idx + funcStart.length(); idx < code.size(); idx++) {
                // If this character is a comma at function bracket depth
                if(code[idx] == ',' && bracketDepth == 0) {
                    currentParam = trimWhitespace(currentParam);
                    assert(!currentParam.empty());

                    // Add parameter to array
                    params.push_back(currentParam);
                    currentParam = "";
                }
                // Otherwise
                else {
                    // If this is an open bracket, increase bracket depth
                    if(code[idx] == '(') {
                        bracketDepth++;
                    }
                    // Otherwise, it's a close bracket
                    else if(code[idx] == ')') {
                        // If we are at a deeper bracket depth than function, decrease bracket depth
                        if(bracketDepth > 0) {
                            bracketDepth--;
                        }
                        // Otherwise
                        else {
                            currentParam = trimWhitespace(currentParam);
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
                            code.replace(startIdx, idx - startIdx + 1, replaceFunc);
                            break;
                        }
                    }

                    // Add character to parameter string
                    currentParam += code[idx];
                }
            }

            // Find start of next function to replace
            idx = code.find(funcStart, idx);
        }
    }
}
//----------------------------------------------------------------------------
void genTypeRange(CodeStream &os, const Type::ResolvedType &type, const std::string &prefix)
{
    const auto &numeric = type.getNumeric();
    os << "#define " << prefix << "_MIN " << Utils::writePreciseString(numeric.min, numeric.maxDigits10) << numeric.literalSuffix << std::endl << std::endl;

    os << "#define " << prefix << "_MAX " << Utils::writePreciseString(numeric.max, numeric.maxDigits10) << numeric.literalSuffix << std::endl;
}
//----------------------------------------------------------------------------
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
    ensureMathFunctionFtype(code);
    return code;
}
//----------------------------------------------------------------------------
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
//----------------------------------------------------------------------------
std::string disambiguateNamespaceFunction(const std::string supportCode, const std::string code, std::string namespaceName) {
    // Regex for function call - looks for words with succeeding parentheses with or without any data inside the parentheses (arguments)
    std::regex funcCallRegex(R"(\w+(?=\(.*\)))");
    std::smatch matchedInCode;
    std::regex_search(code.begin(), code.end(), matchedInCode, funcCallRegex);
    std::string newCode = code;

    // Regex for function definition - looks for words with succeeding parentheses with or without any data inside the parentheses (arguments) followed by braces on the same or new line
    std::regex supportCodeRegex(R"(\w+(?=\(.*\)\s*\{))");
    std::smatch matchedInSupportCode;
    std::regex_search(supportCode.begin(), supportCode.end(), matchedInSupportCode, supportCodeRegex);

    // Iterating each function in code
    for (const auto& funcInCode : matchedInCode) {
        // Iterating over every function in support code to check if that function is indeed defined in support code (and not called - like fmod())
        for (const auto& funcInSupportCode : matchedInSupportCode) {
            if (funcInSupportCode.str() == funcInCode.str()) {
                newCode = std::regex_replace(newCode, std::regex(funcInCode.str()), namespaceName + "_$&");
                break;
            }
        }
    }
    return newCode;
}
//----------------------------------------------------------------------------
std::string upgradeCodeString(const std::string &codeString)
{
    // **TODO** snake-case -> camel case known built in variables e.g id_pre -> idPre
    // **TODO** old style function call to standard C (these are ambiguous so need to be applied to existing genn functions)
    std::regex variable(R"(\$\(([_a-zA-Z][a-zA-Z0-9]*)\))");
 
    return std::regex_replace(codeString, variable, "$1");
}

}   // namespace GeNN::CodeGenerator
