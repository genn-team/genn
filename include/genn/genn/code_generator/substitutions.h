#pragma once

// Standard C++ includes
#include <map>
#include <stdexcept>
#include <string>

// Standard C includes
#include <cassert>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "logging.h"

//--------------------------------------------------------------------------
// Substitutions
//--------------------------------------------------------------------------
namespace CodeGenerator
{
class GENN_EXPORT Substitutions
{
public:
    //! Immutable structure for specifying how to implement
    //! a generic function e.g. gennrand_uniform
    /*! **NOTE** for the sake of easy initialisation first two parameters of GenericFunction are repeated (C++17 fixes) */
    struct FunctionTemplate
    {
        // **HACK** while GCC and CLang automatically generate this fine/don't require it, VS2013 seems to need it
        FunctionTemplate operator = (const FunctionTemplate &o)
        {
            return FunctionTemplate{o.genericName, o.numArguments, o.funcTemplate};
        }

        //! Generic name used to refer to function in user code
        const std::string genericName;

        //! Number of function arguments
        const unsigned int numArguments;

        //! The function template (for use with ::functionSubstitute) used when model uses double precision
        const std::string funcTemplate;
    };

    Substitutions(const Substitutions *parent = nullptr) : m_Parent(parent)
    {
        assert(m_Parent != this);
    }

    Substitutions(const std::vector<FunctionTemplate> &functions) : m_Parent(nullptr)
    {
        // Loop through functions and add as substitutions
        for(const auto &f: functions) {
            addFuncSubstitution(f.genericName, f.numArguments, f.funcTemplate);
        }
    }

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    template<typename T>
    void addVarNameSubstitution(const std::vector<T> &variables, const std::string &sourceSuffix = "",
                                const std::string &destPrefix = "", const std::string &destSuffix = "")
    {
        for(const auto &v : variables) {
            addVarSubstitution(v.name + sourceSuffix,
                               destPrefix + v.name + destSuffix);
        }
    }

    template<typename T, typename S, typename F>
    void addVarNameSubstitution(const std::vector<T> &variables, const std::string &sourceSuffix,
                                const std::string &destPrefix, S getDestSuffixFn, F filterFn)
    {
        for(size_t i = 0; i < variables.size(); i++) {
            const T &v = variables[i];
            if (filterFn(v.access, i)) {
                addVarSubstitution(v.name + sourceSuffix,
                                   destPrefix + v.name + getDestSuffixFn(v.access, i));
            }
        }
    }

    template<typename T, typename S>
    void addVarNameSubstitution(const std::vector<T> &variables, const std::string &sourceSuffix,
                                const std::string &destPrefix, S getDestSuffixFn)
    {
        typedef decltype(T::access) AccessType;
        addVarNameSubstitution(variables, sourceSuffix, destPrefix, 
                               getDestSuffixFn, [](AccessType, size_t) { return true; });
    }

    template<typename T>
    void addVarValueSubstitution(const std::vector<T> &variables, const std::vector<double> &values,
                                 const std::string &sourceSuffix = "")
    {
        if(variables.size() != values.size()) {
            throw std::runtime_error("Number of variables does not match number of values");
        }

        auto var = variables.cbegin();
        auto val = values.cbegin();
        for (;var != variables.cend() && val != values.cend(); var++, val++) {
            addVarSubstitution(var->name + sourceSuffix,
                               "(" + Utils::writePreciseString(*val) + ")");
        }
    }

    void addParamValueSubstitution(const std::vector<std::string> &paramNames, const std::vector<double> &values,
                                   const std::string &sourceSuffix = "");

    template<typename G>
    void addParamValueSubstitution(const std::vector<std::string> &paramNames, const std::vector<double> &values, G isHeterogeneousFn,
                                   const std::string &sourceSuffix = "", const std::string &destPrefix = "", const std::string &destSuffix = "")
    {
        if(paramNames.size() != values.size()) {
            throw std::runtime_error("Number of parameters does not match number of values");
        }

        for(size_t i = 0; i < paramNames.size(); i++) {
            if(isHeterogeneousFn(i)) {
                addVarSubstitution(paramNames[i] + sourceSuffix,
                                   destPrefix + paramNames[i] + destSuffix);
            }
            else {
                addVarSubstitution(paramNames[i] + sourceSuffix,
                                   "(" + Utils::writePreciseString(values[i]) + ")");
            }
        }
    }

    template<typename T, typename G>
    void addVarValueSubstitution(const std::vector<T> &variables, const std::vector<double> &values, G isHeterogeneousFn,
                                 const std::string &sourceSuffix = "", const std::string &destPrefix = "", const std::string &destSuffix = "")
    {
        if(variables.size() != values.size()) {
            throw std::runtime_error("Number of variables does not match number of values");
        }

        for(size_t i = 0; i < variables.size(); i++) {
            if(isHeterogeneousFn(i)) {
                addVarSubstitution(variables[i].name + sourceSuffix,
                                   destPrefix + variables[i].name + destSuffix);
            }
            else {
                addVarSubstitution(variables[i].name + sourceSuffix,
                                   "(" + Utils::writePreciseString(values[i]) + ")");
            }
        }
    }

    void addVarSubstitution(const std::string &source, const std::string &destionation, bool allowOverride = false);
    void addFuncSubstitution(const std::string &source, unsigned int numArguments, const std::string &funcTemplate, bool allowOverride = false);
    bool hasVarSubstitution(const std::string &source) const;

    const std::string &getVarSubstitution(const std::string &source) const;

    void apply(std::string &code) const;
    void applyCheckUnreplaced(std::string &code, const std::string &context) const;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    const std::string operator[] (const std::string &source) const
    {
        return getVarSubstitution(source);
    }

private:
    //--------------------------------------------------------------------------
    // Private API
    //--------------------------------------------------------------------------
    void applyFuncs(std::string &code) const;
    void applyVars(std::string &code) const;

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    std::map<std::string, std::string> m_VarSubstitutions;
    std::map<std::string, std::pair<unsigned int, std::string>> m_FuncSubstitutions;
    const Substitutions *m_Parent;
};
}   // namespace CodeGenerator
