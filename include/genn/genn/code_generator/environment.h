#pragma once

// Standard C++ includes
#include <functional>
#include <unordered_map>
#include <variant>

// GeNN includes
#include "gennUtils.h"
#include "varAccess.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"

// GeNN transpiler includes
#include "transpiler/prettyPrinter.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentExternal
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class EnvironmentExternal : public Transpiler::PrettyPrinter::EnvironmentBase
{
protected:
    using EnvironmentBase = Transpiler::PrettyPrinter::EnvironmentBase;
public:
    EnvironmentExternal(EnvironmentBase &enclosing)
    :   m_Context(enclosing)
    {
    }

    EnvironmentExternal(CodeStream &os)
    :   m_Context(os)
    {
    }

    EnvironmentExternal(const EnvironmentExternal&) = delete;

    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string define(const std::string &name);
    
protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    auto &getContext() const{ return m_Context; }

    CodeStream &getContextStream() const;

    std::string getContextName(const std::string &name, std::optional<Type::ResolvedType> type) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::variant<std::reference_wrapper<EnvironmentBase>, std::reference_wrapper<CodeStream>> m_Context;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentSubstitute
//----------------------------------------------------------------------------
//! Standard pretty printing environment simply allowing substitutions to be implemented
class EnvironmentSubstitute : public EnvironmentExternal
{
public:
    EnvironmentSubstitute(EnvironmentSubstitute &enclosing)
    :   EnvironmentExternal(static_cast<EnvironmentBase&>(enclosing)), m_Contents(m_ContentsStream)
    {
    }

    EnvironmentSubstitute(EnvironmentExternal &enclosing)
    :   EnvironmentExternal(static_cast<EnvironmentBase&>(enclosing)), m_Contents(m_ContentsStream)
    {
    }
    
    EnvironmentSubstitute(CodeStream &os)
    :   EnvironmentExternal(os), m_Contents(m_ContentsStream)
    {
    }

    EnvironmentSubstitute(const EnvironmentSubstitute&) = delete;

    ~EnvironmentSubstitute();

    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string getName(const std::string &name, std::optional<Type::ResolvedType> type = std::nullopt) final;

    virtual CodeStream &getStream() final
    {
        return m_Contents;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void addSubstitution(const std::string &source, const std::string &destination,
                         std::vector<size_t> initialisers = {});

    size_t addInitialiser(const std::string &initialiser);

    template<typename T>
    void addVarNameSubstitution(const std::vector<T> &variables)
    {
        for(const auto &v : variables) {
            addSubstitution(v.name, "group->" + v.name);
        }
    }

    template<typename G>
    void addParamValueSubstitution(const std::vector<std::string> &paramNames, const std::unordered_map<std::string, double> &values, 
                                   G isHeterogeneousFn)
    {
        if(paramNames.size() != values.size()) {
            throw std::runtime_error("Number of parameters does not match number of values");
        }

        for(const auto &p : paramNames) {
            if(isHeterogeneousFn(p)) {
                addSubstitution(p, "group->" + p);
            }
            else {
                // **TODO** scalar suffix
                addSubstitution(p, Utils::writePreciseString(values.at(p)));
            }
        }
    }

    template<typename T, typename G>
    void addVarValueSubstitution(const std::vector<T> &variables, const std::unordered_map<std::string, double> &values, 
                                 G isHeterogeneousFn)
    {
        if(variables.size() != values.size()) {
            throw std::runtime_error("Number of variables does not match number of values");
        }

        for(const auto &v : variables) {
            if(isHeterogeneousFn(v.name)) {
                addSubstitution(v.name, "group->" + v.name);
            }
            else {
                addSubstitution(v.name, Utils::writePreciseString(values.at(v.name)));
            }
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::ostringstream m_ContentsStream;
    CodeStream m_Contents;
    std::unordered_map<std::string, std::pair<std::string, std::vector<size_t>>> m_VarSubstitutions;
    std::vector<std::pair<bool, std::string>> m_Initialisers;

};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentLocalVarCache
//----------------------------------------------------------------------------
//! Pretty printing environment which caches used variables in local variables
template<typename A, typename G>
class EnvironmentLocalVarCache : public EnvironmentExternal
{
    //! Type of a single definition
    using DefType = typename std::invoke_result_t<decltype(&A::getDefs), A>::value_type;

    //! Type of a single initialiser
    using InitialiserType = typename std::remove_reference_t<std::invoke_result_t<decltype(&A::getInitialisers), A>>::mapped_type;

    //! Function used to provide index strings based on initialiser and access type
    using GetIndexFn = std::function<std::string(const std::string&, InitialiserType, decltype(DefType::access))>;

public:
    EnvironmentLocalVarCache(const G &group, const Type::TypeContext &context, EnvironmentExternal &enclosing, 
                             GetIndexFn getReadIndex, GetIndexFn getWriteIndex, const std::string &localPrefix = "l")
    :   EnvironmentExternal(static_cast<EnvironmentBase&>(enclosing)), m_Group(group), m_Context(context), m_Contents(m_ContentsStream), 
        m_LocalPrefix(localPrefix), m_GetReadIndex(getReadIndex), m_GetWriteIndex(getWriteIndex)
    {
        // Add name of each definition to map, initially with value set to value
        const auto defs = A(m_Group).getDefs();
        std::transform(defs.cbegin(), defs.cend(), std::inserter(m_VariablesReferenced, m_VariablesReferenced.end()),
                       [](const auto &v){ return std::make_pair(v.name, false); });
    }

    EnvironmentLocalVarCache(const G &group, const Type::TypeContext &context, EnvironmentExternal &enclosing, GetIndexFn getIndex, const std::string &localPrefix = "l")
    :   EnvironmentExternal(static_cast<EnvironmentBase&>(enclosing)), m_Group(group), m_Context(context), 
        m_Contents(m_ContentsStream), m_LocalPrefix(localPrefix), m_GetReadIndex(getIndex), m_GetWriteIndex(getIndex)
    {
        // Add name of each definition to map, initially with value set to value
        const auto defs = A(m_Group).getDefs();
        std::transform(defs.cbegin(), defs.cend(), std::inserter(m_VariablesReferenced, m_VariablesReferenced.end()),
                       [](const auto &v){ return std::make_pair(v.name, false); });
    }

    EnvironmentLocalVarCache(const EnvironmentLocalVarCache&) = delete;

    ~EnvironmentLocalVarCache()
    {
        A adapter(m_Group);

        // Copy definitions which have been referenced into new vector
        const auto defs = adapter.getDefs();
        std::remove_const_t<decltype(defs)> referencedVars;
        std::copy_if(defs.cbegin(), defs.cend(), std::back_inserter(referencedVars),
                     [this](const auto &v){ return m_VariablesReferenced.at(v.name); });

        // Loop through referenced variables
        const auto &initialisers = adapter.getInitialisers();
        for(const auto &v : referencedVars) {
            if(v.access & VarAccessMode::READ_ONLY) {
                getContextStream() << "const ";
            }
            getContextStream() << v.type.resolve(m_Context).getName() << " " << m_LocalPrefix << v.name;

            // If this isn't a reduction, read value from memory
            // **NOTE** by not initialising these variables for reductions, 
            // compilers SHOULD emit a warning if user code doesn't set it to something
            if(!(v.access & VarAccessModeAttribute::REDUCE)) {
                getContextStream() << " = group->" << v.name << "[" << m_GetReadIndex(v.name, initialisers.at(v.name), v.access) << "]";
            }
            getContextStream() << ";" << std::endl;
        }

        // Write contents to context stream
        getContextStream() << m_ContentsStream.str();

        // Loop through referenced variables again
        for(const auto &v : referencedVars) {
            // If variables are read-write
            if(v.access & VarAccessMode::READ_WRITE) {
                getContextStream() << "group->" << v.name << "[" << m_GetWriteIndex(v.name, initialisers.at(v.name), v.access) << "]";
                getContextStream() << " = " << m_LocalPrefix << v.name << ";" << std::endl;
            }
        }
    }

    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string getName(const std::string &name, std::optional<Type::ResolvedType> type = std::nullopt) final
    {
        // If variable with this name isn't found, try and get name from context
        auto var = m_VariablesReferenced.find(name);
        if(var == m_VariablesReferenced.end()) {
            return getContextName(name, type);
        }
        // Otherwise
        else {
            // Set flag to indicate that variable has been referenced
            var->second = true;

            // Add local prefix to variable name
            return m_LocalPrefix + name;
        }
    }

    virtual CodeStream &getStream() final
    {
        return m_Contents;
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const G &m_Group;
    const Type::TypeContext &m_Context;
    std::ostringstream m_ContentsStream;
    CodeStream m_Contents;
    std::string m_LocalPrefix;
    GetIndexFn m_GetReadIndex;
    GetIndexFn m_GetWriteIndex;
    std::unordered_map<std::string, bool> m_VariablesReferenced;
};
}   // namespace GeNN::CodeGenerator
