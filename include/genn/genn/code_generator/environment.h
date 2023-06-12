#pragma once

// Standard C++ includes
#include <functional>
#include <unordered_map>
#include <variant>

// GeNN includes
#include "gennUtils.h"
#include "varAccess.h"
#include "type.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/groupMerged.h"

// GeNN transpiler includes
#include "transpiler/prettyPrinter.h"
#include "transpiler/typeChecker.h"

namespace GeNN::Transpiler
{
class ErrorHandlerBase;
struct Token;
}
//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentExternalBase
//----------------------------------------------------------------------------
//! Base class for external environments i.e. those defines OUTSIDE of transpiled code by code generator
namespace GeNN::CodeGenerator
{
class EnvironmentExternalBase : public Transpiler::PrettyPrinter::EnvironmentBase, public Transpiler::TypeChecker::EnvironmentBase
{
public:
    explicit EnvironmentExternalBase(EnvironmentExternalBase &enclosing)
    :   m_Context(enclosing)
    {
    }

    explicit EnvironmentExternalBase(CodeStream &os)
    :   m_Context(os)
    {
    }

    EnvironmentExternalBase(const EnvironmentExternalBase&) = delete;

    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string define(const std::string &name) override;
   
    //------------------------------------------------------------------------
    // TypeChecker::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual void define(const Transpiler::Token &name, const GeNN::Type::ResolvedType &type, 
                        Transpiler::ErrorHandlerBase &errorHandler) override;

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    std::string operator[] (const std::string &name)
    {
        return getName(name);
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    //! Get stream exposed by context
    CodeStream &getContextStream() const;

    //! Get name from context if it provides this functionality
    std::string getContextName(const std::string &name, std::optional<Type::ResolvedType> type) const;

    //! Get vector of types from context if it provides this functionality
    std::vector<Type::ResolvedType> getContextTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler)  const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::variant<std::reference_wrapper<EnvironmentExternalBase>, std::reference_wrapper<CodeStream>> m_Context;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentExternal
//----------------------------------------------------------------------------
//! Minimal external environment, not tied to any sort of group - just lets you define things
class EnvironmentExternal : public EnvironmentExternalBase
{
public:
    using EnvironmentExternalBase::EnvironmentExternalBase;
    EnvironmentExternal(const EnvironmentExternal&) = delete;
    ~EnvironmentExternal();

    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string getName(const std::string &name, std::optional<Type::ResolvedType> type = std::nullopt) final;
    virtual CodeStream &getStream() final { return  m_Contents;; }

    //------------------------------------------------------------------------
    // TypeChecker::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Type::ResolvedType> getTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler) final;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Map a type (for type-checking) and a value (for pretty-printing) to an identifier
    void add(const GeNN::Type::ResolvedType &type, const std::string &name, const std::string &value,
             const std::vector<size_t> &initialisers = {}, const std::vector<std::string> &dependents = {});

    size_t addInitialiser(const std::string &initialiser);
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::ostringstream m_ContentsStream;
    CodeStream m_Contents;

    std::unordered_map<std::string, std::tuple<Type::ResolvedType, std::string, std::vector<size_t>, std::vector<std::string>>> m_Environment;
    std::vector<std::pair<bool, std::string>> m_Initialisers;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentGroupMergedField
//----------------------------------------------------------------------------
//! External environment, for substituting 
template<typename G>
class EnvironmentGroupMergedField : public EnvironmentExternalBase
{
    using GroupInternal = typename G::GroupInternal;
    using IsHeterogeneousFn = bool (G::*)(const std::string&) const;
    using IsVarInitHeterogeneousFn = bool (G::*)(const std::string&, const std::string&) const;

    using GroupInternal = typename G::GroupInternal;
    using GetVarSuffixFn = const std::string &(GroupInternal::*)(void) const;
    using GetParamValuesFn = const std::unordered_map<std::string, double> &(GroupInternal::*)(void) const;

    template<typename V>
    using GetVarReferencesFn = const std::unordered_map<std::string, V> &(GroupInternal::*)(void) const;

public:
    EnvironmentGroupMergedField(G &group, EnvironmentExternalBase &enclosing)
    :   EnvironmentExternalBase(enclosing), m_Group(group)
    {
    }
    EnvironmentGroupMergedField(G &group, CodeStream &os)
    :   EnvironmentExternalBase(os), m_Group(group)
    {
    }

    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string getName(const std::string &name, std::optional<Type::ResolvedType> type = std::nullopt) final
    {
        // If name isn't found in environment
        auto env = m_Environment.find(name);
        if (env == m_Environment.end()) {
            return getContextName(name, type);
        }
        // Otherwise, visit field in environment
        else {
            return "group->" + std::get<1>(env->second.second); 
        }
    }
    virtual CodeStream &getStream() final { return getContextStream(); }

    //------------------------------------------------------------------------
    // TypeChecker::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Type::ResolvedType> getTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler) final
    {
        // If name isn't found in environment
        auto env = m_Environment.find(name);
        if (env == m_Environment.end()) {
            return getContextType(name, type);
        }
        // Otherwise, return type
        else {
            // If field hasn't already been added
            if (!std::get<1>(env->second)) {
                // Call function to add field to underlying merged group
                const auto &field = std::get<2>(env->second);
                m_GroupMerged.addField(std::get<0>(field), std::get<1>(field),
                                       std::get<2>(field), std::get<3>(field));

                // Set flag so field doesn't get re-added
                std::get<1>(env->second) = true;
            }
            // Return type
            return {std::get<0>(env->second)};
        }
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Map a type (for type-checking) and a group merged field to back it to an identifier
    void add(const GeNN::Type::ResolvedType &type, const std::string &name,
             const GeNN::Type::ResolvedType &fieldType, const std::string &fieldName, typename G::GetFieldValueFunc getFieldValue,
             GroupMergedFieldType mergedFieldType = GroupMergedFieldType::STANDARD)
    {
        if(!m_Environment.try_emplace(name, std::piecewise_construct,
                                      std::forward_as_tuple(type),
                                      std::forward_as_tuple(false),
                                      std::forward_as_tuple(std::in_place, fieldType, fieldName, getFieldValue, mergedFieldType)).second) 
        {
            throw std::runtime_error("Redeclaration of '" + std::string{name} + "'");
        }
    }

    void addScalar(const std::string &name, const std::string &fieldSuffix, typename G::GetFieldDoubleValueFunc getFieldValue)
    {
        add(m_Group.getScalarType().addConst(), name,
            m_Group.getScalarType(), name + fieldSuffix,
            [getFieldValue, this](const auto &g, size_t i)
            {
                return getScalarString(getFieldValue(g, i);
            });
    }

    void addParams(const Snippet::Base::StringVec &paramNames, const std::string &fieldSuffix, 
                   GetParamValuesFn getParamValues, IsHeterogeneousFn isHeterogeneous)
    {
        // Loop through params
        for(const auto &p : paramNames) {
            if (std::invoke(isHeterogeneous, m_Group, p)) {
                addScalar(p, fieldSuffix,
                          [p, getParamValues](const auto &g, size_t)
                          {
                              return std::invoke(getParamValues, g).at(p);
                          });
            }
            // Otherwise, just add a const-qualified scalar to the type environment
            else {
                add(m_Group.getScalarType().addConst(), p, getScalarString(std::invoke(getParamValues, m_Group.getArchetype()).at(p)));
            }
        }
    }

    void addDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, const std::string &fieldSuffix,
                          GetParamValuesFn getDerivedParamValues, IsHeterogeneousFn isHeterogeneous)
    {
        // Loop through derived params
        for(const auto &d : derivedParams) {
            if (std::invoke(isHeterogeneous, m_Group, d.name)) {
                addScalar(d.name, fieldSuffix,
                          [d, getDerivedParamValues](const auto &g, size_t)
                          {
                              return std::invoke(getDerivedParamValues, g).at(d.name);
                          });
            }
            else {
                add(m_Group.getScalarType().addConst(), d.name, getScalarString(std::invoke(getDerivedParamValues, m_Group).at(d.name));
            }
        }
    }

    template<typename A>
    void addVarInitParams(IsVarInitHeterogeneousFn isHeterogeneous, const std::string &fieldSuffix = "")
    {
        // Loop through weight update model variables
        const A archetypeAdaptor(m_Group.getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // Loop through parameters
            for(const auto &p : archetypeAdaptor.getInitialisers().at(v.name).getParams()) {
                if(std::invoke(isHeterogeneous, m_Group, v.name, p.first)) {
                    defineScalarField(p.first, v.name + fieldSuffix,
                                      [p, v](const auto &g, size_t)
                                      {
                                          return  A(g).getInitialisers().at(v.name).getParams().at(p.first);
                                       });
                }
                else {
                    defineField(m_Group.getScalarType().addConst(), p.first);
                }
            }
        }
    }

    template<typename A>
    void addVarInitDerivedParams(IsVarInitHeterogeneousFn isHeterogeneous, const std::string &fieldSuffix = "")
    {
        // Loop through weight update model variables
        const A archetypeAdaptor(m_Group.getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // Loop through parameters
            for(const auto &p : archetypeAdaptor.getInitialisers().at(v.name).getDerivedParams()) {
                if(std::invoke(isHeterogeneous, m_Group, v.name, p.first)) {
                    defineScalarField(p.first, v.name + fieldSuffix,
                                      [p, v](const auto &g, size_t)
                                      {
                                          return A(g).getInitialisers().at(v.name).getDerivedParams().at(p.first);
                                      });
                }
                else {
                    defineField(m_Group.getScalarType().addConst(), p.first);
                }
            }
        }
    }

    template<typename A>
    void addVars(const std::string &arrayPrefix, const std::string &fieldSuffix = "")
    {
        // Loop through variables
        const A archetypeAdaptor(m_Group.getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            const auto resolvedType = v.type.resolve(m_Group.getTypeContext())
            const auto qualifiedType = (getVarAccessMode(v.access) & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
            add(qualifiedType, v.name,
                resolvedType.createPointer(), v.name + fieldSuffix, 
                [arrayPrefix, v](const auto &g, size_t) 
                { 
                    return prefix + v.name + A(g).getNameSuffix();
                });
        }
    }

    template<typename A>
    void addVarRefs(const std::string &arrayPrefix, const std::string &fieldSuffix = "")
    {
        // Loop through variable references
        const A archetypeAdaptor(m_Group.getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // If variable access is read-only, qualify type with const
            const auto resolvedType = v.type.resolve(m_Group.getTypeContext());
            const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
            defineField(qualifiedType, v.name,
                        resolvedType.createPointer(), v.name + fieldSuffix,
                        [arrayPrefix, v](const auto &g, size_t) 
                        { 
                            const auto varRef = A(g).getInitialisers().at(v.name);
                            return arrayPrefix + varRef.getVar().name + varRef.getTargetName(); 
                        });
        }
    }
  
    template<typename A>
    void addEGPs(const std::string &arrayPrefix, const std::string &varName = "", const std::string &fieldSuffix = "")
    {
        // Loop through EGPs
        const A archetypeAdaptor(m_Group.getArchetype());
        for(const auto &e : archetypeAdaptor.getDefs()) {
            const auto pointerType = e.type.resolve(m_Group.getTypeContext()).createPointer();
            defineField(pointerType, e.name,
                        pointerType, e.name + varName + fieldSuffix,
                        [arrayPrefix, e, varName](const auto &g, size_t) 
                        {
                            return arrayPrefix + e.name + varName + g.getName(); 
                        },
                        GroupMergedFieldType::DYNAMIC);
        }
    }

private:
    std::string getScalarString(double scalar) const
    {
        return (Utils::writePreciseString(scalar, m_GroupMerged.getScalarType().getNumeric().maxDigits10) 
                + m_GroupMerged.getScalarType().getNumeric().literalSuffix));
    }
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::reference_wrapper<G> m_Group;

    //! Environment mapping names to types to fields to pull values from
    std::unordered_map<std::string, std::tuple<Type::ResolvedType, bool, std::variant<std::string, typename G::Field>>> m_Environment;
};


//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentSubstitute
//----------------------------------------------------------------------------
//! Standard pretty printing environment simply allowing substitutions to be implemented
class EnvironmentSubstitute : public EnvironmentExternalBase
{
public:
    EnvironmentSubstitute(EnvironmentExternalBase &enclosing)
    :   EnvironmentExternalBase(static_cast<EnvironmentExternalBase&>(enclosing)), m_Contents(m_ContentsStream)
    {
    }
    
    EnvironmentSubstitute(CodeStream &os)
    :   EnvironmentExternalBase(os), m_Contents(m_ContentsStream)
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
    void addVarNameSubstitution(const std::vector<T> &variables, const std::string &fieldSuffix = "")
    {
        for(const auto &v : variables) {
            addSubstitution(v.name, "group->" + v.name + fieldSuffix);
        }
    }

    template<typename G>
    void addParamValueSubstitution(const std::vector<std::string> &paramNames, const std::unordered_map<std::string, double> &values, 
                                   const std::string &fieldSuffix, G isHeterogeneousFn)
    {
        if(paramNames.size() != values.size()) {
            throw std::runtime_error("Number of parameters does not match number of values");
        }

        for(const auto &p : paramNames) {
            if(isHeterogeneousFn(p)) {
                addSubstitution(p, "group->" + p + fieldSuffix);
            }
            else {
                // **TODO** scalar suffix
                addSubstitution(p, Utils::writePreciseString(values.at(p)));
            }
        }
    }

    template<typename T, typename G>
    void addVarValueSubstitution(const std::vector<T> &variables, const std::unordered_map<std::string, double> &values, 
                                 const std::string &fieldSuffix, G isHeterogeneousFn)
    {
        if(variables.size() != values.size()) {
            throw std::runtime_error("Number of variables does not match number of values");
        }

        for(const auto &v : variables) {
            if(isHeterogeneousFn(v.name)) {
                addSubstitution(v.name, "group->" + v.name + fieldSuffix);
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
class EnvironmentLocalVarCache : public EnvironmentExternalBase
{
    //! Type of a single definition
    using DefType = typename std::invoke_result_t<decltype(&A::getDefs), A>::value_type;

    //! Type of a single initialiser
    using InitialiserType = typename std::remove_reference_t<std::invoke_result_t<decltype(&A::getInitialisers), A>>::mapped_type;

    //! Function used to provide index strings based on initialiser and access type
    using GetIndexFn = std::function<std::string(const std::string&, InitialiserType, decltype(DefType::access))>;

public:
    EnvironmentLocalVarCache(const G &group, const Type::TypeContext &context, EnvironmentExternal &enclosing, 
                             const std::string &fieldSuffix, const std::string & localPrefix,
                             GetIndexFn getReadIndex, GetIndexFn getWriteIndex)
    :   EnvironmentExternal(static_cast<EnvironmentBase&>(enclosing)), m_Group(group), m_Context(context), m_Contents(m_ContentsStream), 
        m_FieldSuffix(fieldSuffix), m_LocalPrefix(localPrefix), m_GetReadIndex(getReadIndex), m_GetWriteIndex(getWriteIndex)
    {
        // Add name of each definition to map, initially with value set to value
        const auto defs = A(m_Group).getDefs();
        std::transform(defs.cbegin(), defs.cend(), std::inserter(m_VariablesReferenced, m_VariablesReferenced.end()),
                       [](const auto &v){ return std::make_pair(v.name, false); });
    }

    EnvironmentLocalVarCache(const G &group, const Type::TypeContext &context, EnvironmentExternal &enclosing, 
                             const std::string &fieldSuffix, const std::string & localPrefix, GetIndexFn getIndex)
    :   EnvironmentExternal(static_cast<EnvironmentBase&>(enclosing)), m_Group(group), m_Context(context), 
        m_Contents(m_ContentsStream), m_FieldSuffix(fieldSuffix), m_LocalPrefix(localPrefix), m_GetReadIndex(getIndex), m_GetWriteIndex(getIndex)
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
                getContextStream() << " = group->" << v.name << m_FieldSuffix << "[" << m_GetReadIndex(v.name, initialisers.at(v.name), v.access) << "]";
            }
            getContextStream() << ";" << std::endl;
        }

        // Write contents to context stream
        getContextStream() << m_ContentsStream.str();

        // Loop through referenced variables again
        for(const auto &v : referencedVars) {
            // If variables are read-write
            if(v.access & VarAccessMode::READ_WRITE) {
                getContextStream() << "group->" << v.name << m_FieldSuffix << "[" << m_GetWriteIndex(v.name, initialisers.at(v.name), v.access) << "]";
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
    std::string m_FieldSuffix;
    std::string m_LocalPrefix;
    GetIndexFn m_GetReadIndex;
    GetIndexFn m_GetWriteIndex;
    std::unordered_map<std::string, bool> m_VariablesReferenced;
};
}   // namespace GeNN::CodeGenerator
