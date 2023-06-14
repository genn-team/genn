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
#include "transpiler/token.h"
#include "transpiler/typeChecker.h"

// Forward declarations
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
// GeNN::CodeGenerator::EnvironmentLibrary
//----------------------------------------------------------------------------
class EnvironmentLibrary : public EnvironmentExternalBase
{
public:
    using Library = std::unordered_multimap<std::string, std::pair<Type::ResolvedType, std::string>>;

    EnvironmentLibrary(EnvironmentExternalBase &enclosing, const Library &library)
    :   EnvironmentExternalBase(enclosing), m_Library(library)
    {}

    EnvironmentLibrary(CodeStream &os, const Library &library)
    :   EnvironmentExternalBase(os), m_Library(library)
    {}

    //------------------------------------------------------------------------
    // TypeChecker::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Type::ResolvedType> getTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler) final;

    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string getName(const std::string &name, std::optional<Type::ResolvedType> type = std::nullopt) final;
    virtual CodeGenerator::CodeStream &getStream() final;

private:
    std::reference_wrapper<const Library> m_Library;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentSubstitutionPolicy
//----------------------------------------------------------------------------
struct EnvironmentSubstitutionPolicy
{
    using Payload = std::string;

    std::string getName(const std::string &payload)
    {
        return payload;
    }

    void setRequired(std::string&)
    {
    }
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentFieldPolicy
//----------------------------------------------------------------------------
template<typename G>
struct EnvironmentFieldPolicy
{
    using Payload = std::tuple<bool, std::string, std::optional<typename G::Field>>;

    EnvironmentFieldPolicy(G &group) : m_Group(group)
    {
    }

    std::string getName(const Payload &payload)
    {
        // If a field is specified
        if(std::get<2>(payload)) {
            return "group->" + std::get<1>(std::get<2>(payload).get()) + std::get<1>(payload); 
        }
        // Otherwise, use value directly
        else {
            assert(!std::get<1>(payload).empty());
            return std::get<1>(payload); 
        }
    }

    void setRequired(Payload &payload)
    {
        // If a field is specified but it hasn't already been added
        if (std::get<2>(payload) && !std::get<0>(payload)) {
            // Call function to add field to underlying merged group
            const auto &field = std::get<2>(payload).get();
            m_Group.addField(std::get<0>(field), std::get<1>(field),
                             std::get<2>(field), std::get<3>(field));

            // Set flag so field doesn't get re-added
            std::get<0>(payload) = true;
        }
    }

private:
    std::reference_wrapper<G> m_Group;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentExternalDynamicBase
//----------------------------------------------------------------------------
template<typename P>
class EnvironmentExternalDynamicBase : public EnvironmentExternalBase, private P
{
public:
    template<typename... PolicyArgs>
    EnvironmentExternalDynamicBase(EnvironmentExternalBase &enclosing, PolicyArgs&&... policyArgs)
    :   EnvironmentExternalBase(enclosing), P(std::forward<PolicyArgs>(policyArgs)...)
    {}

    template<typename... PolicyArgs>
    EnvironmentExternalDynamicBase(CodeStream &os, PolicyArgs&&... policyArgs)
    :   EnvironmentExternalBase(os), P(std::forward<PolicyArgs>(policyArgs)...)
    {}

    ~EnvironmentExternalDynamicBase()
    {
        // Loop through initialiser
        for(const auto &i : m_Initialisers) {
            // If variable requiring initialiser has been referenced, write out initialiser
            if (i.first) {
                getContextStream() << i.second << std::endl;
            }
        }
        
        // Write contents to context stream
        getContextStream() << m_ContentsStream.str();
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
        // Otherwise, get name from payload
        else {
            return getName(std::get<3>(env->second));
        }
    }

    virtual CodeStream &getStream() final { return  m_Contents; }

    //------------------------------------------------------------------------
    // TypeChecker::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Type::ResolvedType> getTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler) final
    {
        // If name isn't found in environment
        auto env = m_Environment.find(name.lexeme);
        if (env == m_Environment.end()) {
            return getContextTypes(name, errorHandler);
        }
        // Otherwise
        else {
            // If this identifier relies on any initialiser statements, mark these initialisers as required
            for(size_t i : std::get<1>(env->second)) {
                m_Initialisers.at(i).first = true;
            }

            // If this identifier relies on any others, get their types
            // **YUCK**
            for(const std::string &id : std::get<2>(env->second)) {
                getTypes(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, id, 0}, 
                         errorHandler);
            }

            // Perform any type-specific logic to mark this identifier as required
            setRequired(std::get<3>(env->second));

            // Return type of variables
            return {std::get<0>(env->second)};
        }
    }

   
    size_t addInitialiser(const std::string &initialiser)
    {
        m_Initialisers.emplace_back(false, initialiser);
        return (m_Initialisers.size() - 1);
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    //! Map an identifier to a type (for type-checking), lists of initialisers and dependencies and a payload 
    void addInternal(const GeNN::Type::ResolvedType &type, const std::string &name, const typename P::Payload &payload,
                     const std::vector<size_t> &initialisers = {}, const std::vector<std::string> &dependents = {})
    {
        if(!m_Environment.try_emplace(name, type, initialisers, dependents, payload).second) {
            throw std::runtime_error("Redeclaration of '" + std::string{name} + "'");
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::ostringstream m_ContentsStream;
    CodeStream m_Contents;

    std::unordered_map<std::string, std::tuple<Type::ResolvedType, std::vector<size_t>, std::vector<std::string>, typename P::Payload>> m_Environment;
    std::vector<std::pair<bool, std::string>> m_Initialisers;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentExternal
//----------------------------------------------------------------------------
//! Minimal external environment, not tied to any sort of group - just lets you define things
class EnvironmentExternal : public EnvironmentExternalDynamicBase<EnvironmentSubstitutionPolicy>
{
public:
    using EnvironmentExternalDynamicBase::EnvironmentExternalDynamicBase;
    EnvironmentExternal(const EnvironmentExternal&) = delete;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Map a type (for type-checking) and a value (for pretty-printing) to an identifier
    void add(const GeNN::Type::ResolvedType &type, const std::string &name, const std::string &value,
             const std::vector<size_t> &initialisers = {}, const std::vector<std::string> &dependents = {})
    {
        addInternal(type, name, value, initialisers, dependents);
    }
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentGroupMergedField
//----------------------------------------------------------------------------
//! External environment, for substituting 
template<typename G>
class EnvironmentGroupMergedField : public EnvironmentExternalDynamicBase<EnvironmentFieldPolicy<G>>
{
    using GroupInternal = typename G::GroupInternal;
    using IsHeterogeneousFn = bool (G::*)(const std::string&) const;
    using IsVarInitHeterogeneousFn = bool (G::*)(const std::string&, const std::string&) const;

    using GetVarSuffixFn = const std::string &(GroupInternal::*)(void) const;
    using GetParamValuesFn = const std::unordered_map<std::string, double> &(GroupInternal::*)(void) const;

    template<typename V>
    using GetVarReferencesFn = const std::unordered_map<std::string, V> &(GroupInternal::*)(void) const;

public:
    using EnvironmentExternalDynamicBase::EnvironmentExternalDynamicBase;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Map a type and a value to an identifier
    void add(const GeNN::Type::ResolvedType &type, const std::string &name, const std::string &value,
             const std::vector<size_t> &initialisers = {}, const std::vector<std::string> &dependents = {})
    {
        addInternal(type, name, std::make_tuple(false, value, std::nullopt),
                    initialisers, dependents);
    }

    //! Map a type (for type-checking) and a group merged field to back it to an identifier
    void add(const GeNN::Type::ResolvedType &type, const std::string &name,
             const GeNN::Type::ResolvedType &fieldType, const std::string &fieldName, typename G::GetFieldValueFunc getFieldValue,
             const std::string &indexSuffix = "", GroupMergedFieldType mergedFieldType = GroupMergedFieldType::STANDARD,
             const std::vector<size_t> &initialisers = {}, const std::vector<std::string> &dependents = {})
    {
         addInternal(type, name, std::make_tuple(false, indexSuffix, std::forward_as_tuple(std::in_place, fieldType, fieldName, getFieldValue, mergedFieldType)),
                    initialisers, dependents);
    }

    void addScalar(const std::string &name, const std::string &fieldSuffix, typename G::GetFieldDoubleValueFunc getFieldValue)
    {
        add(m_Group.getScalarType().addConst(), name,
            m_Group.getScalarType(), name + fieldSuffix,
            [getFieldValue, this](const auto &g, size_t i)
            {
                return getScalarString(getFieldValue(g, i));
            });
    }

    void addParams(const Snippet::Base::StringVec &paramNames, const std::string &fieldSuffix, 
                   GetParamValuesFn getParamValues, IsHeterogeneousFn isHeterogeneous)
    {
        // Loop through params
        for(const auto &p : paramNames) {
            // If parameter is heterogeneous, add scalar field
            if (std::invoke(isHeterogeneous, m_Group, p)) {
                addScalar(p, fieldSuffix,
                          [p, getParamValues](const auto &g, size_t)
                          {
                              return std::invoke(getParamValues, g).at(p);
                          });
            }
            // Otherwise, just add a const-qualified scalar to the type environment
            else {
                add(m_Group.getScalarType().addConst(), p, 
                    getScalarString(std::invoke(getParamValues, m_Group.getArchetype()).at(p)));
            }
        }
    }

    void addDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, const std::string &fieldSuffix,
                          GetParamValuesFn getDerivedParamValues, IsHeterogeneousFn isHeterogeneous)
    {
        // Loop through derived params
        for(const auto &d : derivedParams) {
            // If derived parameter is heterogeneous, add scalar field
            if (std::invoke(isHeterogeneous, m_Group, d.name)) {
                addScalar(d.name, fieldSuffix,
                          [d, getDerivedParamValues](const auto &g, size_t)
                          {
                              return std::invoke(getDerivedParamValues, g).at(d.name);
                          });
            }
            // Otherwise, just add a const-qualified scalar to the type environment with archetype value
            else {
                add(m_Group.getScalarType().addConst(), d.name, 
                    getScalarString(std::invoke(getDerivedParamValues, m_Group).at(d.name)));
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
                // If parameter is heterogeneous, add scalar field
                if(std::invoke(isHeterogeneous, m_Group, v.name, p.first)) {
                    addScalar(p.first, v.name + fieldSuffix,
                              [p, v](const auto &g, size_t)
                              {
                                  return  A(g).getInitialisers().at(v.name).getParams().at(p.first);
                              });
                }
                // Otherwise, just add a const-qualified scalar to the type environment with archetype value
                else {
                    add(m_Group.getScalarType().addConst(), p.first, getScalarString(p.second));
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
                // If derived parameter is heterogeneous, add scalar field
                if(std::invoke(isHeterogeneous, m_Group, v.name, p.first)) {
                    addScalar(p.first, v.name + fieldSuffix,
                              [p, v](const auto &g, size_t)
                              {
                                  return A(g).getInitialisers().at(v.name).getDerivedParams().at(p.first);
                              });
                }
                // Otherwise, just add a const-qualified scalar to the type environment with archetype value
                else {
                    add(m_Group.getScalarType().addConst(), p.first, getScalarString(p.second));
                }
            }
        }
    }

    template<typename A, typename I>
    void addVars(const std::string &arrayPrefix, I getIndexFn, const std::string &fieldSuffix = "",
                 const std::vector<std::string> &dependents = {})
    {
        // Loop through variables
        const A archetypeAdaptor(m_Group.getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            const auto resolvedType = v.type.resolve(m_Group.getTypeContext())
            const auto qualifiedType = (getVarAccessMode(v.access) & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
            add(qualifiedType, v.name,
                resolvedType.createPointer(), v.name + fieldSuffix, 
                [arrayPrefix, getIndexFn, v](const auto &g, size_t) 
                { 
                    return prefix + v.name + A(g).getNameSuffix();
                },
                getIndexFn(v.access, v.name), GroupMergedFieldType::STANDARD, {}, dependents);
        }
    }

    template<typename A>
    void addVars(const std::string &arrayPrefix, const std::string &index, const std::string &fieldSuffix = "",
                 const std::vector<std::string> &dependents = {})
    {
        addVars<A>(arrayPrefix, [&index](VarAccess a, const std::string &) { return index; }, 
                   fieldSuffix, dependents);
    }

    template<typename A, typename I>
    void addVarRefs(const std::string &arrayPrefix, I getIndexFn, const std::string &fieldSuffix = "",
                    const std::vector<std::string> &dependents = {})
    {
        // Loop through variable references
        const A archetypeAdaptor(m_Group.getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // If variable access is read-only, qualify type with const
            const auto resolvedType = v.type.resolve(m_Group.getTypeContext());
            const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
            add(qualifiedType, v.name,
                resolvedType.createPointer(), v.name + fieldSuffix,
                [arrayPrefix, v](const auto &g, size_t) 
                { 
                    const auto varRef = A(g).getInitialisers().at(v.name);
                    return arrayPrefix + varRef.getVar().name + varRef.getTargetName(); 
                },
                getIndexFn(v.access, v.name), GroupMergedFieldType::STANDARD, {}, dependents);
        }
    }

    template<typename A>
    void addVarRefs(const std::string &arrayPrefix, const std::string &index, const std::string &fieldSuffix = "",
                    const std::vector<std::string> &dependents = {})
    {
        addVarRefs<A>(arrayPrefix, [&index](VarAccess a, const std::string &) { return index; }, 
                      fieldSuffix, dependents);
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
                        "", GroupMergedFieldType::DYNAMIC);
        }
    }

private:
    //------------------------------------------------------------------------
    // Private API
    //------------------------------------------------------------------------
    std::string getScalarString(double scalar) const
    {
        return (Utils::writePreciseString(scalar, m_GroupMerged.getScalarType().getNumeric().maxDigits10) 
                + m_GroupMerged.getScalarType().getNumeric().literalSuffix);
    }
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::reference_wrapper<G> m_Group;

    //! Environment mapping names to types to fields to pull values from
    std::unordered_map<std::string, std::tuple<Type::ResolvedType, bool, std::string, std::optional<typename G::Field>>> m_Environment;
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
