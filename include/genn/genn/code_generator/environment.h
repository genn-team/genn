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
#include "code_generator/lazyString.h"

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
    :   m_Context(std::make_pair(&enclosing, &enclosing))
    {
    }

    explicit EnvironmentExternalBase(Transpiler::PrettyPrinter::EnvironmentBase &enclosing)
    :   m_Context(std::make_pair(nullptr, &enclosing))
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
    // Public API
    //------------------------------------------------------------------------
    void print(const std::string &format);
    void printLine(const std::string &format);

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
    std::variant<std::pair<Transpiler::TypeChecker::EnvironmentBase*, Transpiler::PrettyPrinter::EnvironmentBase*>,
                 std::reference_wrapper<CodeStream>> m_Context;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentLibrary
//----------------------------------------------------------------------------
class EnvironmentLibrary : public EnvironmentExternalBase
{
public:
    using Library = std::unordered_multimap<std::string, std::pair<Type::ResolvedType, std::string>>;

    explicit EnvironmentLibrary(EnvironmentExternalBase &enclosing, const Library &library)
    :   EnvironmentExternalBase(enclosing), m_Library(library)
    {}

    explicit EnvironmentLibrary(Transpiler::PrettyPrinter::EnvironmentBase &enclosing, const Library &library)
    :   EnvironmentExternalBase(enclosing), m_Library(library)
    {
    }

    explicit EnvironmentLibrary(CodeStream &os, const Library &library)
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
class EnvironmentSubstitutionPolicy
{
protected:
    using Payload = LazyString;

    std::string getNameInternal(const LazyString &payload)
    {
        return payload.str();
    }

    void setRequired(LazyString&)
    {
    }
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentFieldPolicy
//----------------------------------------------------------------------------
template<typename G, typename F>
class EnvironmentFieldPolicy
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const G &getGroup() const{ return m_Group; }

protected:
    using Payload = std::tuple<bool, LazyString, std::optional<typename G::Field>>;

    EnvironmentFieldPolicy(G &group, F &fieldGroup)
    :   m_Group(group), m_FieldGroup(fieldGroup)
    {
    }

    // **TODO** only enable if G == F
    EnvironmentFieldPolicy(G &group) : EnvironmentFieldPolicy(group, group)
    {
    }

    std::string getNameInternal(const Payload &payload)
    {
        // If a field is specified
        const auto str = std::get<1>(payload).str();
        if(std::get<2>(payload)) {
            // If there is no value specified, access field directly
            if(str.empty()) {
                 return "group->" + std::get<1>(std::get<2>(payload).value());
            }
            // Otherwise, treat value as index
            else {
                return "group->" + std::get<1>(std::get<2>(payload).value()) + "[" + str + "]"; 
            }
        }
        // Otherwise, use value directly
        else {
            assert(!str.empty());
            return str; 
        }
    }

    void setRequired(Payload &payload)
    {
        // If a field is specified but it hasn't already been added
        if (std::get<2>(payload) && !std::get<0>(payload)) {
            // Extract field from payload
            const auto &field = std::get<2>(payload).value();
            const auto &group = getGroup();

            // Add to field group using lambda function to potentially map from group to field
            // **NOTE** this will have been destroyed by the point this is called so need careful capturing!
            m_FieldGroup.get().addField(std::get<0>(field), std::get<1>(field),
                                        [field, &group](const typename F::GroupInternal &, size_t i)
                                        {
                                            return std::get<2>(field)(group.getGroups().at(i), i);
                                        },
                                        std::get<3>(field));

            // Set flag so field doesn't get re-added
            std::get<0>(payload) = true;
        }
    }

private:
    std::reference_wrapper<F> m_FieldGroup;
    std::reference_wrapper<G> m_Group;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentExternalDynamicBase
//----------------------------------------------------------------------------
template<typename P>
class EnvironmentExternalDynamicBase : public EnvironmentExternalBase, public P
{
public:
    template<typename... PolicyArgs>
    EnvironmentExternalDynamicBase(EnvironmentExternalBase &enclosing, PolicyArgs&&... policyArgs)
    :   EnvironmentExternalBase(enclosing), P(std::forward<PolicyArgs>(policyArgs)...), m_Contents(m_ContentsStream)
    {}

    template<typename... PolicyArgs>
    EnvironmentExternalDynamicBase(Transpiler::PrettyPrinter::EnvironmentBase &enclosing, PolicyArgs&&... policyArgs)
    :   EnvironmentExternalBase(enclosing), P(std::forward<PolicyArgs>(policyArgs)...), m_Contents(m_ContentsStream)
    {}

    template<typename... PolicyArgs>
    EnvironmentExternalDynamicBase(CodeStream &os, PolicyArgs&&... policyArgs)
    :   EnvironmentExternalBase(os), P(std::forward<PolicyArgs>(policyArgs)...), m_Contents(m_ContentsStream)
    {}

    ~EnvironmentExternalDynamicBase()
    {
        // Loop through initialisers
        std::vector<std::string> initialiserCode(m_Initialisers.size());

        // Because initialisers may refer to other initialisers, 
        // keep evaluating initialisers until no new ones are founf
        bool anyReferences;
        do {
            // Loop through initialiser
            anyReferences = false;
            for(size_t i = 0; i < m_Initialisers.size(); i++) {
                // If initialiser has been referenced
                auto &initialiser = m_Initialisers[i];
                if (initialiser.first) {
                    // Evaluate lazy string into vector
                    initialiserCode[i] = initialiser.second.str();

                    // Clear referenced flag and set flag to ensure another iteration occurs
                    initialiser.first = false;
                    anyReferences = true;
                }
            }
        } while(anyReferences);

        // Write out generated initialiser code
        // **NOTE** in order
        for(const auto &i : initialiserCode) {
            if(!i.empty()) {
                getContextStream() << i << std::endl;
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
            // If this identifier relies on any initialiser statements, mark these initialisers as required
            for(size_t i : std::get<1>(env->second)) {
                m_Initialisers.at(i).first = true;
            }

            // Perform any type-specific logic to mark this identifier as required
            setRequired(std::get<2>(env->second));

            return getNameInternal(std::get<2>(env->second));
        }
    }

    virtual CodeStream &getStream() final { return m_Contents; }

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

            // Perform any type-specific logic to mark this identifier as required
            setRequired(std::get<2>(env->second));

            // Return type of variables
            return {std::get<0>(env->second)};
        }
    }

    size_t addInitialiser(const std::string &format)
    {
        m_Initialisers.emplace_back(false, LazyString{format, *this});
        return (m_Initialisers.size() - 1);
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    //! Map an identifier to a type (for type-checking), lists of initialisers and a payload 
    void addInternal(const GeNN::Type::ResolvedType &type, const std::string &name, const typename P::Payload &payload,
                     const std::vector<size_t> &initialisers = {})
    {
        if(!m_Environment.try_emplace(name, type, initialisers, payload).second) {
            throw std::runtime_error("Redeclaration of '" + std::string{name} + "'");
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::ostringstream m_ContentsStream;
    CodeStream m_Contents;

    std::unordered_map<std::string, std::tuple<Type::ResolvedType, std::vector<size_t>, typename P::Payload>> m_Environment;
    std::vector<std::pair<bool, LazyString>> m_Initialisers;
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
             const std::vector<size_t> &initialisers = {})
    {
        addInternal(type, name, LazyString{value, *this}, initialisers);
    }
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentGroupMergedField
//----------------------------------------------------------------------------
//! External environment, for substituting 
template<typename G, typename F = G>
class EnvironmentGroupMergedField : public EnvironmentExternalDynamicBase<EnvironmentFieldPolicy<G, F>>
{
    using GroupInternal = typename G::GroupInternal;
    using IsHeterogeneousFn = bool (G::*)(const std::string&) const;
    using IsVarInitHeterogeneousFn = bool (G::*)(const std::string&, const std::string&) const;
    using GetParamValuesFn = const std::unordered_map<std::string, double> &(GroupInternal::*)(void) const;
    using GetVarIndexFn = std::function<std::string(VarAccess, const std::string&)>;

    template<typename A>
    using GetVarRefIndexFn = std::function<std::string(VarAccessMode, const typename A::RefType&)>;

    template<typename I>
    using GetConnectivityFn = const I &(GroupInternal::*)(void) const;

    template<typename V>
    using GetVarReferencesFn = const std::unordered_map<std::string, V> &(GroupInternal::*)(void) const;

public:
    using EnvironmentExternalDynamicBase::EnvironmentExternalDynamicBase;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Map a type and a value to an identifier
    void add(const GeNN::Type::ResolvedType &type, const std::string &name, const std::string &value,
             const std::vector<size_t> &initialisers = {})
    {
        addInternal(type, name, std::make_tuple(false, LazyString{value, *this}, std::nullopt), initialisers);
    }

    //! Map a type (for type-checking) and a group merged field to back it to an identifier
    void addField(const GeNN::Type::ResolvedType &type, const std::string &name,
                  const GeNN::Type::ResolvedType &fieldType, const std::string &fieldName, typename G::GetFieldValueFunc getFieldValue,
                  const std::string &indexSuffix = "", GroupMergedFieldType mergedFieldType = GroupMergedFieldType::STANDARD,
                  const std::vector<size_t> &initialisers = {})
    {
        addInternal(type, name, std::make_tuple(false, LazyString{indexSuffix, *this}, 
                                                std::make_optional(std::make_tuple(fieldType, fieldName, getFieldValue, mergedFieldType))),
                    initialisers);
    }

    //! Map a type (for type-checking) and a group merged field to back it to an identifier
    void addField(const GeNN::Type::ResolvedType &type, const std::string &name, const std::string &fieldName, 
                  typename G::GetFieldValueFunc getFieldValue, const std::string &indexSuffix = "", GroupMergedFieldType mergedFieldType = GroupMergedFieldType::STANDARD,
                  const std::vector<size_t> &initialisers = {})
    {
         addField(type, name, type, fieldName, getFieldValue, indexSuffix, mergedFieldType, initialisers);
    }

    void addScalar(const std::string &name, const std::string &fieldSuffix, typename G::GetFieldDoubleValueFunc getFieldValue)
    {
        // **NOTE** this will have been destroyed by the point this is called so need careful capturing!
        const auto &scalarType = getGroup().getScalarType();
        addField(scalarType.addConst(), name,
                 scalarType, name + fieldSuffix,
                 [getFieldValue, scalarType](const auto &g, size_t i)
                 {
                     return writePreciseLiteral(getFieldValue(g, i), scalarType);
                 });
    }

    void addParams(const Snippet::Base::StringVec &paramNames, const std::string &fieldSuffix, 
                   GetParamValuesFn getParamValues, IsHeterogeneousFn isHeterogeneous)
    {
        // Loop through params
        for(const auto &p : paramNames) {
            // If parameter is heterogeneous, add scalar field
            if (std::invoke(isHeterogeneous, getGroup(), p)) {
                addScalar(p, fieldSuffix,
                          [p, getParamValues](const auto &g, size_t)
                          {
                              return std::invoke(getParamValues, g).at(p);
                          });
            }
            // Otherwise, just add a const-qualified scalar to the type environment
            else {
                add(getGroup().getScalarType().addConst(), p, 
                    writePreciseLiteral(std::invoke(getParamValues, getGroup().getArchetype()).at(p),
                                        getGroup().getScalarType()));
            }
        }
    }

    void addDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, const std::string &fieldSuffix,
                          GetParamValuesFn getDerivedParamValues, IsHeterogeneousFn isHeterogeneous)
    {
        // Loop through derived params
        for(const auto &d : derivedParams) {
            // If derived parameter is heterogeneous, add scalar field
            if (std::invoke(isHeterogeneous, getGroup(), d.name)) {
                addScalar(d.name, fieldSuffix,
                          [d, getDerivedParamValues](const auto &g, size_t)
                          {
                              return std::invoke(getDerivedParamValues, g).at(d.name);
                          });
            }
            // Otherwise, just add a const-qualified scalar to the type environment with archetype value
            else {
                add(getGroup().getScalarType().addConst(), d.name, 
                    writePreciseLiteral(std::invoke(getDerivedParamValues, getGroup().getArchetype()).at(d.name),
                                        getGroup().getScalarType()));
            }
        }
    }

    void addExtraGlobalParams(const Snippet::Base::EGPVec &egps, const std::string &arrayPrefix, 
                              const std::string &varName = "", const std::string &fieldSuffix = "")
    {
        // Loop through EGPs
        for(const auto &e : egps) {
            const auto resolvedType = e.type.resolve(getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            const auto pointerType = resolvedType.createPointer();
            addField(pointerType, e.name,
                     pointerType, e.name + varName + fieldSuffix,
                     [arrayPrefix, e, varName](const auto &g, size_t) 
                     {
                         return arrayPrefix + e.name + varName + g.getName(); 
                     },
                     "", GroupMergedFieldType::DYNAMIC);
        }
    }

    template<typename I>
    void addConnectInitParams(const std::string &fieldSuffix, GetConnectivityFn<I> getConnectivity, 
                              IsHeterogeneousFn isHeterogeneous)
    {
        // Loop through params
        const auto &connectInit = std::invoke(getConnectivity, getGroup().getArchetype());
        const auto *snippet = connectInit.getSnippet();
        for(const auto &p : snippet->getParamNames()) {
            // If parameter is heterogeneous, add scalar field
            if (std::invoke(isHeterogeneous, getGroup(), p)) {
                addScalar(p, fieldSuffix,
                          [p, getConnectivity](const auto &g, size_t)
                          {
                              return std::invoke(getConnectivity, g).getParams().at(p);
                          });
            }
            // Otherwise, just add a const-qualified scalar to the type environment
            else {
                add(getGroup().getScalarType().addConst(), p, 
                    writePreciseLiteral(connectInit.getParams().at(p), getGroup().getScalarType()));
            }
        }
    }

    template<typename I>
    void addConnectInitDerivedParams(const std::string &fieldSuffix,  GetConnectivityFn<I> getConnectivity, 
                                     IsHeterogeneousFn isHeterogeneous)
    {
        // Loop through params
        const auto &connectInit = std::invoke(getConnectivity, getGroup().getArchetype());
        const auto *snippet = connectInit.getSnippet();
        for(const auto &d : snippet->getDerivedParams()) {
            // If parameter is heterogeneous, add scalar field
            if (std::invoke(isHeterogeneous, getGroup(), d.name)) {
                addScalar(d.name, fieldSuffix,
                          [d, getConnectivity](const auto &g, size_t)
                          {
                              return std::invoke(getConnectivity, g).getDerivedParams().at(d.name);
                          });
            }
            // Otherwise, just add a const-qualified scalar to the type environment
            else {
                add(getGroup().getScalarType().addConst(), d.name, 
                    writePreciseLiteral(connectInit.getDerivedParams().at(d.name), getGroup().getScalarType()));
            }
        }
    }

    template<typename A>
    void addVarInitParams(IsVarInitHeterogeneousFn isHeterogeneous, 
                          const std::string &varName, const std::string &fieldSuffix = "")
    {
        // Loop through parameters
        for(const auto &p : A(getGroup().getArchetype()).getInitialisers().at(varName).getParams()) {
            // If parameter is heterogeneous, add scalar field
            if(std::invoke(isHeterogeneous, getGroup(), varName, p.first)) {
                addScalar(p.first, varName + fieldSuffix,
                            [p, varName](const auto &g, size_t)
                            {
                                return  A(g).getInitialisers().at(varName).getParams().at(p.first);
                            });
            }
            // Otherwise, just add a const-qualified scalar to the type environment with archetype value
            else {
                add(getGroup().getScalarType().addConst(), p.first, 
                    writePreciseLiteral(p.second, getGroup().getScalarType()));
            }
        }
    }

    template<typename A>
    void addVarInitDerivedParams(IsVarInitHeterogeneousFn isHeterogeneous, 
                                 const std::string &varName, const std::string &fieldSuffix = "")
    {
        // Loop through derived parameters
        for(const auto &p : A(getGroup().getArchetype()).getInitialisers().at(varName).getDerivedParams()) {
            // If derived parameter is heterogeneous, add scalar field
            if(std::invoke(isHeterogeneous, getGroup(), varName, p.first)) {
                addScalar(p.first, varName + fieldSuffix,
                            [p, varName](const auto &g, size_t)
                            {
                                return A(g).getInitialisers().at(varName).getDerivedParams().at(p.first);
                            });
            }
            // Otherwise, just add a const-qualified scalar to the type environment with archetype value
            else {
                add(getGroup().getScalarType().addConst(), p.first, 
                    writePreciseLiteral(p.second, getGroup().getScalarType()));
            }
        }
    }

    template<typename A>
    void addVars(const std::string &arrayPrefix, GetVarIndexFn getIndexFn, const std::string &fieldSuffix = "")
    {
        // Loop through variables
        const A archetypeAdaptor(getGroup().getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            const auto resolvedType = v.type.resolve(getGroup().getTypeContext());
            const auto qualifiedType = (getVarAccessMode(v.access) & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
            addField(qualifiedType, v.name,
                     resolvedType.createPointer(), v.name + fieldSuffix, 
                     [arrayPrefix, v](const auto &g, size_t) 
                     { 
                         return arrayPrefix + v.name + A(g).getNameSuffix();
                     },
                     getIndexFn(v.access, v.name));
        }
    }

    template<typename A>
    void addVars(const std::string &arrayPrefix, const std::string &indexSuffix, const std::string &fieldSuffix = "")
    {
        addVars<A>(arrayPrefix, [&indexSuffix](VarAccess a, const std::string &) { return indexSuffix; }, 
                   fieldSuffix);
    }

    template<typename A>
    void addVarRefs(const std::string &arrayPrefix, GetVarRefIndexFn<A> getIndexFn, const std::string &fieldSuffix = "")
    {
        // Loop through variable references
        const A archetypeAdaptor(getGroup().getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // If variable access is read-only, qualify type with const
            const auto resolvedType = v.type.resolve(getGroup().getTypeContext());
            const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
            addField(qualifiedType, v.name,
                     resolvedType.createPointer(), v.name + fieldSuffix,
                     [arrayPrefix, v](const auto &g, size_t) 
                     { 
                         const auto varRef = A(g).getInitialisers().at(v.name);
                         return arrayPrefix + varRef.getVar().name + varRef.getTargetName(); 
                     },
                     getIndexFn(v.access, archetypeAdaptor.getInitialisers().at(v.name)));
        }
    }

    template<typename A>
    void addVarRefs(const std::string &arrayPrefix, const std::string &indexSuffix, const std::string &fieldSuffix = "")
    {
        addVarRefs<A>(arrayPrefix, [&indexSuffix](VarAccess a, auto &) { return indexSuffix; }, 
                      fieldSuffix);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //! Environment mapping names to types to fields to pull values from
    std::unordered_map<std::string, std::tuple<Type::ResolvedType, bool, std::string, std::optional<typename G::Field>>> m_Environment;
};

template<typename A, typename G>
class VarCachePolicy
{
public:
    using GroupInternal = typename G::GroupInternal;
    using GetIndexFn = std::function<std::string(const std::string&, VarAccessDuplication)>;

    VarCachePolicy(GetIndexFn getReadIndex, GetIndexFn getWriteIndex)
    :   m_GetReadIndex(getReadIndex), m_GetWriteIndex(getWriteIndex)
    {}

    VarCachePolicy(GetIndexFn getIndex)
    :   m_GetReadIndex(getIndex), m_GetWriteIndex(getIndex)
    {}

    std::string getReadIndex(G &g, const Models::Base::Var &var)
    {
        return m_GetReadIndex(var.name, getVarAccessDuplication(var.access));
    }

    std::string getWriteIndex(G &g, const Models::Base::Var &var)
    {
        return m_GetWriteIndex(var.name, getVarAccessDuplication(var.access));
    }

    static std::string getVarSuffix(const GroupInternal &g, const Models::Base::Var &var)
    {
        return A(g).getNameSuffix();
    }

private:
    GetIndexFn m_GetReadIndex;
    GetIndexFn m_GetWriteIndex;
};

template<typename A, typename G>
class VarRefCachePolicy
{
protected:
    using GroupInternal = typename G::GroupInternal;
    using GetIndexFn = std::function<std::string(const std::string&, const typename A::RefType&)>;
  

    VarRefCachePolicy(GetIndexFn getReadIndex, GetIndexFn getWriteIndex)
    :   m_GetReadIndex(getReadIndex), m_GetWriteIndex(getWriteIndex)
    {}

    VarRefCachePolicy(GetIndexFn getIndex)
    :   m_GetReadIndex(getIndex), m_GetWriteIndex(getIndex)
    {}

    std::string getReadIndex(G &g, const Models::Base::VarRef &var)
    {
        return m_GetReadIndex(var.name, A(g.getArchetype()).getInitialisers().at(var.name));
    }

    std::string getWriteIndex(G &g, const Models::Base::VarRef &var)
    {
        return m_GetWriteIndex(var.name, A(g.getArchetype()).getInitialisers().at(var.name));
    }

    static std::string getVarSuffix(const GroupInternal &g, const Models::Base::VarRef &var)
    {
        return A(g).getInitialisers().at(var.name).getTargetName();
    }


private:
    GetIndexFn m_GetReadIndex;
    GetIndexFn m_GetWriteIndex;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentLocalVarCache
//----------------------------------------------------------------------------
//! Pretty printing environment which caches used variables in local variables
template<typename P, typename A, typename G, typename F = G>
class EnvironmentLocalCacheBase : public EnvironmentExternalBase, public P
{
    //! Type of a single definition
    using Def = typename std::invoke_result_t<decltype(&A::getDefs), A>::value_type;

public:
    template<typename... PolicyArgs>
    EnvironmentLocalCacheBase(G &group, F &fieldGroup, const Type::TypeContext &context, EnvironmentExternalBase &enclosing, 
                              const std::string &arrayPrefix, const std::string &fieldSuffix, const std::string &localPrefix,
                              PolicyArgs&&... policyArgs)
    :   EnvironmentExternalBase(enclosing), P(std::forward<PolicyArgs>(policyArgs)...), m_Group(group), m_FieldGroup(fieldGroup), 
        m_Context(context), m_Contents(m_ContentsStream), m_ArrayPrefix(arrayPrefix), m_FieldSuffix(fieldSuffix), m_LocalPrefix(localPrefix)
    {
        // Copy variables into variables referenced, alongside boolean
        const auto defs = A(m_Group.get().getArchetype()).getDefs();
        std::transform(defs.cbegin(), defs.cend(), std::inserter(m_VariablesReferenced, m_VariablesReferenced.end()),
                       [](const auto &v){ return std::make_pair(v.name, std::make_pair(false, v)); });
    }

    /*template<typename... PolicyArgs>
    EnvironmentLocalCacheBase(G &group, const Type::TypeContext &context, EnvironmentExternalBase &enclosing, 
                              const std::string &arrayPrefix, const std::string &fieldSuffix, const std::string &localPrefix,
                              PolicyArgs&&... policyArgs)
    :   EnvironmentLocalVarCache(group, group, context, enclosing, arrayPrefix, fieldSuffix, localPrefix, std::forward<PolicyArgs>(policyArgs)...)
    {}*/
   

    EnvironmentLocalCacheBase(const EnvironmentLocalCacheBase&) = delete;

    ~EnvironmentLocalCacheBase()
    {
        A archetypeAdapter(m_Group.get().getArchetype());

        // Copy definitions of variables which have been referenced into new vector
        const auto varDefs = archetypeAdapter.getDefs();
        std::vector<Def> referencedDefs;
        std::copy_if(varDefs.cbegin(), varDefs.cend(), std::back_inserter(referencedDefs),
                     [this](const auto &v){ return m_VariablesReferenced.at(v.name).first; });

        // Loop through referenced definitions
        for(const auto &v : referencedDefs) {
            const auto resolvedType = v.type.resolve(m_Context.get());

            // Add field to underlying field group
            const auto &group = m_Group.get();
            const auto &arrayPrefix = m_ArrayPrefix;
            m_FieldGroup.get().addField(resolvedType.createPointer(), v.name + m_FieldSuffix,
                                        [arrayPrefix, v, &group](const typename F::GroupInternal &, size_t i)
                                        {
                                            return arrayPrefix + v.name + getVarSuffix(group.getGroups().at(i), v);
                                        });

            if(v.access & VarAccessMode::READ_ONLY) {
                getContextStream() << "const ";
            }
            getContextStream() << resolvedType.getName() << " " << m_LocalPrefix << v.name;

            // If this isn't a reduction, read value from memory
            // **NOTE** by not initialising these variables for reductions, 
            // compilers SHOULD emit a warning if user code doesn't set it to something
            if(!(v.access & VarAccessModeAttribute::REDUCE)) {
                getContextStream() << " = group->" << v.name << m_FieldSuffix << "[" << printSubs(getReadIndex(m_Group.get(), v), *this) << "]";
            }
            getContextStream() << ";" << std::endl;
        }

        // Write contents to context stream
        getContextStream() << m_ContentsStream.str();

        // Loop through referenced definitions again
        for(const auto &v : referencedDefs) {
            // If variables are read-write
            if(v.access & VarAccessMode::READ_WRITE) {
                getContextStream() << "group->" << v.name << m_FieldSuffix << "[" << printSubs(getWriteIndex(m_Group.get(), v), *this) << "]";
                getContextStream() << " = " << m_LocalPrefix << v.name << ";" << std::endl;
            }
        }
    }

    //------------------------------------------------------------------------
    // TypeChecker::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Type::ResolvedType> getTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler) final
    {
        // If name isn't found in environment
        auto var = m_VariablesReferenced.find(name.lexeme);
        if (var == m_VariablesReferenced.end()) {
            return getContextTypes(name, errorHandler);
        }
        // Otherwise
        else {
            // Set flag to indicate that variable has been referenced
            var->second.first = true;

            // Resolve type, add qualifier if required and return
            const auto resolvedType = var->second.second.type.resolve(m_Context.get());
            const auto qualifiedType = (var->second.second.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
            return {qualifiedType};
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
            var->second.first = true;

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
    std::reference_wrapper<G> m_Group;
    std::reference_wrapper<F> m_FieldGroup;
    std::reference_wrapper<const Type::TypeContext> m_Context;
    std::ostringstream m_ContentsStream;
    CodeStream m_Contents;
    std::string m_ArrayPrefix;
    std::string m_FieldSuffix;
    std::string m_LocalPrefix;
    std::unordered_map<std::string, std::pair<bool, Def>> m_VariablesReferenced;
};

template<typename A, typename G, typename F = G>
using EnvironmentLocalVarCache = EnvironmentLocalCacheBase<VarCachePolicy<A, G>, A, G, F>;

template<typename A, typename G, typename F = G>
using EnvironmentLocalVarRefCache = EnvironmentLocalCacheBase<VarRefCachePolicy<A, G>, A, G, F>;

}   // namespace GeNN::CodeGenerator
