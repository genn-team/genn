#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <unordered_map>
#include <variant>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "varAccess.h"
#include "type.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/groupMerged.h"
#include "code_generator/lazyString.h"

// GeNN runtime includes
#include "runtime/runtime.h"

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
class GENN_EXPORT EnvironmentExternalBase : public Transpiler::PrettyPrinter::EnvironmentBase, public Transpiler::TypeChecker::EnvironmentBase
{
public:
    explicit EnvironmentExternalBase(EnvironmentExternalBase &enclosing)
    :   m_Context{&enclosing, &enclosing, nullptr}
    {
    }

    explicit EnvironmentExternalBase(Transpiler::PrettyPrinter::EnvironmentBase &enclosing)
    :   m_Context{nullptr, &enclosing, nullptr}
    {
    }

    explicit EnvironmentExternalBase(CodeStream &os)
    :   m_Context{nullptr, nullptr, &os}
    {
    }

    EnvironmentExternalBase(EnvironmentExternalBase &enclosing, CodeStream &os)
    :   m_Context{&enclosing, &enclosing, &os}
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
    std::tuple<Transpiler::TypeChecker::EnvironmentBase*, Transpiler::PrettyPrinter::EnvironmentBase*, CodeStream*> m_Context;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentLibrary
//----------------------------------------------------------------------------
class GENN_EXPORT EnvironmentLibrary : public EnvironmentExternalBase
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

    EnvironmentLibrary(EnvironmentExternalBase &enclosing, CodeStream &os, const Library &library)
    :   EnvironmentExternalBase(enclosing, os), m_Library(library)
    {
    }

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
                 return "group->" + std::get<2>(payload).value().name;
            }
            // Otherwise, treat value as index
            else {
                return "group->" + std::get<2>(payload).value().name + "[" + str + "]"; 
            }
        }
        // Otherwise, use value directly
        else {
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
            m_FieldGroup.get().addField(field.type, field.name, 
                                        [field, &group](auto &runtime, const typename F::GroupInternal &, size_t i)
                                        {
                                            return field.getValue(runtime, group.getGroups().at(i), i);
                                        },
                                        field.fieldType);

            // Set flag so field doesn't get re-added
            std::get<0>(payload) = true;
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::reference_wrapper<G> m_Group;
    std::reference_wrapper<F> m_FieldGroup;
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

    template<typename... PolicyArgs>
    EnvironmentExternalDynamicBase(EnvironmentExternalBase &enclosing, CodeStream &os, PolicyArgs&&... policyArgs)
    :   EnvironmentExternalBase(enclosing, os), P(std::forward<PolicyArgs>(policyArgs)...), m_Contents(m_ContentsStream)
    {
    }

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
            this->setRequired(std::get<2>(env->second));

            return this->getNameInternal(std::get<2>(env->second));
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
            this->setRequired(std::get<2>(env->second));

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
template<typename G, typename F>
class EnvironmentGroupMergedField : public EnvironmentExternalDynamicBase<EnvironmentFieldPolicy<G, F>>
{
    using GroupInternal = typename G::GroupInternal;
    using GroupExternal = typename GroupInternal::GroupExternal;
    using NonNumericFieldValue =  std::variant<const Runtime::ArrayBase*,
                                               std::pair<Type::NumericValue, 
                                                         Runtime::MergedDynamicFieldDestinations&>>;
    using GetFieldNonNumericValueFunc = std::function<NonNumericFieldValue(Runtime::Runtime&, const GroupInternal&, size_t)>;
    using GetFieldNumericValueFunc = std::function<Type::NumericValue(const GroupInternal&, size_t)>;
    using IsDynamicFn = bool (GroupInternal::*)(const std::string&) const;
    using IsVarInitHeterogeneousFn = bool (G::*)(const std::string&, const std::string&) const;
    using GetParamValuesFn = const std::map<std::string, Type::NumericValue> &(GroupInternal::*)(void) const;

    template<typename A>
    using AdapterDef = typename std::invoke_result_t<decltype(&A::getDefs), A>::value_type;

    template<typename A>
    using GetVarIndexFn = std::function<std::string(typename AdapterDef<A>::AccessType, const std::string&)>;

    template<typename A>
    using GetVarRefIndexFn = std::function<std::string(VarAccessMode, const typename A::RefType&)>;

    template<typename I>
    using GetInitialiserFn = const I &(GroupExternal::*)(void) const;

public:
    using EnvironmentExternalDynamicBase<EnvironmentFieldPolicy<G, F>>::EnvironmentExternalDynamicBase;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Map a type and a value to an identifier
    void add(const GeNN::Type::ResolvedType &type, const std::string &name, const std::string &value,
             const std::vector<size_t> &initialisers = {})
    {
        this->addInternal(type, name, std::make_tuple(false, LazyString{value, *this}, std::nullopt), initialisers);
    }

    //! Map a type (for type-checking) and a group merged field to back it to an identifier
    void addField(const GeNN::Type::ResolvedType &type, const std::string &name,
                  const GeNN::Type::ResolvedType &fieldType, const std::string &fieldName, 
                  GetFieldNumericValueFunc getFieldValue)
    {
        // Get value of field for archetype group
        const auto archetypeValue = getFieldValue(this->getGroup().getArchetype(), 0);

        // Determine if this is heterogeneous across groups
        bool heterogeneous = false;
        for(size_t i = 0; i < this->getGroup().getGroups().size(); i++) {
            if(getFieldValue(this->getGroup().getGroups()[i], i) != archetypeValue) {
                heterogeneous = true;
                break;
            }
        }

        // If type isn't const or values are heterogeneous, add field
        if(!type.isConst || heterogeneous) {
            typename G::Field field{fieldName, fieldType, GroupMergedFieldType::STANDARD,
                                    [getFieldValue](Runtime::Runtime&, const auto &g, size_t i){ return getFieldValue(g, i); }};
            this->addInternal(type, name, std::make_tuple(false, LazyString{"", *this}, std::make_optional(field)));
        }
        // Otherwise, just add value
        else {
            this->add(type, name, Type::writeNumeric(archetypeValue, type));
        }
    }

    //! Map a type (for type-checking) and a group merged field to back it to an identifier
    void addField(const GeNN::Type::ResolvedType &type, const std::string &name,
                  const GeNN::Type::ResolvedType &fieldType, const std::string &fieldName, 
                  GetFieldNonNumericValueFunc getFieldValue, const std::string &indexSuffix = "", 
                  GroupMergedFieldType mergedFieldType = GroupMergedFieldType::STANDARD, 
                  const std::vector<size_t> &initialisers = {})
    {
        typename G::Field field{fieldName, fieldType, mergedFieldType,
                                [getFieldValue](Runtime::Runtime &r, const GroupInternal &g, size_t i)
                                {
                                    return std::visit(
                                        [](const auto &res)->typename G::FieldValue { return res; },
                                        getFieldValue(r, g, i));
                                }};
        this->addInternal(type, name, std::make_tuple(false, LazyString{indexSuffix, *this}, std::make_optional(field)),
                          initialisers);
    }

    //! Map a type (for type-checking) and a group merged field to back it to an identifier
    void addField(const GeNN::Type::ResolvedType &type, const std::string &name, const std::string &fieldName, 
                  GetFieldNumericValueFunc getFieldValue)
    {
         addField(type, name, type, fieldName, getFieldValue);
    }

    //! Map a type (for type-checking) and a group merged field to back it to an identifier
    void addField(const GeNN::Type::ResolvedType &type, const std::string &name, const std::string &fieldName, 
                  GetFieldNonNumericValueFunc getFieldValue, const std::string &indexSuffix = "", 
                  GroupMergedFieldType mergedFieldType = GroupMergedFieldType::STANDARD, const std::vector<size_t> &initialisers = {})
    {
         addField(type, name, type, fieldName, getFieldValue, indexSuffix, mergedFieldType, initialisers);
    }

    void addParams(const Snippet::Base::ParamVec &params, const std::string &fieldSuffix, 
                   GetParamValuesFn getParamValues, IsDynamicFn isDynamic)
    {
        // Loop through params
        for(const auto &p : params) {
            // If parameter is dynamic
            const auto resolvedType = p.type.resolve(this->getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            if(std::invoke(isDynamic, this->getGroup().getArchetype(), p.name)) {
                addField(resolvedType.addConst(), p.name, resolvedType, p.name + fieldSuffix,
                         [p, getParamValues](auto &runtime, const auto &g, size_t)
                         {
                             return std::make_pair(std::invoke(getParamValues, g).at(p.name), 
                                                   std::ref(runtime.getMergedParamDestinations(g, p.name)));
                         },
                         "", GroupMergedFieldType::DYNAMIC);
            }
            // Otherwise, add field
            else {
                addField(resolvedType.addConst(), p.name, resolvedType, p.name + fieldSuffix,
                         [p, getParamValues](const auto &g, size_t)
                         {
                             return std::invoke(getParamValues, g).at(p.name);
                         });
            }
        }
    }

    void addDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, const std::string &fieldSuffix,
                          GetParamValuesFn getDerivedParamValues)
    {
        // Loop through derived params and add scalar fields
        for(const auto &d : derivedParams) {
            const auto resolvedType = d.type.resolve(this->getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            addField(resolvedType.addConst(), d.name, resolvedType, d.name + fieldSuffix,
                     [d, getDerivedParamValues](const auto &g, size_t)
                     {
                         return std::invoke(getDerivedParamValues, g).at(d.name);
                     });
        }
    }

    void addExtraGlobalParams(const Snippet::Base::EGPVec &egps, const std::string &arraySuffix = "", 
                              const std::string &fieldSuffix = "", bool hidden = false)
    {
        // Loop through EGPs
        for(const auto &e : egps) {
            const auto name = hidden ? ("_" + e.name) : e.name;
            const auto resolvedType = e.type.resolve(this->getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            const auto pointerType = resolvedType.createPointer();
            addField(pointerType, name,
                     pointerType, e.name + arraySuffix + fieldSuffix,
                     [e, arraySuffix](auto &runtime, const auto &g, size_t) 
                     {
                         return runtime.getArray(g, e.name + arraySuffix); 
                     },
                     "", GroupMergedFieldType::DYNAMIC);
        }
    }

    void addExtraGlobalParamExposeAliases(const Snippet::Base::EGPVec &egps)
    {
        // Loop through variables and add unhiding aliases
        for(const auto &e : egps) {
            const auto resolvedType = e.type.resolve(this->getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            const auto pointerType = resolvedType.createPointer();
            add(pointerType, e.name, "$(_" + e.name + ")");
        }
    }

    void addExtraGlobalParamRefs(const Models::Base::EGPRefVec &egpRefs, const std::string &fieldSuffix = "")
    {
        // Loop through EGP references
        for(const auto &e : egpRefs) {
            const auto resolvedType = e.type.resolve(this->getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            const auto pointerType = resolvedType.createPointer();
            addField(pointerType, e.name,
                     pointerType, e.name + fieldSuffix,
                     [e](auto &runtime, const auto &g, size_t) 
                     {
                         return g.getEGPReferences().at(e.name).getTargetArray(runtime);
                     },
                     "", GroupMergedFieldType::DYNAMIC);
        }
    }

    template<typename I>
    void addInitialiserParams(const std::string &fieldSuffix, GetInitialiserFn<I> getInitialiser, 
                              std::optional<IsDynamicFn> isDynamic = std::nullopt)
    {
        // Loop through params
        const auto &initialiser = std::invoke(getInitialiser, this->getGroup().getArchetype());
        const auto *snippet = initialiser.getSnippet();
        for(const auto &p : snippet->getParams()) {
            // If parameter is dynamic
            const auto resolvedType = p.type.resolve(this->getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            if(isDynamic && std::invoke(*isDynamic, this->getGroup().getArchetype(), p.name)) {
                addField(resolvedType.addConst(), p.name,
                         resolvedType, p.name + fieldSuffix,
                         [p, getInitialiser](auto &runtime, const auto &g, size_t)
                         {
                             return std::make_pair(std::invoke(getInitialiser, g).getParams().at(p.name), 
                                                   std::ref(runtime.getMergedParamDestinations(g, p.name)));
                         },
                         "", GroupMergedFieldType::DYNAMIC);
            }
            // Otherwise, add standard field
            else {
                addField(resolvedType.addConst(), p.name,
                         resolvedType, p.name + fieldSuffix,
                         [p, getInitialiser](const auto &g, size_t)
                         {
                             return std::invoke(getInitialiser, g).getParams().at(p.name);
                         });
            }
        }
    }

    template<typename I>
    void addInitialiserDerivedParams(const std::string &fieldSuffix,  GetInitialiserFn<I> getInitialiser)
    {
        // Loop through params
        const auto &initialiser = std::invoke(getInitialiser, this->getGroup().getArchetype());
        const auto *snippet = initialiser.getSnippet();
        for(const auto &d : snippet->getDerivedParams()) {
            // If parameter is heterogeneous, add scalar field
            const auto resolvedType = d.type.resolve(this->getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            addField(resolvedType.addConst(), d.name, resolvedType, d.name + fieldSuffix,
                        [d, getInitialiser](const auto &g, size_t)
                        {
                            return std::invoke(getInitialiser, g).getDerivedParams().at(d.name);
                        });
        }
    }

    template<typename A>
    void addVarInitParams(const std::string &varName, const std::string &fieldSuffix = "")
    {
        // Loop through parameters
        const auto &initialiser = A(this->getGroup().getArchetype()).getInitialisers().at(varName);
        const auto *snippet = initialiser.getSnippet();
        for(const auto &p : snippet->getParams()) {
            // If parameter is heterogeneous, add field
            const auto resolvedType = p.type.resolve(this->getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            addField(resolvedType.addConst(), p.name, resolvedType, p.name + varName + fieldSuffix,
                        [p, varName](const auto &g, size_t)
                        {
                            return A(g).getInitialisers().at(varName).getParams().at(p.name);
                        });
        }
    }

    template<typename A>
    void addVarInitDerivedParams(const std::string &varName, const std::string &fieldSuffix = "")
    {
        // Loop through derived parameters
        const auto &initialiser = A(this->getGroup().getArchetype()).getInitialisers().at(varName);
        const auto *snippet = initialiser.getSnippet();
        for(const auto &d : snippet->getDerivedParams()) {
            // If derived parameter is heterogeneous, add scalar field
            const auto resolvedType = d.type.resolve(this->getGroup().getTypeContext());
            assert(!resolvedType.isPointer());
            addField(resolvedType.addConst(), d.name, resolvedType, d.name + varName + fieldSuffix,
                        [d, varName](const auto &g, size_t)
                        {
                            return A(g).getInitialisers().at(varName).getDerivedParams().at(d.name);
                        });
        }
    }

    template<typename A>
    void addVars(GetVarIndexFn<A> getIndexFn, const std::string &fieldSuffix = "",
                 bool readOnly = false, bool hidden = false)
    {
        // Loop through variables
        // Loop through variables
        // **TODO** get default access from adaptor
        const A archetypeAdaptor(this->getGroup().getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            const auto resolvedType = v.type.resolve(this->getGroup().getTypeContext());
            const auto qualifiedType = (readOnly || (getVarAccessMode(v.access) & VarAccessModeAttribute::READ_ONLY)) ? resolvedType.addConst() : resolvedType;
            const auto name = hidden ? ("_" + v.name) : v.name;
            addField(qualifiedType, name,
                     resolvedType.createPointer(), v.name + fieldSuffix, 
                     [v](auto &runtime, const auto &g, size_t) 
                     { 
                         return runtime.getArray(A(g).getTarget(), v.name);
                     },
                     getIndexFn(v.access, v.name));
        }
    }

    template<typename A>
    void addVars(const std::string &indexSuffix, const std::string &fieldSuffix = "",
                 bool readOnly = false, bool hidden = false)
    {
        addVars<A>([&indexSuffix](typename AdapterDef<A>::AccessType, const std::string &) { return indexSuffix; }, 
                   fieldSuffix, readOnly, hidden);
    }

    template<typename A>
    void addVarRefs(GetVarRefIndexFn<A> getIndexFn,  const std::string &fieldSuffix = "",
                    bool readOnly = false, bool hidden = false)
    {
        // Loop through variable references
        const A archetypeAdaptor(this->getGroup().getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // If variable access is read-only, qualify type with const
            const auto resolvedType = v.type.resolve(this->getGroup().getTypeContext());
            const auto qualifiedType = (readOnly || (v.access & VarAccessModeAttribute::READ_ONLY)) ? resolvedType.addConst() : resolvedType;
            const auto name = hidden ? ("_" + v.name) : v.name;
            addField(qualifiedType, name,
                     resolvedType.createPointer(), v.name + fieldSuffix,
                     [v](auto &runtime, const auto &g, size_t) 
                     { 
                         return A(g).getInitialisers().at(v.name).getTargetArray(runtime); 
                     },
                     getIndexFn(v.access, archetypeAdaptor.getInitialisers().at(v.name)));
        }
    }

    template<typename A>
    void addVarRefs(const std::string &indexSuffix, const std::string &fieldSuffix = "",
                    bool readOnly = false, bool hidden = false)
    {
        addVarRefs<A>([&indexSuffix](VarAccess, auto&) { return indexSuffix; }, 
                      fieldSuffix, readOnly, hidden);
    }

    template<typename A>
    void addVarPointers(const std::string &fieldSuffix = "", bool hidden = false)
    {
        // Loop through variables and add private pointer field 
        const A archetypeAdaptor(this->getGroup().getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            const auto resolvedType = v.type.resolve(this->getGroup().getTypeContext());
            const auto name = hidden ? ("_" + v.name) : v.name;
            addField(resolvedType.createPointer(), name, v.name + fieldSuffix,
                     [v](auto &runtime, const auto &g, size_t) 
                     { 
                         return runtime.getArray(g, v.name);
                     });
        }
    }

    template<typename A>
    void addVarRefPointers(bool hidden = false)
    {
        // Loop through variable references and add private pointer field 
        const A archetypeAdaptor(this->getGroup().getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            const auto resolvedType = v.type.resolve(this->getGroup().getTypeContext());
            const auto name = hidden ? ("_" + v.name) : v.name;
            addField(resolvedType.createPointer(), name, v.name,
                     [v](auto &runtime, const auto &g, size_t) 
                     { 
                         return A(g).getInitialisers().at(v.name).getTargetArray(runtime);
                     });
        }
    }

    template<typename A>
    void addVarExposeAliases(bool readOnly = false)
    {
        // Loop through variables and add unhiding aliases
        const A archetypeAdaptor(this->getGroup().getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            const auto resolvedType = v.type.resolve(this->getGroup().getTypeContext());
            const auto qualifiedType = (readOnly || (getVarAccessMode(v.access) & VarAccessModeAttribute::READ_ONLY)) ? resolvedType.addConst() : resolvedType;
            add(qualifiedType, v.name, "$(_" + v.name + ")");
        }
    }


    template<typename A>
    void addLocalVarRefs(bool readOnly = false)
    {
        // Loop through variable references
        const A archetypeAdaptor(this->getGroup().getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // If variable access is read-only, qualify type with const
            const auto resolvedType = v.type.resolve(this->getGroup().getTypeContext());
            const auto qualifiedType = (readOnly || (v.access & VarAccessModeAttribute::READ_ONLY)) ? resolvedType.addConst() : resolvedType;        

            // Get ARCHETYPE variable reference
            // **NOTE** this means all local variable references
            // in merged group must point to same variable
            const auto varRef = archetypeAdaptor.getInitialisers().at(v.name);

            // Add alias from variable reference name to hidden variable name
            add(qualifiedType, v.name, "$(_" + varRef.getVarName() + ")");
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //! Environment mapping names to types to fields to pull values from
    std::unordered_map<std::string, std::tuple<Type::ResolvedType, bool, std::string, std::optional<typename G::Field>>> m_Environment;
};


//------------------------------------------------------------------------
// GeNN::CodeGenerator::VarCachePolicy
//------------------------------------------------------------------------
//! Policy for use with EnvironmentLocalCacheBase for caching state variables
template<typename A, typename G>
class VarCachePolicy
{
public:
    using GroupInternal = typename G::GroupInternal;
    using AdapterDef = typename std::invoke_result_t<decltype(&A::getDefs), A>::value_type;
    using AdapterAccess = typename AdapterDef::AccessType;
    using GetIndexFn = std::function<std::string(const std::string&, AdapterAccess, bool)>;

    VarCachePolicy(GetIndexFn getReadIndex, GetIndexFn getWriteIndex,
                   bool alwaysCopyIfDelayed = true)
    :   m_GetReadIndex(getReadIndex), m_GetWriteIndex(getWriteIndex), 
        m_AlwaysCopyIfDelayed(alwaysCopyIfDelayed)
    {}

    VarCachePolicy(GetIndexFn getIndex, bool alwaysCopyIfDelayed = true)
    :   m_GetReadIndex(getIndex), m_GetWriteIndex(getIndex),
        m_AlwaysCopyIfDelayed(alwaysCopyIfDelayed)
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool shouldAlwaysCopy(G &group, const AdapterDef &var) const
    {
        return A(group.getArchetype()).getNumVarDelaySlots(var.name).has_value() && m_AlwaysCopyIfDelayed;
    }

    std::string getReadIndex(G &group, const AdapterDef &var, const std::string &delaySlot) const
    {
        assert(delaySlot.empty());
        return m_GetReadIndex(var.name, var.access, A(group.getArchetype()).getNumVarDelaySlots(var.name).has_value());
    }

    std::string getWriteIndex(G &group, const AdapterDef &var, const std::string &delaySlot) const
    {
        assert(delaySlot.empty());
        return m_GetWriteIndex(var.name, var.access, A(group.getArchetype()).getNumVarDelaySlots(var.name).has_value());
    }

    const Runtime::ArrayBase *getArray(const Runtime::Runtime &runtime, const GroupInternal &g, const AdapterDef &var) const
    {
        return runtime.getArray(A(g).getTarget(), var.name);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    GetIndexFn m_GetReadIndex;
    GetIndexFn m_GetWriteIndex;
    bool m_AlwaysCopyIfDelayed;
};

//------------------------------------------------------------------------
// GeNN::CodeGenerator::VarRefCachePolicy
//------------------------------------------------------------------------
//! Policy for use with EnvironmentLocalCacheBase for caching variable references
template<typename A, typename G>
class VarRefCachePolicy
{
protected:
    using GroupInternal = typename G::GroupInternal;
    using AdapterDef = typename std::invoke_result_t<decltype(&A::getDefs), A>::value_type;
    using GetIndexFn = std::function<std::string(const std::string&, const typename A::RefType&, const std::string&)>;

    VarRefCachePolicy(GetIndexFn getReadIndex, GetIndexFn getWriteIndex)
    :   m_GetReadIndex(getReadIndex), m_GetWriteIndex(getWriteIndex)
    {}

    VarRefCachePolicy(GetIndexFn getIndex)
    :   m_GetReadIndex(getIndex), m_GetWriteIndex(getIndex)
    {}
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool shouldAlwaysCopy(G&, const AdapterDef&) const
    {
        // **NOTE** something else is managing the actual variables
        // and is therefore responsible for copying between delay slots etc
        return false;
    }
    
    std::string getReadIndex(G &g, const AdapterDef &var, const std::string &delaySlot) const
    {
        assert(delaySlot.empty());
        return m_GetReadIndex(var.name, A(g.getArchetype()).getInitialisers().at(var.name),
                              delaySlot);
    }

    std::string getWriteIndex(G &g, const AdapterDef &var, const std::string &delaySlot) const
    {
        return m_GetWriteIndex(var.name, A(g.getArchetype()).getInitialisers().at(var.name),
                               delaySlot);
    }

    const Runtime::ArrayBase *getArray(const Runtime::Runtime &runtime, const GroupInternal &g, const AdapterDef &var) const
    {
        return A(g).getInitialisers().at(var.name).getTargetArray(runtime);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
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
public:
    using AdapterDef = typename std::invoke_result_t<decltype(&A::getDefs), A>::value_type;

    template<typename... PolicyArgs>
    EnvironmentLocalCacheBase(G &group, F &fieldGroup, const Type::TypeContext &context, EnvironmentExternalBase &enclosing, 
                              const std::string &fieldSuffix, const std::string &localPrefix,
                              bool hidden, PolicyArgs&&... policyArgs)
    :   EnvironmentExternalBase(enclosing), P(std::forward<PolicyArgs>(policyArgs)...), m_Group(group), 
        m_FieldGroup(fieldGroup),  m_Context(context), m_Contents(m_ContentsStream), 
        m_FieldSuffix(fieldSuffix), m_LocalPrefix(localPrefix)
    {
        // Copy variables into variables referenced, alongside boolean
        const auto defs = A(m_Group.get().getArchetype()).getDefs();
        std::transform(defs.cbegin(), defs.cend(), std::inserter(m_VariablesReferenced, m_VariablesReferenced.end()),
                       [hidden](const auto &v){ return std::make_pair(hidden ? "_" + v.name : v.name, 
                                                                      std::make_pair(false, v)); });
    }

    EnvironmentLocalCacheBase(const EnvironmentLocalCacheBase&) = delete;

    ~EnvironmentLocalCacheBase()
    {
        A archetypeAdapter(m_Group.get().getArchetype());

        // Copy definitions of variables which have been referenced into new vector or all if always copy set
        std::vector<AdapterDef> referencedDefs;
        for(const auto &v : m_VariablesReferenced) {
            const bool alwaysCopy = this->shouldAlwaysCopy(m_Group.get(), v.second.second); 
            if(alwaysCopy || v.second.first) {
                referencedDefs.push_back(v.second.second);
            }
        }

        // Loop through referenced definitions
        for(const auto &v : referencedDefs) {
            const auto resolvedType = v.type.resolve(m_Context.get());

            // Add field to underlying field group
            const auto &group = m_Group.get();
            m_FieldGroup.get().addField(resolvedType.createPointer(), v.name + m_FieldSuffix,
                                        [v, &group, this](auto &runtime, const typename F::GroupInternal &, size_t i)
                                        {
                                            return this->getArray(runtime, group.getGroups().at(i), v);
                                        });

            if(getVarAccessMode(v.access) == VarAccessMode::READ_ONLY) {
                getContextStream() << "const ";
            }
            getContextStream() << resolvedType.getName() << " _" << m_LocalPrefix << v.name;

            // If this isn't a reduction or broadcast, read value from memory
            // **NOTE** by not initialising these variables for reductions, 
            // compilers SHOULD emit a warning if user code doesn't set it to something
            if(!(v.access & VarAccessModeAttribute::REDUCE) && !(v.access & VarAccessModeAttribute::BROADCAST)) {
                getContextStream() << " = group->" << v.name << m_FieldSuffix << "[" << printSubs(this->getReadIndex(m_Group.get(), v, ""), *this) << "]";
            }
            getContextStream() << ";" << std::endl;
        }

        // Write contents to context stream
        getContextStream() << m_ContentsStream.str();

        // Loop through referenced definitions again
        for(const auto &v : referencedDefs) {
            // If writes to this variable should be broadcast
            const auto numVarDelaySlots = archetypeAdapter.getNumVarDelaySlots(v.name);
            if(numVarDelaySlots && (v.access & VarAccessModeAttribute::BROADCAST)) {
                getContextStream() << "for(int d = 0; d < " << numVarDelaySlots.value() << "; d++)";
                {
                    CodeStream::Scope b(getContextStream());
                    getContextStream() << "group->" << v.name << m_FieldSuffix << "[" << printSubs(this->getWriteIndex(m_Group.get(), v, "d"), *this) << "]";
                    getContextStream() << " = _" << m_LocalPrefix << v.name << ";" << std::endl;
                }
            }
            // Otherwise, if we should always copy variable, variable is read-write or variable is broadcast
            else if(this->shouldAlwaysCopy(m_Group.get(), v) || (getVarAccessMode(v.access) == VarAccessMode::READ_WRITE)
                    || (v.access & VarAccessModeAttribute::BROADCAST)) 
            {
                getContextStream() << "group->" << v.name << m_FieldSuffix << "[" << printSubs(this->getWriteIndex(m_Group.get(), v, ""), *this) << "]";
                getContextStream() << " = _" << m_LocalPrefix << v.name << ";" << std::endl;
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
            const auto access = var->second.second.access;
            if(access & VarAccessModeAttribute::READ_ONLY) {
                return {resolvedType.addConst()};
            }
            else if((access & VarAccessModeAttribute::REDUCE) || (access & VarAccessModeAttribute::BROADCAST)) {
                return {resolvedType.addWriteOnly()};
            }
            else {
                return {resolvedType};
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
            var->second.first = true;

            // Add underscore and local prefix to variable name
            // **NOTE** we use variable name here not, 'name' which could have an underscore
            return "_" + m_LocalPrefix + var->second.second.name;
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
    std::string m_FieldSuffix;
    std::string m_LocalPrefix;
    std::unordered_map<std::string, std::pair<bool, AdapterDef>> m_VariablesReferenced;
};

template<typename A, typename G, typename F = G>
using EnvironmentLocalVarCache = EnvironmentLocalCacheBase<VarCachePolicy<A, G>, A, G, F>;

template<typename A, typename G, typename F = G>
using EnvironmentLocalVarRefCache = EnvironmentLocalCacheBase<VarRefCachePolicy<A, G>, A, G, F>;
}   // namespace GeNN::CodeGenerator
