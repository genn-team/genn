#pragma once

// Standard C++ includes
#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// GeNN includes
#include "initVarSnippet.h"
#include "type.h"
#include "varAccess.h"

// Forward declarations
namespace GeNN
{
class NeuronGroup;
class SynapseGroup;
class CurrentSource;
class CustomUpdate;
class CustomUpdateWU;
class CustomConnectivityUpdate;
class NeuronGroupInternal;
class SynapseGroupInternal;
class CurrentSourceInternal;
class CustomUpdateInternal;
class CustomUpdateWUInternal;
class CustomConnectivityUpdateInternal;
}

namespace GeNN::Runtime
{
class ArrayBase;
class Runtime;
}

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_VARS(...) virtual std::vector<Var> getVars() const override{ return __VA_ARGS__; }
#define DEFINE_REF_DETAIL_STRUCT(NAME, GROUP_TYPE, VAR_TYPE) using NAME = Detail<GROUP_TYPE, VAR_TYPE, struct _##NAME>
#define DEFINE_EGP_REF_DETAIL_STRUCT(NAME, GROUP_TYPE) using NAME = Detail<GROUP_TYPE, struct _##NAME>

//----------------------------------------------------------------------------
// GeNN::Models::Base
//----------------------------------------------------------------------------
//! Base class for all models - in addition to the parameters snippets have, models can have state variables
namespace GeNN::Models
{
class GENN_EXPORT Base : public Snippet::Base
{
public:
    //----------------------------------------------------------------------------
    // Structs
    //----------------------------------------------------------------------------
    //! A variable has a name, a type and an access type
    /*! Explicit constructors required as although, through the wonders of C++
        aggregate initialization, access would default to VarAccess::READ_WRITE
        if not specified, this results in a -Wmissing-field-initializers warning on GCC and Clang*/
    template<typename A>
    struct VarBase
    {
        VarBase(const std::string &n, const Type::ResolvedType &t, A a)
        :   name(n), type(t), access(a)
        {}
        VarBase(const std::string &n, const std::string &t, A a)
        :   name(n), type(t), access(a)
        {}

        using AccessType = A;
        
        bool operator == (const VarBase &other) const
        {
            return (std::tie(name, type, access) == std::tie(other.name, other.type, other.access));
        }

        std::string name;
        Type::UnresolvedType type;
        A access;
    };

    struct Var : public VarBase<VarAccess>
    {
        using VarBase<VarAccess>::VarBase;

        Var(const std::string &n, const Type::ResolvedType &t) 
        :   VarBase(n, t, VarAccess::READ_WRITE)
        {}
        Var(const std::string &n, const std::string &t) 
        :   VarBase(n, t, VarAccess::READ_WRITE)
        {}
    };

    struct CustomUpdateVar : public VarBase<CustomUpdateVarAccess>
    {
        using VarBase<CustomUpdateVarAccess>::VarBase;

        CustomUpdateVar(const std::string &n, const Type::ResolvedType &t) 
        :   VarBase(n, t, CustomUpdateVarAccess::READ_WRITE)
        {}
        CustomUpdateVar(const std::string &n, const std::string &t) 
        :   VarBase(n, t, CustomUpdateVarAccess::READ_WRITE)
        {}
    };

    struct GENN_EXPORT VarRef
    {
        VarRef(const std::string &n, const Type::ResolvedType &t, VarAccessMode a = VarAccessMode::READ_WRITE) : name(n), type(t), access(a)
        {}
        VarRef(const std::string &n, const std::string &t, VarAccessMode a = VarAccessMode::READ_WRITE) : name(n), type(t), access(a)
        {}
        
        bool operator == (const VarRef &other) const
        {
            return (std::tie(name, type, access) == std::tie(other.name, other.type, other.access));
        }

        std::string name;
        Type::UnresolvedType type;
        VarAccessMode access;
    };

    struct GENN_EXPORT EGPRef
    {
        EGPRef(const std::string &n, const Type::ResolvedType &t) : name(n), type(t)
        {}
        EGPRef(const std::string &n, const std::string &t);

        bool operator == (const EGPRef &other) const
        {
            return (std::tie(name, type) == std::tie(other.name, other.type));
        }

        std::string name;
        Type::UnresolvedType type;
    };

    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::vector<VarRef> VarRefVec;
    typedef std::vector<EGPRef> EGPRefVec;
};

//----------------------------------------------------------------------------
// GeNN::Models::VarReferenceBase
//----------------------------------------------------------------------------
class GENN_EXPORT VarReferenceBase
{
protected:
    //------------------------------------------------------------------------
    // Detail
    //------------------------------------------------------------------------
    //! Minimal helper class for definining unique struct 
    //! wrappers around group pointers for use with std::variant
    template<typename G, typename V, typename Tag>
    struct Detail
    {
        G *group;
        V var;
    };
};

//----------------------------------------------------------------------------
// GeNN::Models::VarReference
//----------------------------------------------------------------------------
class GENN_EXPORT VarReference : public VarReferenceBase
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Get type of variable
    const Type::UnresolvedType &getVarType() const;

    // Get dimensions of variable
    VarAccessDim getVarDims() const;

    //! Get size of variable
    unsigned int getSize() const;

    //! If variable is delayed, get neuron group which manages its delay
    NeuronGroup *getDelayNeuronGroup() const;
    
    //! Get array associated with referenced variable
    const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime) const;

    //! If this reference points to another custom update, return pointer to it
    /*! This is used to detect circular dependencies */
    CustomUpdate *getReferencedCustomUpdate() const;

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    bool operator < (const VarReference &other) const;

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    static VarReference createVarRef(NeuronGroup *ng, const std::string &varName);
    static VarReference createVarRef(CurrentSource *cs, const std::string &varName);
    static VarReference createVarRef(CustomUpdate *cu, const std::string &varName);
    static VarReference createPreVarRef(CustomConnectivityUpdate *ccu, const std::string &varName);
    static VarReference createPostVarRef(CustomConnectivityUpdate *ccu, const std::string &varName);
    static VarReference createPSMVarRef(SynapseGroup *sg, const std::string &varName);
    static VarReference createWUPreVarRef(SynapseGroup *sg, const std::string &varName);
    static VarReference createWUPostVarRef(SynapseGroup *sg, const std::string &varName);

private:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    DEFINE_REF_DETAIL_STRUCT(NGRef, NeuronGroupInternal, Base::Var);
    DEFINE_REF_DETAIL_STRUCT(PSMRef, SynapseGroupInternal, Base::Var);
    DEFINE_REF_DETAIL_STRUCT(WUPreRef, SynapseGroupInternal, Base::Var);
    DEFINE_REF_DETAIL_STRUCT(WUPostRef, SynapseGroupInternal, Base::Var);
    DEFINE_REF_DETAIL_STRUCT(CSRef, CurrentSourceInternal, Base::Var);
    DEFINE_REF_DETAIL_STRUCT(CURef, CustomUpdateInternal, Base::CustomUpdateVar);
    DEFINE_REF_DETAIL_STRUCT(CCUPreRef, CustomConnectivityUpdateInternal, Base::Var);
    DEFINE_REF_DETAIL_STRUCT(CCUPostRef, CustomConnectivityUpdateInternal, Base::Var);

    //! Variant type used to store 'detail'
    using DetailType = std::variant<NGRef, PSMRef, WUPreRef, WUPostRef, CSRef, 
                                    CURef, CCUPreRef, CCUPostRef>;

    VarReference(const DetailType &detail) : m_Detail(detail)
    {}

    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    const std::string &getVarName() const;
    const std::string &getTargetName() const;

    //------------------------------------------------------------------------
    // Friends
    //------------------------------------------------------------------------
    friend void updateHash(const VarReference &v, boost::uuids::detail::sha1 &hash)
    {
        Utils::updateHash(v.getTargetName(), hash);
        Utils::updateHash(v.getVarName(), hash);
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    DetailType m_Detail;
};

//----------------------------------------------------------------------------
// GeNN::Models::WUVarReference
//----------------------------------------------------------------------------
class GENN_EXPORT WUVarReference : public VarReferenceBase
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Get type of variable
    const Type::UnresolvedType &getVarType() const;

    // Get dimensions of variable
    VarAccessDim getVarDims() const;
    
    //! Get array associated with referenced variable
    const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime) const;
    
    SynapseGroup *getSynapseGroup() const;
    
    // Get type of transpose variable
    std::optional<Type::UnresolvedType> getTransposeVarType() const;

    //! Get dimensions of transpose variable being referenced
    std::optional<VarAccessDim> getTransposeVarDims() const;

    //! Get array associated with referenced transpose variable
    const Runtime::ArrayBase *getTransposeTargetArray(const Runtime::Runtime &runtime) const;

    SynapseGroup *getTransposeSynapseGroup() const;

    //! If this reference points to another custom update, return pointer to it
    /*! This is used to detect circular dependencies */
    CustomUpdateWU *getReferencedCustomUpdate() const;

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    bool operator < (const WUVarReference &other) const;

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    static WUVarReference createWUVarReference(SynapseGroup *sg, const std::string &varName, 
                                               SynapseGroup *transposeSG = nullptr, const std::string &transposeVarName = "");
    static WUVarReference createWUVarReference(CustomUpdateWU *cu, const std::string &varName);
    static WUVarReference createWUVarReference(CustomConnectivityUpdate *ccu, const std::string &varName);

private:
    //------------------------------------------------------------------------
    // WURef
    //------------------------------------------------------------------------
    //! Struct for storing weight update group variable reference - needs
    //! Additional field to store synapse group associated with transpose
    struct WURef
    {
        SynapseGroupInternal *group;
        SynapseGroupInternal *transposeGroup;

        Base::Var var;
        std::optional<Base::Var> transposeVar;
    };

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    DEFINE_REF_DETAIL_STRUCT(CURef, CustomUpdateWUInternal, Base::CustomUpdateVar);
    DEFINE_REF_DETAIL_STRUCT(CCURef, CustomConnectivityUpdateInternal, Base::Var);

     //! Variant type used to store 'detail'
    using DetailType = std::variant<WURef, CURef, CCURef>;

    WUVarReference(const DetailType &detail);

    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    SynapseGroupInternal *getSynapseGroupInternal() const;
    SynapseGroupInternal *getTransposeSynapseGroupInternal() const;
    const std::string &getVarName() const;
    const std::string &getTargetName() const;
    std::optional<std::string> getTransposeVarName() const;
    std::optional<std::string> getTransposeTargetName() const;

    //------------------------------------------------------------------------
    // Friends
    //------------------------------------------------------------------------
    friend void updateHash(const WUVarReference &v, boost::uuids::detail::sha1 &hash)
    {
        Utils::updateHash(v.getTargetName(), hash);
        Utils::updateHash(v.getVarName(), hash);

        if(v.getTransposeSynapseGroup() != nullptr) {
            Utils::updateHash(v.getTransposeTargetName(), hash);
            Utils::updateHash(v.getTransposeVarName(), hash);
        }
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    DetailType m_Detail;
};

//----------------------------------------------------------------------------
// Models::EGPReference
//----------------------------------------------------------------------------
class GENN_EXPORT EGPReference
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const Models::Base::EGP &getEGP() const;
    
    //! Get array associated with referenced EGP
    // **YUCK** dependency on codegenerator and runtime suggests this belongs elsewhere
    const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime) const;

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    static EGPReference createEGPRef(NeuronGroup *ng, const std::string &egpName);
    static EGPReference createEGPRef(CurrentSource *cs, const std::string &egpName);
    static EGPReference createEGPRef(CustomUpdate *cu, const std::string &egpName);
    static EGPReference createEGPRef(CustomUpdateWU *cu, const std::string &egpName);
    static EGPReference createPSMEGPRef(SynapseGroup *sg, const std::string &egpName);
    static EGPReference createWUEGPRef(SynapseGroup *sg, const std::string &egpName);

private:
    //------------------------------------------------------------------------
    // Detail
    //------------------------------------------------------------------------
    //! Minimal helper class for definining unique struct 
    //! wrappers around group pointers for use with std::variant
    template<typename G, typename Tag>
    struct Detail
    {
        G *group;
        Models::Base::EGP egp;
    };

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    DEFINE_EGP_REF_DETAIL_STRUCT(NGRef, NeuronGroup);
    DEFINE_EGP_REF_DETAIL_STRUCT(CSRef, CurrentSource);
    DEFINE_EGP_REF_DETAIL_STRUCT(CURef, CustomUpdate);
    DEFINE_EGP_REF_DETAIL_STRUCT(CUWURef, CustomUpdateWU);
    DEFINE_EGP_REF_DETAIL_STRUCT(CCURef, CustomConnectivityUpdate);
    DEFINE_EGP_REF_DETAIL_STRUCT(PSMRef, SynapseGroup);
    DEFINE_EGP_REF_DETAIL_STRUCT(WURef, SynapseGroup);

    //! Variant type used to store 'detail'
    using DetailType = std::variant<NGRef, CSRef, CURef, CUWURef, CCURef, PSMRef, WURef>;

    EGPReference(const DetailType &detail) : m_Detail(detail)
    {}

    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    const std::string &getEGPName() const;
    const std::string &getTargetName() const;

    //----------------------------------------------------------------------------
    // Friends
    //----------------------------------------------------------------------------
    friend void updateHash(const EGPReference &v, boost::uuids::detail::sha1 &hash)
    {
        Utils::updateHash(v.getTargetName(), hash);
        Utils::updateHash(v.getEGPName(), hash);
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    DetailType m_Detail;
};

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
GENN_EXPORT void updateHash(const Base::Var &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::CustomUpdateVar &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::VarRef &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::EGPRef &e, boost::uuids::detail::sha1 &hash);

//! Helper function to check if variable reference types match those specified in model
template<typename V>
void checkVarReferenceTypes(const std::unordered_map<std::string, V> &varRefs, const Base::VarRefVec &modelVarRefs)
{
    // Loop through all variable references
    for(const auto &modelVarRef : modelVarRefs) {
        const auto varRef = varRefs.at(modelVarRef.name);

        // Check types of variable references against those specified in model
        // **THINK** this is rather conservative but I think not allowing "scalar" and whatever happens to be scalar type is ok
        if(varRef.getVarType() != modelVarRef.type) {
            throw std::runtime_error("Incompatible type for variable reference '" + modelVarRef.name + "'");
        }
    }
}
} // GeNN::Models
