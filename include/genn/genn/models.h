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

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_VARS(...) virtual VarVec getVars() const override{ return __VA_ARGS__; }
#define DEFINE_REF_DETAIL_STRUCT(NAME, GROUP_TYPE) using NAME = Detail<GROUP_TYPE, struct _##NAME>

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
    struct GENN_EXPORT Var
    {
        Var(const std::string &n, const Type::ResolvedType &t)
        :   name(n), type(t)
        {}
        Var(const std::string &n, const std::string &t)
        :   name(n), type(t)
        {}

        Var(const std::string &n, const Type::ResolvedType &t, VarAccess a)
        :   name(n), type(t), access(static_cast<unsigned int>(a))
        {}
        Var(const std::string &n, const std::string &t, VarAccess a)
        :   name(n), type(t), access(static_cast<unsigned int>(a))
        {}

        /*Var(const std::string &n, const Type::ResolvedType &t, NeuronVarAccess a)
        :   name(n), type(t), access(static_cast<unsigned int>(a))
        {}
        Var(const std::string &n, const std::string &t, NeuronVarAccess a)
        :   name(n), type(t), access(static_cast<unsigned int>(a))
        {}*/
        
        bool operator == (const Var &other) const
        {
            return (std::tie(name, type, access) == std::tie(other.name, other.type, other.access));
        }

        unsigned int getAccess(VarAccess defaultAccess) const
        {
            return access.value_or(static_cast<unsigned int>(defaultAccess)); 
        }

        VarAccessMode getAccessMode() const
        {
            if(access) {
                return getVarAccessMode(access.value());
            }
            else {
                return VarAccessMode::READ_WRITE;
            }
        }

        std::string name;
        Type::UnresolvedType type;
        std::optional<unsigned int> access;
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

        VarAccessMode getAccessMode() const
        {
            return access;
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
    typedef std::vector<Var> VarVec;
    typedef std::vector<VarRef> VarRefVec;
    typedef std::vector<EGPRef> EGPRefVec;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Gets model variables
    virtual VarVec getVars() const{ return {}; }

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Find the index of a named variable
    size_t getVarIndex(const std::string &varName) const
    {
        return getNamedVecIndex(varName, getVars());
    }

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void updateHash(boost::uuids::detail::sha1 &hash) const;

    //! Validate names of parameters etc
    void validate(const std::unordered_map<std::string, double> &paramValues, 
                  const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                  const std::string &description) const;
   
};

//----------------------------------------------------------------------------
// GeNN::Models::VarReferenceBase
//----------------------------------------------------------------------------
class GENN_EXPORT VarReferenceBase
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const Models::Base::Var &getVar() const { return m_Var; }
    size_t getVarIndex() const { return m_VarIndex; }
    

protected:
    //------------------------------------------------------------------------
    // Detail
    //------------------------------------------------------------------------
    //! Minimal helper class for definining unique struct 
    //! wrappers around group pointers for use with std::variant
    template<typename G, typename Tag>
    struct Detail
    {
        G *group;
    };
    
    VarReferenceBase(size_t varIndex, const Models::Base::VarVec &varVec)
    : m_VarIndex(varIndex), m_Var(varVec.at(varIndex))
    {}

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    size_t m_VarIndex;
    Models::Base::Var m_Var;
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
    //! Get size of variable
    unsigned int getSize() const { return m_Size; }

    //! If variable is delayed, get neuron group which manages its delay
    NeuronGroup *getDelayNeuronGroup() const;
    
    //! Get suffix to use when accessing target variable names
    // **TODO** rename to getNameSuffix
    std::string getTargetName() const;
    
    //! If model is batched, will the variable this is referencing be duplicated?
    bool isDuplicated() const;

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
    DEFINE_REF_DETAIL_STRUCT(NGRef, NeuronGroupInternal);
    DEFINE_REF_DETAIL_STRUCT(PSMRef, SynapseGroupInternal);
    DEFINE_REF_DETAIL_STRUCT(WUPreRef, SynapseGroupInternal);
    DEFINE_REF_DETAIL_STRUCT(WUPostRef, SynapseGroupInternal);
    DEFINE_REF_DETAIL_STRUCT(CSRef, CurrentSourceInternal);
    DEFINE_REF_DETAIL_STRUCT(CURef, CustomUpdateInternal);
    DEFINE_REF_DETAIL_STRUCT(CCUPreRef, CustomConnectivityUpdateInternal);
    DEFINE_REF_DETAIL_STRUCT(CCUPostRef, CustomConnectivityUpdateInternal);

    //! Variant type used to store 'detail'
    using DetailType = std::variant<NGRef, PSMRef, WUPreRef, WUPostRef, CSRef, 
                                    CURef, CCUPreRef, CCUPostRef>;

    VarReference(size_t varIndex, const Models::Base::VarVec &varVec, unsigned int size,
                 const DetailType &detail)
    :   VarReferenceBase(varIndex, varVec), m_Size(size), m_Detail(detail)
    {}

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    unsigned int m_Size;
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
    const Models::Base::Var &getTransposeVar() const { return *m_TransposeVar; }
    size_t getTransposeVarIndex() const { return *m_TransposeVarIndex; }
    
    //! Get suffix to use when accessing target variable names
    // **TODO** rename to getNameSuffix
    std::string getTargetName() const;
    
    //! If model is batched, will the variable this is referencing be duplicated?
    bool isDuplicated() const;

    SynapseGroup *getSynapseGroup() const;

    SynapseGroup *getTransposeSynapseGroup() const;
    std::string getTransposeTargetName() const;

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
    };

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    DEFINE_REF_DETAIL_STRUCT(CURef, CustomUpdateWUInternal);
    DEFINE_REF_DETAIL_STRUCT(CCURef, CustomConnectivityUpdateInternal);

     //! Variant type used to store 'detail'
    using DetailType = std::variant<WURef, CURef, CCURef>;

    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    SynapseGroupInternal *getSynapseGroupInternal() const;
    SynapseGroupInternal *getTransposeSynapseGroupInternal() const;

    WUVarReference(size_t varIndex, const Models::Base::VarVec &varVec,
                   const DetailType &detail);
    WUVarReference(size_t varIndex, const Models::Base::VarVec &varVec,
                   size_t transposeVarIndex, const Models::Base::VarVec &transposeVarVec,
                   const DetailType &detail);
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::optional<size_t> m_TransposeVarIndex;
    std::optional<Models::Base::Var> m_TransposeVar;
    
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
    const Models::Base::EGP &getEGP() const { return m_EGP; }
    size_t getEGPIndex() const { return m_EGPIndex; }
    std::string getTargetName() const { return m_TargetName; }

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    static EGPReference createEGPRef(const NeuronGroup *ng, const std::string &egpName);
    static EGPReference createEGPRef(const CurrentSource *cs, const std::string &egpName);
    static EGPReference createEGPRef(const CustomUpdate *cu, const std::string &egpName);
    static EGPReference createEGPRef(const CustomUpdateWU *cu, const std::string &egpName);
    static EGPReference createPSMEGPRef(const SynapseGroup *sg, const std::string &egpName);
    static EGPReference createWUEGPRef(const SynapseGroup *sg, const std::string &egpName);

private:
    EGPReference(size_t egpIndex, const Models::Base::EGPVec &egpVec, 
                 const std::string &targetName)
    :   m_EGPIndex(egpIndex), m_EGP(egpVec.at(egpIndex)), m_TargetName(targetName)
    {}
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    size_t m_EGPIndex;
    Models::Base::EGP m_EGP;
    std::string m_TargetName;
};

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
GENN_EXPORT void updateHash(const Base::Var &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::VarRef &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::EGPRef &e, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const VarReference &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const WUVarReference &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const EGPReference &v, boost::uuids::detail::sha1 &hash);

//! Helper function to check if variable reference types match those specified in model
template<typename V>
void checkVarReferences(const std::unordered_map<std::string, V> &varRefs, const Base::VarRefVec &modelVarRefs)
{
    // Loop through all variable references
    for(const auto &modelVarRef : modelVarRefs) {
        const auto varRef = varRefs.at(modelVarRef.name);

        // Check types of variable references against those specified in model
        // **THINK** this is rather conservative but I think not allowing "scalar" and whatever happens to be scalar type is ok
        if(varRef.getVar().type != modelVarRef.type) {
            throw std::runtime_error("Incompatible type for variable reference '" + modelVarRef.name + "'");
        }

        // Check that no reduction targets reference duplicated variables
        // **TODO** default from InitModel class
        if((varRef.getVar().getAccess(VarAccess::READ_WRITE) & VarAccessDuplication::DUPLICATE) 
            && (modelVarRef.access & VarAccessModeAttribute::REDUCE))
        {
            throw std::runtime_error("Reduction target variable reference must be to SHARED or SHARED_NEURON variables.");
        }
    }
}
} // GeNN::Models
