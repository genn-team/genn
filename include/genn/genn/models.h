#pragma once

// Standard C++ includes
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

// GeNN includes
#include "snippet.h"
#include "initVarSnippet.h"
#include "varAccess.h"

// Forward declarations
class CustomConnectivityUpdate;
class CustomUpdate;
class CustomUpdateWU;
class NeuronGroup;
class SynapseGroup;
class CurrentSource;
class NeuronGroupInternal;
class SynapseGroupInternal;
class CurrentSourceInternal;
namespace CodeGenerator
{
class BackendBase;
}

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_MODEL(TYPE, NUM_PARAMS, NUM_VARS)                       \
    DECLARE_SNIPPET(TYPE, NUM_PARAMS);                                  \
    typedef Models::VarInitContainerBase<NUM_VARS> VarValues;        \
    typedef Models::VarInitContainerBase<0> PreVarValues;            \
    typedef Models::VarInitContainerBase<0> PostVarValues

#define IMPLEMENT_MODEL(TYPE) IMPLEMENT_SNIPPET(TYPE)

#define SET_VARS(...) virtual VarVec getVars() const override{ return __VA_ARGS__; }


//----------------------------------------------------------------------------
// Models::Base
//----------------------------------------------------------------------------
//! Base class for all models - in addition to the parameters snippets have, models can have state variables
namespace Models
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
    struct Var
    {
        Var(const std::string &n, const std::string &t, VarAccess a) : name(n), type(t), access(a)
        {}
        Var(const std::string &n, const std::string &t) : Var(n, t, VarAccess::READ_WRITE)
        {}
        Var() : Var("", "", VarAccess::READ_WRITE)
        {}

        bool operator == (const Var &other) const
        {
            return (std::tie(name, type, access) == std::tie(other.name, other.type, other.access));
        }

        std::string name;
        std::string type;
        VarAccess access;
    };

    struct VarRef
    {
        VarRef(const std::string &n, const std::string &t, VarAccessMode a) : name(n), type(t), access(a)
        {}
        VarRef(const std::string &n, const std::string &t) : VarRef(n, t, VarAccessMode::READ_WRITE)
        {}
        VarRef() : VarRef("", "", VarAccessMode::READ_WRITE)
        {}

        bool operator == (const VarRef &other) const
        {
            return (std::tie(name, type, access) == std::tie(other.name, other.type, other.access));
        }

        std::string name;
        std::string type;
        VarAccessMode access;
    };

    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::vector<Var> VarVec;
    typedef std::vector<VarRef> VarRefVec;

    //----------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Gets names and types (as strings) of model variables
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
    void validate() const;
};


//----------------------------------------------------------------------------
// Models::VarInit
//----------------------------------------------------------------------------
//! Class used to bind together everything required to initialise a variable:
//! 1. A pointer to a variable initialisation snippet
//! 2. The parameters required to control the variable initialisation snippet
class VarInit : public Snippet::Init<InitVarSnippet::Base>
{
public:
    VarInit(const InitVarSnippet::Base *snippet, const std::vector<double> &params)
        : Snippet::Init<InitVarSnippet::Base>(snippet, params)
    {
    }

    VarInit(double constant)
        : Snippet::Init<InitVarSnippet::Base>(InitVarSnippet::Constant::getInstance(), {constant})
    {
    }
};

//----------------------------------------------------------------------------
// Models::VarInitContainerBase
//----------------------------------------------------------------------------
template<size_t NumVars>
using VarInitContainerBase = Snippet::InitialiserContainerBase<VarInit, NumVars>;

//----------------------------------------------------------------------------
// Models::VarReferenceBase
//----------------------------------------------------------------------------
class GENN_EXPORT VarReferenceBase
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const Models::Base::Var &getVar() const { return m_Var; }
    size_t getVarIndex() const { return m_VarIndex; }
    std::string getTargetName() const { return m_GetTargetName(); }
    bool isBatched() const{ return m_IsBatched(); }

    bool operator < (const VarReferenceBase &other) const
    {
        // **NOTE** variable and target names are enough to guarantee uniqueness
        const std::string targetName = m_GetTargetName();
        const std::string otherTargetName = other.m_GetTargetName();
        return (std::tie(m_Var.name, targetName) < std::tie(other.m_Var.name, otherTargetName));
    }

protected:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef std::function<std::string(void)> GetTargetNameFn;
    typedef std::function<bool(void)> IsBatchedFn;

    VarReferenceBase(size_t varIndex, const Models::Base::VarVec &varVec, 
                     GetTargetNameFn getTargetName, IsBatchedFn isBatched)
    : m_VarIndex(varIndex), m_Var(varVec.at(varIndex)), m_GetTargetName(getTargetName), m_IsBatched(isBatched)
    {}

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    size_t m_VarIndex;
    Models::Base::Var m_Var;
    GetTargetNameFn m_GetTargetName;
    IsBatchedFn m_IsBatched;
};

//----------------------------------------------------------------------------
// Models::VarReference
//----------------------------------------------------------------------------
class GENN_EXPORT VarReference : public VarReferenceBase
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    unsigned int getSize() const { return m_Size; }
    NeuronGroup *getDelayNeuronGroup() const { return m_GetDelayNeuronGroup(); }

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    static VarReference createVarRef(NeuronGroup *ng, const std::string &varName);
    static VarReference createVarRef(CurrentSource *cs, const std::string &varName);
    static VarReference createVarRef(CustomUpdate *cu, const std::string &varName);
    static VarReference createPreVarRef(CustomConnectivityUpdate *cu, const std::string &varName);
    static VarReference createPostVarRef(CustomConnectivityUpdate *cu, const std::string &varName);
    static VarReference createPSMVarRef(SynapseGroup *sg, const std::string &varName);
    static VarReference createWUPreVarRef(SynapseGroup *sg, const std::string &varName);
    static VarReference createWUPostVarRef(SynapseGroup *sg, const std::string &varName);
    
private:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef std::function<NeuronGroup*(void)> GetDelayNeuronGroupFn;

    VarReference(NeuronGroupInternal *ng, const std::string &varName);
    VarReference(CurrentSourceInternal *cs, const std::string &varName);
    VarReference(CustomUpdate *cu, const std::string &varName);
    VarReference(unsigned int size, GetDelayNeuronGroupFn getDelayNeuronGroup,
                 size_t varIndex, const Models::Base::VarVec &varVec, 
                 GetTargetNameFn getTargetName, IsBatchedFn isBatched);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    unsigned int m_Size;
    GetDelayNeuronGroupFn m_GetDelayNeuronGroup;
};

//----------------------------------------------------------------------------
// Models::VarReferenceContainerBase
//----------------------------------------------------------------------------
template<size_t NumVars>
using VarReferenceContainerBase = Snippet::InitialiserContainerBase<VarReference, NumVars>;

//----------------------------------------------------------------------------
// Models::WUVarReference
//----------------------------------------------------------------------------
class GENN_EXPORT WUVarReference : public VarReferenceBase
{
public:
    WUVarReference(SynapseGroup *sg, const std::string &varName,
                   SynapseGroup *transposeSG = nullptr, const std::string &transposeVarName = "");
    WUVarReference(CustomUpdateWU *cu, const std::string &varName);
    WUVarReference(CustomConnectivityUpdate *cu, const std::string &varName);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    SynapseGroup *getSynapseGroup() const;

    SynapseGroup *getTransposeSynapseGroup() const;
    const Models::Base::Var &getTransposeVar() const { return m_TransposeVar; }
    size_t getTransposeVarIndex() const { return m_TransposeVarIndex; }
    std::string getTransposeTargetName() const { return m_GetTransposeTargetName(); }

    bool operator < (const WUVarReference &other) const
    {
        const bool hasTranspose = (getTransposeSynapseGroup() != nullptr);
        const bool otherHasTranspose = (other.getTransposeSynapseGroup() != nullptr);
        if (hasTranspose && otherHasTranspose) {
            if (other.m_TransposeVar.name < m_TransposeVar.name) {
                return false;
            }
            else if (m_TransposeVar.name < other.m_TransposeVar.name) {
                return true;
            }

            auto transposeTargetName = m_GetTransposeTargetName();
            auto otherTransposeTargetName = other.m_GetTransposeTargetName();
            if (otherTransposeTargetName < transposeTargetName) {
                return false;
            }
            else if (transposeTargetName < otherTransposeTargetName) {
                return true;
            }
        }
        else if (hasTranspose) {
            return false;
        }
        else if (otherHasTranspose) {
            return true;
        }
        
        return (VarReferenceBase::operator < (other));
    }
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    SynapseGroupInternal *m_SG;
    SynapseGroupInternal *m_TransposeSG;
    size_t m_TransposeVarIndex;
    Models::Base::Var m_TransposeVar;
    GetTargetNameFn m_GetTransposeTargetName;
};

//----------------------------------------------------------------------------
// Models::WUVarReferenceContainerBase
//----------------------------------------------------------------------------
template<size_t NumVars>
using WUVarReferenceContainerBase = Snippet::InitialiserContainerBase<WUVarReference, NumVars>;

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
GENN_EXPORT void updateHash(const Base::Var &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const Base::VarRef &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const VarReference &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const WUVarReference &v, boost::uuids::detail::sha1 &hash);


//! Helper function to check if variable reference types match those specified in model
template<typename V>
void checkVarReferences(const std::vector<V> &varRefs, const Base::VarRefVec &modelVarRefs)
{
    // Loop through all variable references
    for(size_t i = 0; i < varRefs.size(); i++) {
        const auto varRef = varRefs.at(i);
        const auto modelVarRef = modelVarRefs.at(i);

        // Check types of variable references against those specified in model
        // **THINK** due to GeNN's current string-based type system this is rather conservative
        if(varRef.getVar().type != modelVarRef.type) {
            throw std::runtime_error("Incompatible type for variable reference '" + modelVarRef.name + "'");
        }

        // Check that no reduction targets reference duplicated variables
        if((varRef.getVar().access & VarAccessDuplication::DUPLICATE) 
            && (modelVarRef.access & VarAccessModeAttribute::REDUCE))
        {
            throw std::runtime_error("Reduction target variable reference must be to SHARED or SHARED_NEURON variables.");
        }
    }
}
} // Models
