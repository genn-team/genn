#pragma once

// Standard C++ includes
#include <algorithm>
#include <string>
#include <vector>

// GeNN includes
#include "snippet.h"
#include "initVarSnippet.h"
#include "varAccess.h"

// Forward declarations
class NeuronGroup;
class CurrentSource;
class SynapseGroup;

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
            return ((name == other.name) && (type == other.type) && (access == other.access));
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
            return ((name == other.name) && (type == other.type) && (access == other.access));
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
    bool canBeMerged(const Base *other) const
    {
        // Return true if vars and egps match
        return (Snippet::Base::canBeMerged(other)
                && (getVars() == other->getVars()));
    }
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
// VarReference
//----------------------------------------------------------------------------
class VarReference
{
public:
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class Type
    {
        Neuron,
        CurrentSource,
        PSM,
        WU,
        WUPre,
        WUPost,
    };

    VarReference(const NeuronGroup *ng, const std::string &varName);
    VarReference(const CurrentSource *cs, const std::string &varName);
    VarReference(const SynapseGroup *sg, const std::string &varName, Type type);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getVarIndex() const { return m_VarIndex; }
    const Base::Var &getVar() const { return m_Var; }
    Type getType() const{ return m_Type; }
    const NeuronGroup *getNeuronGroup() const;
    const SynapseGroup *getSynapseGroup() const;
    const CurrentSource *getCurrentSource() const;

    //------------------------------------------------------------------------
    // Operator
    //------------------------------------------------------------------------
    //! Equality operator returns true if references point to the same variable
    bool operator == (const VarReference &other) const
    {
        return ((m_VarIndex == other.m_VarIndex) && (m_Type == other.m_Type) 
                && (m_NG == other.m_NG) && (m_SG == other.m_SG) && (m_CS == other.m_CS));
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    size_t m_VarIndex;
    Base::Var m_Var;
    const Type m_Type;
    
    const NeuronGroup *m_NG;
    const SynapseGroup *m_SG;
    const CurrentSource *m_CS;
};

//----------------------------------------------------------------------------
// Models::VarReferenceContainerBase
//----------------------------------------------------------------------------
template<size_t NumVars>
using VarReferenceContainerBase = Snippet::InitialiserContainerBase<VarReference, NumVars>;
} // Models
