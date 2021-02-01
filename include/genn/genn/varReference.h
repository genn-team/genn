#pragma once

// Standard C++ includes
#include <functional>
#include <string>

// GeNN includes
#include "models.h"

// Forward declarations
class NeuronGroup;
class SynapseGroup;
class CurrentSource;
class NeuronGroupInternal;
class SynapseGroupInternal;
class CurrentSourceInternal;

//----------------------------------------------------------------------------
// VarReferenceBase
//----------------------------------------------------------------------------
class VarReferenceBase
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const Models::Base::Var &getVar() const { return m_Var; }
    size_t getVarIndex() const { return m_VarIndex; }

protected:
    VarReferenceBase(size_t varIndex, const Models::Base::VarVec &varVec)
    : m_VarIndex(varIndex), m_Var(varVec.at(varIndex))
    {}

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const size_t m_VarIndex;
    const Models::Base::Var m_Var;
};

//----------------------------------------------------------------------------
// VarReference
//----------------------------------------------------------------------------
class VarReference : public VarReferenceBase
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    unsigned int getSize() const { return m_Size; }
    const std::string &getTargetName() const { return m_GetTargetNameFn(); }

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    static VarReference createNeuronVarRef(const NeuronGroup *ng, const std::string &varName);
    static VarReference createCurrentSourceVarRef(const CurrentSource *cs, const std::string &varName);
    static VarReference createPSMVarRef(const SynapseGroup *sg, const std::string &varName);
    static VarReference createWUPreVarRef(const SynapseGroup *sg, const std::string &varName);
    static VarReference createWUPostVarRef(const SynapseGroup *sg, const std::string &varName);
    
private:
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    typedef std::function<const std::string &(void)> GetTargetNameFn;

    VarReference(const NeuronGroupInternal *ng, const std::string &varName);
    VarReference(const CurrentSourceInternal *cs, const std::string &varName);
    VarReference(GetTargetNameFn getTargetNameFn, unsigned int size, 
                 size_t varIndex, const Models::Base::VarVec &varVec);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const unsigned int m_Size;
    GetTargetNameFn m_GetTargetNameFn;
};

//----------------------------------------------------------------------------
// WUVarReference
//----------------------------------------------------------------------------
class WUVarReference : public VarReferenceBase
{
public:
    WUVarReference(const SynapseGroup *sg, const std::string &varName);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const SynapseGroup *getSynapseGroup() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const SynapseGroupInternal *m_SG;
};
