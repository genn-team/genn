#pragma once

// Standard C++ includes
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
    VarReferenceBase() = default;

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    void setVar(size_t varIndex, const Models::Base::VarVec &varVec)
    {
        m_VarIndex = varIndex;
        m_Var = varVec.at(varIndex);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    size_t m_VarIndex;
    Models::Base::Var m_Var;
};

//----------------------------------------------------------------------------
// NeuronVarReference
//----------------------------------------------------------------------------
class NeuronVarReference : public VarReferenceBase
{
public:
    NeuronVarReference(const NeuronGroup *ng, const std::string &varName);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    unsigned int getSize() const;
    const NeuronGroup *getNeuronGroup() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const NeuronGroupInternal *m_NG;
};

//----------------------------------------------------------------------------
// CurrentSourceVarReference
//----------------------------------------------------------------------------
class CurrentSourceVarReference : public VarReferenceBase
{
public:
    CurrentSourceVarReference(const CurrentSource *cs, const std::string &varName);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    unsigned int getSize() const;
    const CurrentSource *getCurrentSource() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const CurrentSourceInternal *m_CS;
};

//----------------------------------------------------------------------------
// PSMVarReference
//----------------------------------------------------------------------------
class PSMVarReference : public VarReferenceBase
{
public:
    PSMVarReference(const SynapseGroup *sg, const std::string &varName);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    unsigned int getSize() const;
    const SynapseGroup *getSynapseGroup() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const SynapseGroupInternal *m_SG;
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
    unsigned int getPreSize() const;
    unsigned int getMaxRowLength() const;
    const SynapseGroup *getSynapseGroup() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const SynapseGroupInternal *m_SG;
};


//----------------------------------------------------------------------------
// WUPreVarReference
//----------------------------------------------------------------------------
class WUPreVarReference : public VarReferenceBase
{
public:
    WUPreVarReference(const SynapseGroup *sg, const std::string &varName);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getSize() const;
    const SynapseGroup *getSynapseGroup() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const SynapseGroupInternal *m_SG;
};

//----------------------------------------------------------------------------
// WUPostVarReference
//----------------------------------------------------------------------------
class WUPostVarReference : public VarReferenceBase
{
public:
    WUPostVarReference(const SynapseGroup *sg, const std::string &varName);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getSize() const;
    const SynapseGroup *getSynapseGroup() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const SynapseGroupInternal *m_SG;
};