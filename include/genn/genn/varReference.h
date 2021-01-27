#pragma once

// Standard C++ includes
#include <string>

// GeNN includes
#include "models.h"

// Forward declarations
class NeuronGroup;
class SynapseGroup;
class CurrentSource;
class SynapseGroupInternal;
namespace CodeGenerator
{
class BackendBase;
}

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

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    std::string getVarName() const;
    size_t getVarSize(const CodeGenerator::BackendBase &backend) const;
    Type getType() const
    {
        return m_Type;
    }

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    static VarReference create(const NeuronGroup *ng, const std::string &varName);
    static VarReference create(const CurrentSource *cs, const std::string &varName);
    static VarReference createPSM(const SynapseGroup *sg, const std::string &varName);
    static VarReference createWU(const SynapseGroup *sg, const std::string &varName);
    static VarReference createWUPre(const SynapseGroup *sg, const std::string &varName);
    static VarReference createWUPost(const SynapseGroup *sg, const std::string &varName);

private:
    VarReference(const NeuronGroup *ng, Models::Base::Var var, Type type);
    VarReference(const SynapseGroup *sg, Models::Base::Var var, Type type);
    VarReference(const CurrentSource *cs, Models::Base::Var var, Type type);
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const Models::Base::Var m_Var;
    const Type m_Type;
    
    const NeuronGroup *m_NG;
    const SynapseGroupInternal *m_SG;
    const CurrentSource *m_CS;
};