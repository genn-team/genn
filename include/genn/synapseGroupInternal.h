#pragma once


// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "initSparseConnectivitySnippet.h"
#include "postsynapticModels.h"
#include "weightUpdateModels.h"
#include "synapseGroup.h"
#include "synapseMatrixType.h"
#include "variableMode.h"

// Forward declarations
class NeuronGroupInternal;

//------------------------------------------------------------------------
// SynapseGroupInternal
//------------------------------------------------------------------------
class SynapseGroupInternal : public SynapseGroup
{
public:
    SynapseGroupInternal(const std::string name, SynapseMatrixType matrixType, unsigned int delaySteps,
                         const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<Models::VarInit> &wuVarInitialisers, const std::vector<Models::VarInit> &wuPreVarInitialisers, const std::vector<Models::VarInit> &wuPostVarInitialisers,
                         const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<Models::VarInit> &psVarInitialisers,
                         NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                         const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                         VarLocation defaultVarLocation, VarLocation defaultSparseConnectivityLocation);
    
    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    void setEventThresholdReTestRequired(bool req){ m_EventThresholdReTestRequired = req; }

    void setPSModelMergeTarget(const std::string &targetName)
    {
        m_PSModelTargetName = targetName;
    }
    
    void initDerivedParams(double dt);
    
    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    using SynapseGroup::getSrcNeuronGroup;
    using SynapseGroup::getTrgNeuronGroup;
    
    //! Does synapse group need to handle 'true' spikes
    bool isTrueSpikeRequired() const;

    //! Does synapse group need to handle spike-like events
    bool isSpikeEventRequired() const;

    //!< Does the event threshold needs to be retested in the synapse kernel?
    /*! This is required when the pre-synaptic neuron population's outgoing synapse groups require different event threshold */
    bool isEventThresholdReTestRequired() const{ return m_EventThresholdReTestRequired; }
    
    const std::vector<double> &getWUDerivedParams() const{ return m_WUDerivedParams; }
    const std::vector<double> &getPSDerivedParams() const{ return m_PSDerivedParams; }
    
    const std::string &getPSModelTargetName() const{ return m_PSModelTargetName; }
    bool isPSModelMerged() const{ return m_PSModelTargetName != getName(); }
    
    //! Does this synapse group require dendritic delay?
    bool isDendriticDelayRequired() const;

    //! Does this synapse group require an RNG for it's postsynaptic init code?
    bool isPSInitRNGRequired() const;

    //! Does this synapse group require an RNG for it's weight update init code?
    bool isWUInitRNGRequired() const;

    //! Is device var init code required for any variables in this synapse group's postsynaptic model?
    bool isPSVarInitRequired() const;

    //! Is var init code required for any variables in this synapse group's weight update model?
    bool isWUVarInitRequired() const;

    //! Is var init code required for any presynaptic variables in this synapse group's weight update model
    bool isWUPreVarInitRequired() const;

    //! Is var init code required for any postsynaptic variables in this synapse group's weight update model
    bool isWUPostVarInitRequired() const;

    //! Is sparse connectivity initialisation code required for this synapse group?
    bool isSparseConnectivityInitRequired() const;

    //! Is any form of device initialisation required?
    bool isInitRequired() const;

    //! Is any form of sparse device initialisation required?
    bool isSparseInitRequired() const;
    
     //! Get the expression to calculate the delay slot for accessing
    //! Presynaptic neuron state variables, taking into account axonal delay
    std::string getPresynapticAxonalDelaySlot(const std::string &devPrefix) const;

    //! Get the expression to calculate the delay slot for accessing
    //! Postsynaptic neuron state variables, taking into account back propagation delay
    std::string getPostsynapticBackPropDelaySlot(const std::string &devPrefix) const;

    std::string getDendriticDelayOffset(const std::string &devPrefix, const std::string &offset = "") const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //!< Derived parameters for weight update model
    std::vector<double> m_WUDerivedParams;
    
     //!< Derived parameters for post synapse model
    std::vector<double> m_PSDerivedParams;

     //!< Does the event threshold needs to be retested in the synapse kernel?
    /*! This is required when the pre-synaptic neuron population's outgoing synapse groups require different event threshold */
    bool m_EventThresholdReTestRequired;
    
    
    //! Name of the synapse group in which postsynaptic model is located
    /*! This may not be the name of this group if it has been merged*/
    std::string m_PSModelTargetName;
};