#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "initSparseConnectivitySnippet.h"
#include "neuronGroup.h"
#include "newPostsynapticModels.h"
#include "newWeightUpdateModels.h"
#include "synapseMatrixType.h"

//------------------------------------------------------------------------
// SynapseGroup
//------------------------------------------------------------------------
class SynapseGroup
{
public:
    SynapseGroup(const std::string name, SynapseMatrixType matrixType, unsigned int delaySteps,
                 const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<NewModels::VarInit> &wuVarInitialisers, const std::vector<NewModels::VarInit> &wuPreVarInitialisers, const std::vector<NewModels::VarInit> &wuPostVarInitialisers,
                 const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<NewModels::VarInit> &psVarInitialisers,
                 NeuronGroup *srcNeuronGroup, NeuronGroup *trgNeuronGroup,
                 const InitSparseConnectivitySnippet::Init &connectivityInitialiser);
    SynapseGroup(const SynapseGroup&) = delete;
    SynapseGroup() = delete;

    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class SpanType
    {
        POSTSYNAPTIC,
        PRESYNAPTIC
    };

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    NeuronGroup *getSrcNeuronGroup(){ return m_SrcNeuronGroup; }
    NeuronGroup *getTrgNeuronGroup(){ return m_TrgNeuronGroup; }

    void setTrueSpikeRequired(bool req){ m_TrueSpikeRequired = req; }
    void setSpikeEventRequired(bool req){ m_SpikeEventRequired = req; }
    void setEventThresholdReTestRequired(bool req){ m_EventThresholdReTestRequired = req; }

    void setPSModelMergeTarget(const std::string &targetName)
    {
        m_PSModelTargetName = targetName;
    }

    //! Set variable mode of weight update model state variable
    /*! This is ignored for CPU simulations */
    void setWUVarMode(const std::string &varName, VarMode mode);

    //! Set variable mode of weight update model presynaptic state variable
    /*! This is ignored for CPU simulations */
    void setWUPreVarMode(const std::string &varName, VarMode mode);
    
    //! Set variable mode of weight update model postsynaptic state variable
    /*! This is ignored for CPU simulations */
    void setWUPostVarMode(const std::string &varName, VarMode mode);
    
    //! Set variable mode of postsynaptic model state variable
    /*! This is ignored for CPU simulations */
    void setPSVarMode(const std::string &varName, VarMode mode);

    //! Set variable mode used for variables used to combine input from this synapse group
    /*! This is ignored for CPU simulations */
    void setInSynVarMode(VarMode mode) { m_InSynVarMode = mode; }

    //! Set variable mode used for sparse connectivity
    /*! This is ignored for CPU simulations */
    void setSparseConnectivityVarMode(VarMode mode){ m_SparseConnectivityVarMode = mode; }

    //! Set variable mode used for this synapse group's dendritic delay buffers
    /*! This is ignored for CPU simulations */
    void setDendriticDelayVarMode(VarMode mode) { m_DendriticDelayVarMode = mode; }

    //! Sets the maximum number of target neurons any source neurons can connect to
    /*! Use with synaptic matrix types with SynapseMatrixConnectivity::SPARSE to optimise CUDA implementation */
    void setMaxConnections(unsigned int maxConnections);

    //! Sets the maximum number of source neurons any target neuron can connect to
    /*! Use with synaptic matrix types with SynapseMatrixConnectivity::SPARSE and postsynaptic learning to optimise CUDA implementation */
    void setMaxSourceConnections(unsigned int maxPostConnections);
    
    //! Sets the maximum dendritic delay for synapses in this synapse group
    void setMaxDendriticDelayTimesteps(unsigned int maxDendriticDelay);
    
    //! Set how CUDA implementation is parallelised
    /*! with a thread per target neuron (default) or a thread per source spike */
    void setSpanType(SpanType spanType);

    //! Sets the number of delay steps used to delay postsynaptic spikes travelling back along dendrites to synapses
    void setBackPropDelaySteps(unsigned int timesteps);

    void initDerivedParams(double dt);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    SpanType getSpanType() const{ return m_SpanType; }
    unsigned int getDelaySteps() const{ return m_DelaySteps; }
    unsigned int getBackPropDelaySteps() const{ return m_BackPropDelaySteps; }
    unsigned int getMaxConnections() const{ return m_MaxConnections; }
    unsigned int getMaxSourceConnections() const{ return m_MaxSourceConnections; }
    unsigned int getMaxDendriticDelayTimesteps() const{ return m_MaxDendriticDelayTimesteps; }
    SynapseMatrixType getMatrixType() const{ return m_MatrixType; }

    //! Get variable mode used for variables used to combine input from this synapse group
    VarMode getInSynVarMode() const { return m_InSynVarMode; }

    //! Get variable mode used for sparse connectivity
    VarMode getSparseConnectivityVarMode() const{ return m_SparseConnectivityVarMode; }

    //! Get variable mode used for this synapse group's dendritic delay buffers
    VarMode getDendriticDelayVarMode() const{ return m_DendriticDelayVarMode; }

    unsigned int getPaddedDynKernelSize(unsigned int blockSize) const;
    unsigned int getPaddedPostLearnKernelSize(unsigned int blockSize) const;

    const NeuronGroup *getSrcNeuronGroup() const{ return m_SrcNeuronGroup; }
    const NeuronGroup *getTrgNeuronGroup() const{ return m_TrgNeuronGroup; }

    int getClusterHostID() const{ return m_TrgNeuronGroup->getClusterHostID(); }
    int getClusterDeviceID() const{ return m_TrgNeuronGroup->getClusterDeviceID(); }

    bool isTrueSpikeRequired() const{ return m_TrueSpikeRequired; }
    bool isSpikeEventRequired() const{ return m_SpikeEventRequired; }
    bool isEventThresholdReTestRequired() const{ return m_EventThresholdReTestRequired; }

    const WeightUpdateModels::Base *getWUModel() const{ return m_WUModel; }

    const std::vector<double> &getWUParams() const{ return m_WUParams; }
    const std::vector<double> &getWUDerivedParams() const{ return m_WUDerivedParams; }
    const std::vector<NewModels::VarInit> &getWUVarInitialisers() const{ return m_WUVarInitialisers; }
    const std::vector<NewModels::VarInit> &getWUPreVarInitialisers() const{ return m_WUPreVarInitialisers; }
    const std::vector<NewModels::VarInit> &getWUPostVarInitialisers() const{ return m_WUPostVarInitialisers; }
    const std::vector<double> getWUConstInitVals() const;

    const PostsynapticModels::Base *getPSModel() const{ return m_PSModel; }

    const std::vector<double> &getPSParams() const{ return m_PSParams; }
    const std::vector<double> &getPSDerivedParams() const{ return m_PSDerivedParams; }
    const std::vector<NewModels::VarInit> &getPSVarInitialisers() const{ return m_PSVarInitialisers; }
    const std::vector<double> getPSConstInitVals() const;

    const InitSparseConnectivitySnippet::Init &getConnectivityInitialiser() const{ return m_ConnectivityInitialiser; }

    const std::string &getPSModelTargetName() const{ return m_PSModelTargetName; }
    bool isPSModelMerged() const{ return m_PSModelTargetName != getName(); }

    bool isZeroCopyEnabled() const;
    bool isWUVarZeroCopyEnabled(const std::string &var) const{ return (getWUVarMode(var) & VarLocation::ZERO_COPY); }
    bool isPSVarZeroCopyEnabled(const std::string &var) const{ return (getPSVarMode(var) & VarLocation::ZERO_COPY); }

    //! Get variable mode used by weight update model per-synapse state variable
    VarMode getWUVarMode(const std::string &var) const;

    //! Get variable mode used by weight update model per-synapse state variable
    VarMode getWUVarMode(size_t index) const{ return m_WUVarMode[index]; }

    //! Get variable mode used by weight update model presynaptic state variable
    VarMode getWUPreVarMode(const std::string &var) const;

    //! Get variable mode used by weight update model presynaptic state variable
    VarMode getWUPreVarMode(size_t index) const{ return m_WUPreVarMode[index]; }

    //! Get variable mode used by weight update model postsynaptic state variable
    VarMode getWUPostVarMode(const std::string &var) const;

    //! Get variable mode used by weight update model postsynaptic state variable
    VarMode getWUPostVarMode(size_t index) const{ return m_WUPostVarMode[index]; }

    //! Get variable mode used by postsynaptic model state variable
    VarMode getPSVarMode(const std::string &var) const;

    //! Get variable mode used by postsynaptic model state variable
    VarMode getPSVarMode(size_t index) const{ return m_PSVarMode[index]; }

    void addExtraGlobalConnectivityInitialiserParams(std::map<std::string, std::string> &kernelParameters) const;
    void addExtraGlobalNeuronParams(std::map<std::string, std::string> &kernelParameters) const;
    void addExtraGlobalSynapseParams(std::map<std::string, std::string> &kernelParameters) const;
    void addExtraGlobalPostLearnParams(std::map<std::string, std::string> &kernelParameters) const;
    void addExtraGlobalSynapseDynamicsParams(std::map<std::string, std::string> &kernelParameters) const;

    //! Get the expression to calculate the delay slot for accessing
    //! Presynaptic neuron state variables, taking into account axonal delay
    std::string getPresynapticAxonalDelaySlot(const std::string &devPrefix) const;

    //! Get the expression to calculate the delay slot for accessing
    //! Postsynaptic neuron state variables, taking into account back propagation delay
    std::string getPostsynapticBackPropDelaySlot(const std::string &devPrefix) const;

    std::string getDendriticDelayOffset(const std::string &devPrefix, const std::string &offset = "") const;

    //! Does this synapse group require dendritic delay?
    bool isDendriticDelayRequired() const;

    //! Does this synapse group require an RNG for it's postsynaptic init code?
    bool isPSInitRNGRequired(VarInit varInitMode) const;

    //! Does this synapse group require an RNG for it's weight update init code?
    bool isWUInitRNGRequired(VarInit varInitMode) const;

    //! Is device var init code required for any variables in this synapse group's postsynaptic model?
    bool isPSDeviceVarInitRequired() const;

    //! Is device var init code required for any variables in this synapse group's weight update model?
    bool isWUDeviceVarInitRequired() const;

    //! Is device var init code required for any presynaptic variables in this synapse group's weight update model
    bool isWUDevicePreVarInitRequired() const;

    //! Is device var init code required for any postsynaptic variables in this synapse group's weight update model
    bool isWUDevicePostVarInitRequired() const;

    //! Is device sparse connectivity initialisation code required for this synapse group?
    bool isDeviceSparseConnectivityInitRequired() const;

    //! Is any form of device initialisation required?
    bool isDeviceInitRequired() const;

    //! Is any form of sparse device initialisation required?
    bool isDeviceSparseInitRequired() const;
private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void addExtraGlobalSimParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                 std::map<std::string, std::string> &kernelParameters) const;
    void addExtraGlobalPostLearnParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                       std::map<std::string, std::string> &kernelParameters) const;
    void addExtraGlobalSynapseDynamicsParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                             std::map<std::string, std::string> &kernelParameters) const;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //!< Name of the synapse group
    std::string m_Name;

    //!< Execution order of synapses in the kernel. It determines whether synapses are executed in parallel for every postsynaptic neuron, or for every presynaptic neuron.
    SpanType m_SpanType;

    //!< Global synaptic conductance delay for the group (in time steps)
    unsigned int m_DelaySteps;

    //!< Global backpropagation delay for postsynaptic spikes to synapse (in time
    unsigned int m_BackPropDelaySteps;

    //!< Maximum number of target neurons any source neuron can connect to
    unsigned int m_MaxConnections;
    
    //!< Maximum number of source neurons any target neuron can connect to
    unsigned int m_MaxSourceConnections;

    //!< Maximum dendritic delay timesteps supported for synapses in this population
    unsigned int m_MaxDendriticDelayTimesteps;
    
    //!< Connectivity type of synapses
    SynapseMatrixType m_MatrixType;

    //!< Pointer to presynaptic neuron group
    NeuronGroup *m_SrcNeuronGroup;

    //!< Pointer to postsynaptic neuron group
    NeuronGroup *m_TrgNeuronGroup;

    //!< Defines if synapse update is done after detection of real spikes (only one point after threshold)
    bool m_TrueSpikeRequired;

    //!< Defines if synapse update is done after detection of spike events (every point above threshold)
    bool m_SpikeEventRequired;

    //!< Defines whether the Evnt Threshold needs to be retested in the synapse kernel due to multiple non-identical events in the pre-synaptic neuron population
    bool m_EventThresholdReTestRequired;

    //!< Variable mode used for variables used to combine input from this synapse group
    VarMode m_InSynVarMode;

    //!< Variable mode used for this synapse group's dendritic delay buffers
    VarMode m_DendriticDelayVarMode;

    //!< Weight update model type
    const WeightUpdateModels::Base *m_WUModel;

    //!< Parameters of weight update model
    std::vector<double> m_WUParams;

    //!< Derived parameters for weight update model
    std::vector<double> m_WUDerivedParams;

    //!< Initialisers for weight update model per-synapse variables
    std::vector<NewModels::VarInit> m_WUVarInitialisers;

    //!< Initialisers for weight update model per-presynaptic neuron variables
    std::vector<NewModels::VarInit> m_WUPreVarInitialisers;

    //!< Initialisers for weight update model post-presynaptic neuron variables
    std::vector<NewModels::VarInit> m_WUPostVarInitialisers;
    
    //!< Post synapse update model type
    const PostsynapticModels::Base *m_PSModel;

    //!< Parameters of post synapse model
    std::vector<double> m_PSParams;

    //!< Derived parameters for post synapse model
    std::vector<double> m_PSDerivedParams;

    //!< Initialisers for post synapse model variables
    std::vector<NewModels::VarInit> m_PSVarInitialisers;

    //!< Whether individual per-synapse state variables of weight update model should use zero-copied memory
    std::vector<VarMode> m_WUVarMode;

    //!< Whether individual presynaptic state variables of weight update model should use zero-copied memory
    std::vector<VarMode> m_WUPreVarMode;

    //!< Whether individual postsynaptic state variables of weight update model should use zero-copied memory
    std::vector<VarMode> m_WUPostVarMode;

    //!< Whether indidividual state variables of post synapse should use zero-copied memory
    std::vector<VarMode> m_PSVarMode;

    //!< Initialiser used for creating sparse connectivity
    InitSparseConnectivitySnippet::Init m_ConnectivityInitialiser;

    //!< Variable mode used for sparse connectivity
    VarMode m_SparseConnectivityVarMode;

    //! Name of the synapse group in which postsynaptic model is located
    /*! This may not be the name of this group if it has been merged*/
    std::string m_PSModelTargetName;
};
