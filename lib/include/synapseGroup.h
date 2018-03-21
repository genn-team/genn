#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
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
                 const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<NewModels::VarInit> &wuVarInitialisers,
                 const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<NewModels::VarInit> &psVarInitialisers,
                 NeuronGroup *srcNeuronGroup, NeuronGroup *trgNeuronGroup);
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

    //!< Function to enable the use of zero-copied memory for a particular weight update model state variable (deprecated use SynapseGroup::setWUVarMode):
    /*! May improve IO performance at the expense of kernel performance */
    void setWUVarZeroCopyEnabled(const std::string &varName, bool enabled)
    {
        setWUVarMode(varName, enabled ? VarMode::LOC_ZERO_COPY_INIT_HOST : VarMode::LOC_HOST_DEVICE_INIT_HOST);
    }

    //!< Function to enable the use zero-copied memory for a particular postsynaptic model state variable (deprecated use SynapseGroup::setWUVarMode)
    /*! May improve IO performance at the expense of kernel performance */
    void setPSVarZeroCopyEnabled(const std::string &varName, bool enabled)
    {
        setPSVarMode(varName, enabled ? VarMode::LOC_ZERO_COPY_INIT_HOST : VarMode::LOC_HOST_DEVICE_INIT_HOST);
    }

    //! Set variable mode of weight update model state variable
    /*! This is ignored for CPU simulations */
    void setWUVarMode(const std::string &varName, VarMode mode);

    //! Set variable mode of postsynaptic model state variable
    /*! This is ignored for CPU simulations */
    void setPSVarMode(const std::string &varName, VarMode mode);

    //! Set variable mode used for variables used to combine input from this synapse group
    /*! This is ignored for CPU simulations */
    void setInSynVarMode(VarMode mode) { m_InSynVarMode = mode; }

    //! Sets the maximum number of target neurons any source neurons can connect to
    /*! Use with SynapseMatrixType::SPARSE_GLOBALG and SynapseMatrixType::SPARSE_INDIVIDUALG to optimise CUDA implementation */
    void setMaxConnections(unsigned int maxConnections);

    //! Set how CUDA implementation is parallelised
    /*! with a thread per target neuron (default) or a thread per source spike */
    void setSpanType(SpanType spanType);

    void initDerivedParams(double dt);
    void calcKernelSizes(unsigned int blockSize, unsigned int &paddedKernelIDStart);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    std::pair<unsigned int, unsigned int> getPaddedKernelIDRange() const{ return m_PaddedKernelIDRange; }

    const std::string &getName() const{ return m_Name; }

    SpanType getSpanType() const{ return m_SpanType; }
    unsigned int getDelaySteps() const{ return m_DelaySteps; }
    unsigned int getMaxConnections() const{ return m_MaxConnections; }
    SynapseMatrixType getMatrixType() const{ return m_MatrixType; }

    //! Get variable mode used for variables used to combine input from this synapse group
    VarMode getInSynVarMode() const { return m_InSynVarMode; }

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
    const std::vector<double> getWUConstInitVals() const;

    const PostsynapticModels::Base *getPSModel() const{ return m_PSModel; }

    const std::vector<double> &getPSParams() const{ return m_PSParams; }
    const std::vector<double> &getPSDerivedParams() const{ return m_PSDerivedParams; }
    const std::vector<NewModels::VarInit> &getPSVarInitialisers() const{ return m_PSVarInitialisers; }
    const std::vector<double> getPSConstInitVals() const;

    bool isZeroCopyEnabled() const;
    bool isWUVarZeroCopyEnabled(const std::string &var) const{ return (getWUVarMode(var) & VarLocation::ZERO_COPY); }
    bool isPSVarZeroCopyEnabled(const std::string &var) const{ return (getPSVarMode(var) & VarLocation::ZERO_COPY); }

    //! Get variable mode used by weight update model state variable
    VarMode getWUVarMode(const std::string &var) const;

    //! Get variable mode used by weight update model state variable
    VarMode getWUVarMode(size_t index) const{ return m_WUVarMode[index]; }

    //! Get variable mode used by postsynaptic model state variable
    VarMode getPSVarMode(const std::string &var) const;

    //! Get variable mode used by postsynaptic model state variable
    VarMode getPSVarMode(size_t index) const{ return m_PSVarMode[index]; }

    void addExtraGlobalNeuronParams(std::map<string, string> &kernelParameters) const;
    void addExtraGlobalSynapseParams(std::map<string, string> &kernelParameters) const;
    void addExtraGlobalPostLearnParams(std::map<string, string> &kernelParameters) const;
    void addExtraGlobalSynapseDynamicsParams(std::map<string, string> &kernelParameters) const;

    // **THINK** do these really belong here - they are very code-generation specific
    std::string getOffsetPre() const;
    std::string getOffsetPost(const std::string &devPrefix) const;

    //! Does this synapse group require an RNG for it's postsynaptic init code
    bool isPSInitRNGRequired(VarInit varInitMode) const;

    //! Does this synapse group require an RNG for it's weight update init code
    bool isWUInitRNGRequired(VarInit varInitMode) const;

    //! Is device var init code required for any variables in this synapse group's postsynaptic model
    bool isPSDeviceVarInitRequired() const;

    //! Is device var init code required for any variables in this synapse group's weight update model
    bool isWUDeviceVarInitRequired() const;

    //! Can this synapse group run on the CPU?
    /*! If we are running in CPU_ONLY mode this is always true,
        but some GPU functionality will prevent models being run on both CPU and GPU.*/
    bool canRunOnCPU() const;

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
    //!< Range of indices of this synapse group in synapse kernel
    std::pair<unsigned int, unsigned int> m_PaddedKernelIDRange;

    //!< Name of the synapse group
    std::string m_Name;

    //!< Execution order of synapses in the kernel. It determines whether synapses are executed in parallel for every postsynaptic neuron, or for every presynaptic neuron.
    SpanType m_SpanType;

    //!< Global synaptic conductance delay for the group (in time steps)
    unsigned int m_DelaySteps;

    //!< Padded summed maximum number of connections for a neuron in the neuron groups
    unsigned int m_MaxConnections;

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

    //!< Weight update model type
    const WeightUpdateModels::Base *m_WUModel;

    //!< Parameters of weight update model
    std::vector<double> m_WUParams;

    //!< Derived parameters for weight update model
    std::vector<double> m_WUDerivedParams;

    //!< Initialisers for weight update model variables
    std::vector<NewModels::VarInit> m_WUVarInitialisers;

    //!< Post synapse update model type
    const PostsynapticModels::Base *m_PSModel;

    //!< Parameters of post synapse model
    std::vector<double> m_PSParams;

    //!< Derived parameters for post synapse model
    std::vector<double> m_PSDerivedParams;

    //!< Initialisers for post synapse model variables
    std::vector<NewModels::VarInit> m_PSVarInitialisers;

    //!< Whether indidividual state variables of weight update model should use zero-copied memory
    std::vector<VarMode> m_WUVarMode;

    //!< Whether indidividual state variables of post synapse should use zero-copied memory
    std::vector<VarMode> m_PSVarMode;
};