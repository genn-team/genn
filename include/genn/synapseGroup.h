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
#include "synapseMatrixType.h"
#include "variableMode.h"

// Forward declarations
class NeuronGroupInternal;

//------------------------------------------------------------------------
// SynapseGroup
//------------------------------------------------------------------------
class SynapseGroup
{
public:
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
    //! Set location of weight update model state variable
    /*! This is ignored for simulations on harware with a single memory space */
    void setWUVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of weight update model presynaptic state variable
    /*! This is ignored for simulations on harware with a single memory space */
    void setWUPreVarLocation(const std::string &varName, VarLocation loc);
    
    //! Set location of weight update model postsynaptic state variable
    /*! This is ignored for simulations on harware with a single memory space */
    void setWUPostVarLocation(const std::string &varName, VarLocation loc);
    
    //! Set location of postsynaptic model state variable
    /*! This is ignored for simulations on harware with a single memory space */
    void setPSVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of variables used to combine input from this synapse group
    /*! This is ignored for simulations on harware with a single memory space */
    void setInSynVarLocation(VarLocation loc) { m_InSynLocation = loc; }

    //! Set variable mode used for sparse connectivity
    /*! ThisThis is ignored for simulations on harware with a single memory space */
    void setSparseConnectivityLocation(VarLocation loc){ m_SparseConnectivityLocation = loc; }

    //! Set variable mode used for this synapse group's dendritic delay buffers
    /*! This is ignored for simulations on harware with a single memory space */
    void setDendriticDelayLocation(VarLocation loc) { m_DendriticDelayLocation = loc; }

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
    VarLocation getInSynLocation() const { return m_InSynLocation; }

    //! Get variable mode used for sparse connectivity
    VarLocation getSparseConnectivityLocation() const{ return m_SparseConnectivityLocation; }

    //! Get variable mode used for this synapse group's dendritic delay buffers
    VarLocation getDendriticDelayLocation() const{ return m_DendriticDelayLocation; }

    int getClusterHostID() const;
    int getClusterDeviceID() const;

    const WeightUpdateModels::Base *getWUModel() const{ return m_WUModel; }

    const std::vector<double> &getWUParams() const{ return m_WUParams; }
    const std::vector<Models::VarInit> &getWUVarInitialisers() const{ return m_WUVarInitialisers; }
    const std::vector<Models::VarInit> &getWUPreVarInitialisers() const{ return m_WUPreVarInitialisers; }
    const std::vector<Models::VarInit> &getWUPostVarInitialisers() const{ return m_WUPostVarInitialisers; }
    const std::vector<double> getWUConstInitVals() const;

    const PostsynapticModels::Base *getPSModel() const{ return m_PSModel; }

    const std::vector<double> &getPSParams() const{ return m_PSParams; }
    const std::vector<Models::VarInit> &getPSVarInitialisers() const{ return m_PSVarInitialisers; }
    const std::vector<double> getPSConstInitVals() const;

    const InitSparseConnectivitySnippet::Init &getConnectivityInitialiser() const{ return m_ConnectivityInitialiser; }

    bool isZeroCopyEnabled() const;
    bool isWUVarZeroCopyEnabled(const std::string &var) const{ return (getWUVarLocation(var) & VarLocation::ZERO_COPY); }
    bool isPSVarZeroCopyEnabled(const std::string &var) const{ return (getPSVarLocation(var) & VarLocation::ZERO_COPY); }

    //! Get variable mode used by weight update model per-synapse state variable
    VarLocation getWUVarLocation(const std::string &var) const;

    //! Get variable mode used by weight update model per-synapse state variable
    VarLocation getWUVarLocation(size_t index) const{ return m_WUVarLocation[index]; }

    //! Get variable mode used by weight update model presynaptic state variable
    VarLocation getWUPreVarLocation(const std::string &var) const;

    //! Get variable mode used by weight update model presynaptic state variable
    VarLocation getWUPreVarLocation(size_t index) const{ return m_WUPreVarLocation[index]; }

    //! Get variable mode used by weight update model postsynaptic state variable
    VarLocation getWUPostVarLocation(const std::string &var) const;

    //! Get variable mode used by weight update model postsynaptic state variable
    VarLocation getWUPostVarLocation(size_t index) const{ return m_WUPostVarLocation[index]; }

    //! Get variable mode used by postsynaptic model state variable
    VarLocation getPSVarLocation(const std::string &var) const;

    //! Get variable mode used by postsynaptic model state variable
    VarLocation getPSVarLocation(size_t index) const{ return m_PSVarLocation[index]; }

protected:
    SynapseGroup(const std::string name, SynapseMatrixType matrixType, unsigned int delaySteps,
                 const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<Models::VarInit> &wuVarInitialisers, const std::vector<Models::VarInit> &wuPreVarInitialisers, const std::vector<Models::VarInit> &wuPostVarInitialisers,
                 const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<Models::VarInit> &psVarInitialisers,
                 NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                 const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                 VarLocation defaultVarLocation, VarLocation defaultSparseConnectivityLocation);

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const NeuronGroupInternal *getSrcNeuronGroup() const{ return m_SrcNeuronGroup; }
    const NeuronGroupInternal *getTrgNeuronGroup() const{ return m_TrgNeuronGroup; }
    
    NeuronGroupInternal *getSrcNeuronGroup(){ return m_SrcNeuronGroup; }
    NeuronGroupInternal *getTrgNeuronGroup(){ return m_TrgNeuronGroup; }
    
    void initInitialiserDerivedParams(double dt);
private:
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
    NeuronGroupInternal *m_SrcNeuronGroup;

    //!< Pointer to postsynaptic neuron group
    NeuronGroupInternal *m_TrgNeuronGroup;

    //!< Variable mode used for variables used to combine input from this synapse group
    VarLocation m_InSynLocation;

    //!< Variable mode used for this synapse group's dendritic delay buffers
    VarLocation m_DendriticDelayLocation;

    //!< Weight update model type
    const WeightUpdateModels::Base *m_WUModel;

    //!< Parameters of weight update model
    std::vector<double> m_WUParams;

    //!< Initialisers for weight update model per-synapse variables
    std::vector<Models::VarInit> m_WUVarInitialisers;

    //!< Initialisers for weight update model per-presynaptic neuron variables
    std::vector<Models::VarInit> m_WUPreVarInitialisers;

    //!< Initialisers for weight update model post-presynaptic neuron variables
    std::vector<Models::VarInit> m_WUPostVarInitialisers;
    
    //!< Post synapse update model type
    const PostsynapticModels::Base *m_PSModel;

    //!< Parameters of post synapse model
    std::vector<double> m_PSParams;

    //!< Initialisers for post synapse model variables
    std::vector<Models::VarInit> m_PSVarInitialisers;

    //!< Location of individual per-synapse state variables 
    std::vector<VarLocation> m_WUVarLocation;

    //!< Location of individual presynaptic state variables
    std::vector<VarLocation> m_WUPreVarLocation;

    //!< Location of individual postsynaptic state variables
    std::vector<VarLocation> m_WUPostVarLocation;

    //!< Whether indidividual state variables of post synapse should use zero-copied memory
    std::vector<VarLocation> m_PSVarLocation;

    //!< Initialiser used for creating sparse connectivity
    InitSparseConnectivitySnippet::Init m_ConnectivityInitialiser;

    //!< Location of sparse connectivity
    VarLocation m_SparseConnectivityLocation;
};
