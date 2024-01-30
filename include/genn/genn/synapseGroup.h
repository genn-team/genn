#pragma once

// Standard includes
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "initSparseConnectivitySnippet.h"
#include "initToeplitzConnectivitySnippet.h"
#include "postsynapticModels.h"
#include "weightUpdateModels.h"
#include "synapseMatrixType.h"
#include "varLocation.h"

// Forward declarations
namespace GeNN
{
class CustomConnectivityUpdateInternal;
class CustomUpdateWUInternal;
class NeuronGroupInternal;
class SynapseGroupInternal;
}

//------------------------------------------------------------------------
// GeNN::SynapseGroup
//------------------------------------------------------------------------
namespace GeNN
{
class GENN_EXPORT SynapseGroup
{
public:
    SynapseGroup(const SynapseGroup&) = delete;
    SynapseGroup() = delete;

    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class ParallelismHint
    {
        POSTSYNAPTIC,
        PRESYNAPTIC,
        WORD_PACKED_BITMASK,
    };

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set location of weight update model state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setWUVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of weight update model presynaptic state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setWUPreVarLocation(const std::string &varName, VarLocation loc);
    
    //! Set location of weight update model postsynaptic state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setWUPostVarLocation(const std::string &varName, VarLocation loc);
    
    //! Set location of weight update model extra global parameter
    /*! This is ignored for simulations on hardware with a single memory space. */
    void setWUExtraGlobalParamLocation(const std::string &paramName, VarLocation loc);

    //! Set location of postsynaptic model state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setPSVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of postsynaptic model extra global parameter
    /*! This is ignored for simulations on hardware with a single memory space. */
    void setPSExtraGlobalParamLocation(const std::string &paramName, VarLocation loc);

    //! Set whether weight update model parameter is dynamic or not i.e. it can be changed at runtime
    void setWUParamDynamic(const std::string &paramName, bool dynamic = true);

    //! Set whether weight update model parameter is dynamic or not i.e. it can be changed at runtime
    void setPSParamDynamic(const std::string &paramName, bool dynamic = true);

    //! Set name of neuron input variable postsynaptic model will target
    /*! This should either be 'Isyn' or the name of one of the postsynaptic neuron's additional input variables. */
    void setPostTargetVar(const std::string &varName);
    
    //! Set name of neuron input variable $(addToPre, . ) commands will target
    /*! This should either be 'Isyn' or the name of one of the presynaptic neuron's additional input variables. */
    void setPreTargetVar(const std::string &varName);

    //! Set location of variables used for outputs from this synapse group e.g. outPre and outPost
    /*! This is ignored for simulations on hardware with a single memory space */
    void setOutputLocation(VarLocation loc) { m_OutputLocation = loc; }

    //! Set variable mode used for sparse connectivity
    /*! This is ignored for simulations on hardware with a single memory space */
    void setSparseConnectivityLocation(VarLocation loc) { m_SparseConnectivityLocation = loc; }

    //! Set variable mode used for this synapse group's dendritic delay buffers
    void setDendriticDelayLocation(VarLocation loc) { m_DendriticDelayLocation = loc; }

    //! Sets the maximum number of target neurons any source neurons can connect to
    /*! Use with synaptic matrix types with SynapseMatrixConnectivity::SPARSE to optimise CUDA implementation */
    void setMaxConnections(unsigned int maxConnections);

    //! Sets the maximum number of source neurons any target neuron can connect to
    /*! Use with synaptic matrix types with SynapseMatrixConnectivity::SPARSE and postsynaptic learning to optimise CUDA implementation */
    void setMaxSourceConnections(unsigned int maxPostConnections);
    
    //! Sets the maximum dendritic delay for synapses in this synapse group
    void setMaxDendriticDelayTimesteps(unsigned int maxDendriticDelay);

    //! Sets the number of delay steps used to delay events and variables between presynaptic neuron and synapse
    void setAxonalDelaySteps(unsigned int timesteps);

    //! Provide a hint as to how this synapse group should be parallelised
    void setParallelismHint(ParallelismHint parallelismHint){ m_ParallelismHint = parallelismHint; }

    //! Provide hint as to how many threads SIMT backend might use to process each spike if PRESYNAPTIC parallelism is selected
    // **TODO** this shouldn't be in SynapseGroup - it's backend-specific
    void setNumThreadsPerSpike(unsigned int numThreadsPerSpike){ m_NumThreadsPerSpike = numThreadsPerSpike; }
 
    //! Sets the number of delay steps used to delay events and variables between postsynaptic neuron and synapse
    void setBackPropDelaySteps(unsigned int timesteps);

    //! Enables or disables using narrow i.e. less than 32-bit types for sparse matrix indices
    void setNarrowSparseIndEnabled(bool enabled);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    ParallelismHint getParallelismHint() const{ return m_ParallelismHint; }
    unsigned int getNumThreadsPerSpike() const{ return m_NumThreadsPerSpike; }
    unsigned int getBackPropDelaySteps() const{ return m_BackPropDelaySteps; }
    unsigned int getAxonalDelaySteps() const{ return m_AxonalDelaySteps; }
    unsigned int getMaxConnections() const{ return m_MaxConnections; }
    unsigned int getMaxSourceConnections() const{ return m_MaxSourceConnections; }
    unsigned int getMaxDendriticDelayTimesteps() const{ return m_MaxDendriticDelayTimesteps; }
    SynapseMatrixType getMatrixType() const{ return m_MatrixType; }
    const std::vector<unsigned int> &getKernelSize() const { return m_KernelSize; }
    size_t getKernelSizeFlattened() const;
    
    //! Get variable mode used for outputs from this synapse group e.g. outPre and outPost
    VarLocation getOutputLocation() const { return m_OutputLocation; }

    //! Get variable mode used for sparse connectivity
    VarLocation getSparseConnectivityLocation() const{ return m_SparseConnectivityLocation; }

    //! Get variable mode used for this synapse group's dendritic delay buffers
    VarLocation getDendriticDelayLocation() const{ return m_DendriticDelayLocation; }

    //! Get location of weight update model per-synapse state variable by name
    VarLocation getWUVarLocation(const std::string &varName) const{ return m_WUVarLocation.get(varName); }

    //! Get location of weight update model presynaptic state variable by name
    VarLocation getWUPreVarLocation(const std::string &varName) const{ return m_WUPreVarLocation.get(varName); }

    //! Get location of weight update model postsynaptic state variable by name
    VarLocation getWUPostVarLocation(const std::string &varName) const{ return m_WUPostVarLocation.get(varName); }

    //! Get location of weight update model extra global parameter by name
    VarLocation getWUExtraGlobalParamLocation(const std::string &paramName) const{ return m_WUExtraGlobalParamLocation.get(paramName); }

    //! Get location of postsynaptic model state variable
    VarLocation getPSVarLocation(const std::string &varName) const{ return m_WUVarLocation.get(varName); }

    //! Get location of postsynaptic model extra global parameter by name
    VarLocation getPSExtraGlobalParamLocation(const std::string &paramName) const{ return m_PSExtraGlobalParamLocation.get(paramName); }

    //! Is postsynaptic model parameter dynamic i.e. it can be changed at runtime
    bool isPSParamDynamic(const std::string &paramName) const{ return m_PSDynamicParams.get(paramName); }

    //! Is weight update model parameter dynamic i.e. it can be changed at runtime
    bool isWUParamDynamic(const std::string &paramName) const{ return m_WUDynamicParams.get(paramName); }

    //! Does synapse group need to handle 'true' spikes/
    bool isPreSpikeRequired() const;

    //! Does synapse group need to handle presynaptic spike-like events?
    bool isPreSpikeEventRequired() const;

    //! Does synapse group need to handle postsynaptic spikes?
    bool isPostSpikeRequired() const;

    //! Does synapse group need to handle postsynaptic spike-like events?
    bool isPostSpikeEventRequired() const;

    //! Are presynaptic spike times needed?
    bool isPreSpikeTimeRequired() const;

    //! Are presynaptic spike-like-event times needed?
    bool isPreSpikeEventTimeRequired() const;

    //! Are PREVIOUS presynaptic spike times needed?
    bool isPrevPreSpikeTimeRequired() const;

    //! Are PREVIOUS presynaptic spike-like-event times needed?
    bool isPrevPreSpikeEventTimeRequired() const;

    //! Are postsynaptic spike times needed?
    bool isPostSpikeTimeRequired() const;

    //! Are postsynaptic spike-like-event times needed?
    bool isPostSpikeEventTimeRequired() const;

    //! Are PREVIOUS postsynaptic spike times needed?
    bool isPrevPostSpikeTimeRequired() const;

    //! Are PREVIOUS postsynaptic spike-event times needed?
    bool isPrevPostSpikeEventTimeRequired() const;

    const PostsynapticModels::Init &getPSInitialiser() const{ return m_PSInitialiser; }
    const WeightUpdateModels::Init &getWUInitialiser() const{ return m_WUInitialiser; }
      
    const InitSparseConnectivitySnippet::Init &getConnectivityInitialiser() const{ return m_SparseConnectivityInitialiser; }
    const InitToeplitzConnectivitySnippet::Init &getToeplitzConnectivityInitialiser() const { return m_ToeplitzConnectivityInitialiser; }

    bool isZeroCopyEnabled() const;

   
    //! Get name of neuron input variable postsynaptic model will target
    /*! This will either be 'Isyn' or the name of one of the postsynaptic neuron's additional input variables. */
    const std::string &getPostTargetVar() const{ return m_PostTargetVar; }

    //! Get name of neuron input variable which a presynaptic output specified with $(addToPre) will target
    /*! This will either be 'Isyn' or the name of one of the presynaptic neuron's additional input variables. */
    const std::string &getPreTargetVar() const{ return m_PreTargetVar; }
    
protected:
    SynapseGroup(const std::string &name, SynapseMatrixType matrixType,
                 const WeightUpdateModels::Init &wumInitialiser, const PostsynapticModels::Init &psmInitialiser,
                 NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                 const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                 const InitToeplitzConnectivitySnippet::Init &toeplitzInitialiser,
                 VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                 VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    NeuronGroupInternal *getSrcNeuronGroup(){ return m_SrcNeuronGroup; }
    NeuronGroupInternal *getTrgNeuronGroup(){ return m_TrgNeuronGroup; }

    void setFusedPSTarget(const NeuronGroup *ng, const SynapseGroup &target);
    void setFusedSpikeTarget(const NeuronGroup *ng, const SynapseGroup &target);
    void setFusedSpikeEventTarget(const NeuronGroup *ng, const SynapseGroup &target);
    void setFusedWUPrePostTarget(const NeuronGroup *ng, const SynapseGroup &target);
    void setFusedPreOutputTarget(const NeuronGroup *ng, const SynapseGroup &target);
    
    void finalise(double dt);

    //! Add reference to custom connectivity update, referencing this synapse group
    void addCustomUpdateReference(CustomConnectivityUpdateInternal *cu){ m_CustomConnectivityUpdateReferences.push_back(cu); }

    //! Add reference to custom update, referencing this synapse group
    void addCustomUpdateReference(CustomUpdateWUInternal *cu){ m_CustomUpdateReferences.push_back(cu); }

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const NeuronGroupInternal *getSrcNeuronGroup() const{ return m_SrcNeuronGroup; }
    const NeuronGroupInternal *getTrgNeuronGroup() const{ return m_TrgNeuronGroup; }

    const SynapseGroup &getFusedPSTarget() const{ return m_FusedPSTarget ? *m_FusedPSTarget : *this; }
    const SynapseGroup &getFusedWUPreTarget() const { return m_FusedWUPreTarget ? *m_FusedPSTarget : *this; }
    const SynapseGroup &getFusedWUPostTarget() const { return m_FusedWUPostTarget ? *m_FusedWUPostTarget : *this; }
    const SynapseGroup &getFusedPreOutputTarget() const { return m_FusedPreOutputTarget ? *m_FusedPreOutputTarget : *this; }
    const SynapseGroup &getFusedSpikeTarget(const NeuronGroup *ng) const;
    const SynapseGroup &getFusedSpikeEventTarget(const NeuronGroup *ng) const;

    //! Gets custom connectivity updates which reference this synapse group
    /*! Because, if connectivity is sparse, all groups share connectivity this is required if connectivity changes. */
    const std::vector<CustomConnectivityUpdateInternal*> &getCustomConnectivityUpdateReferences() const{ return m_CustomConnectivityUpdateReferences; }

    //! Gets custom updates which reference this synapse group
    /*! Because, if connectivity is sparse, all groups share connectivity this is required if connectivity changes. */
    const std::vector<CustomUpdateWUInternal*> &getCustomUpdateReferences() const{ return m_CustomUpdateReferences; }
    
    //! Can postsynaptic update component of this synapse group be safely fused with others whose hashes match so only one needs simulating at all?
    bool canPSBeFused(const NeuronGroup *ng) const;
    
     //! Can presynaptic output component of this synapse group's weight update model be safely fused with other whose hashes match so only one needs simulating at all?
    bool canPreOutputBeFused(const NeuronGroup *ng) const;   
    
    //! Can spike generation for this synapse group be safely fused?
    bool canSpikeBeFused(const NeuronGroup*) const{ return true; }

    //! Can spike event generation for this synapse group be safely fused?
    bool canWUSpikeEventBeFused(const NeuronGroup *ng) const;

    //! Can presynaptic/postsynaptic update component of this synapse group's weight update model be safely fused with other whose hashes match so only one needs simulating at all?
    bool canWUMPrePostUpdateBeFused(const NeuronGroup *ng) const;

    //! Has this synapse group's postsynaptic model been fused with those from other synapse groups?
    bool isPSModelFused() const{ return m_FusedPSTarget != nullptr; }

    //! Has this synapse group's presynaptic spike generation been fused with those from other synapse groups?
    bool isPreSpikeFused() const{ return m_FusedPreSpikeTarget != nullptr; }

    //! Has this synapse group's postsynaptic spike generation been fused with those from other synapse groups?
    bool isPostSpikeFused() const{ return m_FusedPostSpikeTarget != nullptr; }

    //! Has this synapse group's presynaptic spike event generation been fused with those from other synapse groups?
    bool isPreSpikeEventFused() const{ return m_FusedPreSpikeEventTarget != nullptr; }

    //! Has this synapse group's postsynaptic spike event generation been fused with those from other synapse groups?
    bool isPostSpikeEventFused() const{ return m_FusedPostSpikeEventTarget != nullptr; }

    //! Has the presynaptic component of this synapse group's weight update
    //! model been fused with those from other synapse groups?
    bool isWUPreModelFused() const { return m_FusedWUPreTarget != nullptr; }

    //! Has the postsynaptic component of this synapse group's weight update
    //! model been fused with those from other synapse groups?
    bool isWUPostModelFused() const { return m_FusedWUPostTarget != nullptr; }

    //! Does this synapse group require dendritic delay?
    bool isDendriticDelayRequired() const;

    //! Does this synapse group provide presynaptic output?
    bool isPresynapticOutputRequired() const; 

    //! Does this synapse group provide postsynaptic output?
    bool isPostsynapticOutputRequired() const; 

    //! Does this synapse group require an RNG to generate procedural connectivity?
    bool isProceduralConnectivityRNGRequired() const;

    //! Does this synapse group require an RNG for it's weight update init code?
    bool isWUInitRNGRequired() const;

    //! Is var init code required for any variables in this synapse group's postsynaptic update model?
    bool isPSVarInitRequired() const;

    //! Is var init code required for any variables in this synapse group's weight update model?
    bool isWUVarInitRequired() const;

    //! Is var init code required for any presynaptic variables in this synapse group's weight update model?
    bool isWUPreVarInitRequired() const;

    //! Is var init code required for any presynaptic variables in this synapse group's weight update model?
    bool isWUPostVarInitRequired() const;

    //! Is sparse connectivity initialisation code required for this synapse group?
    bool isSparseConnectivityInitRequired() const;

    //! Is the presynaptic time variable with identifier referenced in weight update model?
    bool isPreTimeReferenced(const std::string &identifier) const;

    //! Is the postsynaptic time variable with identifier referenced in weight update model?
    bool isPostTimeReferenced(const std::string &identifier) const;

    //! Get the type to use for sparse connectivity indices for synapse group
    const Type::ResolvedType &getSparseIndType() const;

    //! Generate hash of weight update component of this synapse group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getWUHashDigest() const;

    //! Generate hash of presynaptic or postsynaptic update component of this synapse group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getWUPrePostHashDigest(const NeuronGroup *ng) const;

    //! Generate hash of postsynaptic update component of this synapse group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getPSHashDigest(const NeuronGroup *ng) const;

    //! Generate hash of presynaptic or postsynaptic spike generation component of this synapse group 
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getSpikeHashDigest(const NeuronGroup *ng) const;

    //! Generate hash of presynaptic or postsynaptic spike event generation component of this synapse group 
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getWUSpikeEventHashDigest(const NeuronGroup *ng) const;

    //! Generate hash of presynaptic or postsynaptic weight update component of this synapse group with additional components to ensure those
    //! with matching hashes can not only be simulated using the same code, but fused so only one needs simulating at all
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getWUPrePostFuseHashDigest(const NeuronGroup *ng) const;

    //! Generate hash of postsynaptic update component of this synapse group with additional components to ensure PSMs 
    //! with matching hashes can not only be simulated using the same code, but fused so only one needs simulating at all
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getPSFuseHashDigest(const NeuronGroup *ng) const;

    //! Generate hash of presynaptic or postsynaptic spike event generation of this synapse group with additional components to ensure PSMs 
    //! with matching hashes can not only be simulated using the same code, but fused so only one needs simulating at all
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getWUSpikeEventFuseHashDigest(const NeuronGroup *ng) const;

    boost::uuids::detail::sha1::digest_type getDendriticDelayUpdateHashDigest() const;

    //! Generate hash of initialisation component of this synapse group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getWUInitHashDigest() const;

    //! Generate hash of presynaptic variable initialisation component of this synapse group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getWUPreInitHashDigest() const;

    //! Generate hash of presynaptic or postsynaptic variable initialisation component of this synapse group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getWUPrePostInitHashDigest(const NeuronGroup *ng) const;

    //! Generate hash of postsynaptic model variable initialisation component of this synapse group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getPSInitHashDigest(const NeuronGroup *ng) const;

    //! Generate hash of presynaptic output initialization component of this synapse group 
     /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getPreOutputInitHashDigest(const NeuronGroup *ng) const;
    
    //! Generate hash of presynaptic output update component of this synapse group 
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getPreOutputHashDigest(const NeuronGroup *ng) const;

    //! Generate hash of connectivity initialisation of this synapse group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getConnectivityInitHashDigest() const;

    //! Generate hash of host connectivity initialisation of this synapse group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getConnectivityHostInitHashDigest() const;
    
    boost::uuids::detail::sha1::digest_type getVarLocationHashDigest() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //! Name of the synapse group
    const std::string m_Name;

    //! Hint as to how synapse group should be parallelised
    ParallelismHint m_ParallelismHint;

    //! How many threads CUDA implementation uses to process each spike when span type is PRESYNAPTIC
    unsigned int m_NumThreadsPerSpike;

    //! Global synaptic conductance delay for the group (in time steps)
    unsigned int m_AxonalDelaySteps;

    //! Global backpropagation delay for postsynaptic spikes to synapse (in time
    unsigned int m_BackPropDelaySteps;

    //! Maximum number of target neurons any source neuron can connect to
    unsigned int m_MaxConnections;
    
    //! Maximum number of source neurons any target neuron can connect to
    unsigned int m_MaxSourceConnections;

    //! Maximum dendritic delay timesteps supported for synapses in this population
    unsigned int m_MaxDendriticDelayTimesteps;

    //! Kernel size 
    std::vector<unsigned int> m_KernelSize;
    
    //! Connectivity type of synapses
    const SynapseMatrixType m_MatrixType;

    //! Pointer to presynaptic neuron group
    NeuronGroupInternal * const m_SrcNeuronGroup;

    //! Pointer to postsynaptic neuron group
    NeuronGroupInternal * const m_TrgNeuronGroup;

    //! Should narrow i.e. less than 32-bit types be used for sparse matrix indices
    bool m_NarrowSparseIndEnabled;

    //! Variable mode used for outputs from this synapse group e.g. outPre and outPost
    VarLocation m_OutputLocation;

    //! Variable mode used for this synapse group's dendritic delay buffers
    VarLocation m_DendriticDelayLocation;

    //! Initialiser used for creating weight update model
    WeightUpdateModels::Init m_WUInitialiser;

    //! Initialiser used for creating postsynaptic update model
    PostsynapticModels::Init m_PSInitialiser;

    //! Initialiser used for creating sparse connectivity
    InitSparseConnectivitySnippet::Init m_SparseConnectivityInitialiser;

    //! Initialiser used for creating toeplitz connectivity
    InitToeplitzConnectivitySnippet::Init m_ToeplitzConnectivityInitialiser;

    //! Location of individual per-synapse state variables
    LocationContainer m_WUVarLocation;

    //! Location of individual presynaptic state variables
    LocationContainer m_WUPreVarLocation;

    //! Location of individual postsynaptic state variables
    LocationContainer m_WUPostVarLocation;

    //! Location of weight update model extra global parameters
    LocationContainer m_WUExtraGlobalParamLocation;

    //! Whether indidividual state variables of post synapse should use zero-copied memory
    LocationContainer m_PSVarLocation;

    //! Location of postsynaptic model extra global parameters
    LocationContainer m_PSExtraGlobalParamLocation;

    //! Location of sparse connectivity
    VarLocation m_SparseConnectivityLocation;

    //! Data structure tracking whether postsynaptic model parameters are dynamic or not
    Snippet::DynamicParameterContainer m_PSDynamicParams;

    //! Data structure tracking whether weight update model parameters are dynamic or not
    Snippet::DynamicParameterContainer m_WUDynamicParams;

    //! Synapse group postsynaptic model has been fused with
    /*! If this is nullptr, postsynaptic model has not been fused */
    const SynapseGroup *m_FusedPSTarget;

    //! Synapse group presynaptic spike generation has been fused with
    /*! If this is nullptr, presynaptic spike generation has not been fused */
    const SynapseGroup *m_FusedPreSpikeTarget;

    //! Synapse group postsynaptic spike generation has been fused with
    /*! If this is nullptr, presynaptic spike generation has not been fused */
    const SynapseGroup *m_FusedPostSpikeTarget;

    //! Synapse group presynaptic spike event generation has been fused with
    /*! If this is nullptr, presynaptic spike event generation has not been fused */
    const SynapseGroup *m_FusedPreSpikeEventTarget;

    //! Synapse group postsynaptic spike event generation has been fused with
    /*! If this is nullptr, postsynaptic spike event generation has not been fused */
    const SynapseGroup *m_FusedPostSpikeEventTarget;

    //! Synapse group presynaptic weight update has been fused with
    /*! If this is nullptr, presynaptic weight update has not been fused */
    const SynapseGroup *m_FusedWUPreTarget;
    
    //! Synapse group postsynaptic weight update has been fused with
    /*! If this is nullptr, postsynaptic weight update  has not been fused */
    const SynapseGroup *m_FusedWUPostTarget;

    //! Synapse group presynaptic output has been fused with
    /*! If this is nullptr, presynaptic output has not been fused */
    const SynapseGroup *m_FusedPreOutputTarget;

    //! Name of neuron input variable postsynaptic model will target
    /*! This should either be 'Isyn' or the name of one of the postsynaptic neuron's additional input variables. */
    std::string m_PostTargetVar;

    //! Name of neuron input variable a presynaptic output specified with $(addToPre) will target
    /*! This will either be 'Isyn' or the name of one of the presynaptic neuron's additional input variables. */
    std::string m_PreTargetVar;

    //! Custom connectivity updates which reference this synapse group
    /*! Because, if connectivity is sparse, all groups share connectivity this is required if connectivity changes. */
    std::vector<CustomConnectivityUpdateInternal*> m_CustomConnectivityUpdateReferences;

    //! Custom updates which reference this synapse group
    /*! Because, if connectivity is sparse, all groups share connectivity this is required if connectivity changes. */
    std::vector<CustomUpdateWUInternal*> m_CustomUpdateReferences;
    
};
}   // namespace GeNN
