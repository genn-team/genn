#pragma once

// GeNN includes
#include "neuronGroupInternal.h"
#include "synapseGroup.h"

//------------------------------------------------------------------------
// GeNN::SynapseGroupInternal
//------------------------------------------------------------------------
namespace GeNN
{
class SynapseGroupInternal : public SynapseGroup
{
public:
    using GroupExternal = SynapseGroup;

    SynapseGroupInternal(const std::string &name, SynapseMatrixType matrixType,
                         const WeightUpdateModels::Init &wumInitialiser, const PostsynapticModels::Init &psmInitialiser,
                         NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                         const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                         const InitToeplitzConnectivitySnippet::Init &toeplitzConnectivityInitialiser,
                         VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                         VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   SynapseGroup(name, matrixType, wumInitialiser, psmInitialiser,
                     srcNeuronGroup, trgNeuronGroup, connectivityInitialiser, 
                     toeplitzConnectivityInitialiser, defaultVarLocation, defaultExtraGlobalParamLocation,
                     defaultSparseConnectivityLocation, defaultNarrowSparseIndEnabled)
    {
        // Add references to target and source neuron groups
        trgNeuronGroup->addInSyn(this);
        srcNeuronGroup->addOutSyn(this);
    }

    using SynapseGroup::getSrcNeuronGroup;
    using SynapseGroup::getTrgNeuronGroup;
    using SynapseGroup::setFusedPSTarget;
    using SynapseGroup::setFusedSpikeTarget;
    using SynapseGroup::setFusedSpikeEventTarget;
    using SynapseGroup::setFusedPreOutputTarget;
    using SynapseGroup::setFusedWUPrePostTarget;
    using SynapseGroup::finalise;
    using SynapseGroup::addCustomUpdateReference;
    using SynapseGroup::getFusedPSTarget;
    using SynapseGroup::getFusedSpikeTarget;
    using SynapseGroup::getFusedSpikeEventTarget;
    using SynapseGroup::getFusedPreOutputTarget;
    using SynapseGroup::getFusedWUPreTarget;
    using SynapseGroup::getFusedWUPostTarget;
    using SynapseGroup::getSparseIndType;
    using SynapseGroup::getCustomConnectivityUpdateReferences;
    using SynapseGroup::getCustomUpdateReferences;
    using SynapseGroup::canPSBeFused;
    using SynapseGroup::canSpikeBeFused;
    using SynapseGroup::canWUMPrePostUpdateBeFused;
    using SynapseGroup::canWUSpikeEventBeFused;
    using SynapseGroup::canPreOutputBeFused;
    using SynapseGroup::isPSModelFused;
    using SynapseGroup::isPreSpikeFused;
    using SynapseGroup::isWUPreModelFused;
    using SynapseGroup::isWUPostModelFused;
    using SynapseGroup::isDendriticOutputDelayRequired;
    using SynapseGroup::isWUPostVarHeterogeneouslyDelayed;
    using SynapseGroup::areAnyWUPostVarHeterogeneouslyDelayed;
    using SynapseGroup::isPresynapticOutputRequired; 
    using SynapseGroup::isPostsynapticOutputRequired;
    using SynapseGroup::isProceduralConnectivityRNGRequired;
    using SynapseGroup::isWUInitRNGRequired;
    using SynapseGroup::isPSVarInitRequired;
    using SynapseGroup::isWUVarInitRequired;
    using SynapseGroup::isWUPreVarInitRequired;
    using SynapseGroup::isWUPostVarInitRequired;
    using SynapseGroup::isSparseConnectivityInitRequired;
    using SynapseGroup::getWUHashDigest;
    using SynapseGroup::getWUPrePostHashDigest;
    using SynapseGroup::getWUSpikeEventHashDigest;
    using SynapseGroup::getPSHashDigest;
    using SynapseGroup::getPSFuseHashDigest;
    using SynapseGroup::getSpikeHashDigest;
    using SynapseGroup::getPreOutputHashDigest;
    using SynapseGroup::getWUPrePostFuseHashDigest;
    using SynapseGroup::getWUSpikeEventFuseHashDigest;
    using SynapseGroup::getDendriticDelayUpdateHashDigest;
    using SynapseGroup::getWUInitHashDigest;
    using SynapseGroup::getWUPrePostInitHashDigest;
    using SynapseGroup::getPSInitHashDigest;
    using SynapseGroup::getPreOutputInitHashDigest;
    using SynapseGroup::getConnectivityInitHashDigest;
    using SynapseGroup::getConnectivityHostInitHashDigest;
    using SynapseGroup::getVarLocationHashDigest;
};
}   // namespace GeNN
