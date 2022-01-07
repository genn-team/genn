#pragma once

// GeNN includes
#include "synapseGroup.h"

//------------------------------------------------------------------------
// SynapseGroupInternal
//------------------------------------------------------------------------
class SynapseGroupInternal : public SynapseGroup
{
public:
    SynapseGroupInternal(const std::string &name, const SynapseGroupInternal *weightSharingMaster, SynapseMatrixType matrixType, unsigned int delaySteps,
                         const WeightUpdateModels::Base *wu, const std::unordered_map<std::string, double> &wuParams, const std::vector<Models::VarInit> &wuVarInitialisers, const std::vector<Models::VarInit> &wuPreVarInitialisers, const std::vector<Models::VarInit> &wuPostVarInitialisers,
                         const PostsynapticModels::Base *ps, const std::unordered_map<std::string, double> &psParams, const std::vector<Models::VarInit> &psVarInitialisers,
                         NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                         const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                         const InitToeplitzConnectivitySnippet::Init &toeplitzConnectivityInitialiser,
                         VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                         VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   SynapseGroup(name, matrixType, delaySteps, wu, wuParams, wuVarInitialisers, wuPreVarInitialisers, wuPostVarInitialisers,
                     ps, psParams, psVarInitialisers, srcNeuronGroup, trgNeuronGroup, weightSharingMaster,
                     connectivityInitialiser, toeplitzConnectivityInitialiser, defaultVarLocation, defaultExtraGlobalParamLocation,
                     defaultSparseConnectivityLocation, defaultNarrowSparseIndEnabled)
    {
        // Add references to target and source neuron groups
        trgNeuronGroup->addInSyn(this);
        srcNeuronGroup->addOutSyn(this);
    }

    using SynapseGroup::getSrcNeuronGroup;
    using SynapseGroup::getTrgNeuronGroup;
    using SynapseGroup::getWeightSharingMaster;
    using SynapseGroup::getWUDerivedParams;
    using SynapseGroup::getPSDerivedParams;
    using SynapseGroup::setEventThresholdReTestRequired;
    using SynapseGroup::setWUVarReferencedByCustomUpdate;
    using SynapseGroup::setFusedPSVarSuffix;
    using SynapseGroup::setFusedPreOutputSuffix;
    using SynapseGroup::setFusedWUPreVarSuffix;
    using SynapseGroup::setFusedWUPostVarSuffix;
    using SynapseGroup::initDerivedParams;
    using SynapseGroup::isEventThresholdReTestRequired;
    using SynapseGroup::areWUVarReferencedByCustomUpdate;
    using SynapseGroup::getFusedPSVarSuffix;
    using SynapseGroup::getFusedPreOutputSuffix;
    using SynapseGroup::getFusedWUPreVarSuffix;
    using SynapseGroup::getFusedWUPostVarSuffix;
    using SynapseGroup::getSparseIndType;
    using SynapseGroup::canPSBeFused;
    using SynapseGroup::canWUMPreUpdateBeFused;
    using SynapseGroup::canWUMPostUpdateBeFused;
    using SynapseGroup::canPreOutputBeFused;
    using SynapseGroup::getWUHashDigest;
    using SynapseGroup::getWUPreHashDigest;
    using SynapseGroup::getWUPostHashDigest;
    using SynapseGroup::getPSHashDigest;
    using SynapseGroup::getPSFuseHashDigest;
    using SynapseGroup::getPreOutputHashDigest;
    using SynapseGroup::getWUPreFuseHashDigest;
    using SynapseGroup::getWUPostFuseHashDigest;
    using SynapseGroup::getDendriticDelayUpdateHashDigest;
    using SynapseGroup::getWUInitHashDigest;
    using SynapseGroup::getWUPreInitHashDigest;
    using SynapseGroup::getWUPostInitHashDigest;
    using SynapseGroup::getPSInitHashDigest;
    using SynapseGroup::getPreOutputInitHashDigest;
    using SynapseGroup::getConnectivityInitHashDigest;
    using SynapseGroup::getConnectivityHostInitHashDigest;
    using SynapseGroup::getVarLocationHashDigest;
};
