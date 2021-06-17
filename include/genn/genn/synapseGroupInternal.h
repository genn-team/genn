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
                         const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<Models::VarInit> &wuVarInitialisers, const std::vector<Models::VarInit> &wuPreVarInitialisers, const std::vector<Models::VarInit> &wuPostVarInitialisers,
                         const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<Models::VarInit> &psVarInitialisers,
                         NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                         const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                         VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                         VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   SynapseGroup(name, matrixType, delaySteps, wu, wuParams, wuVarInitialisers, wuPreVarInitialisers, wuPostVarInitialisers,
                     ps, psParams, psVarInitialisers, srcNeuronGroup, trgNeuronGroup, weightSharingMaster,
                     connectivityInitialiser, defaultVarLocation, defaultExtraGlobalParamLocation,
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
    using SynapseGroup::setPSModelMergeTarget;
    using SynapseGroup::initDerivedParams;
    using SynapseGroup::isEventThresholdReTestRequired;
    using SynapseGroup::areWUVarReferencedByCustomUpdate;
    using SynapseGroup::getPSModelTargetName;
    using SynapseGroup::isPSModelMerged;
    using SynapseGroup::getSparseIndType;
    using SynapseGroup::canWUBeMerged;
    using SynapseGroup::canWUPreBeMerged;
    using SynapseGroup::canWUPostBeMerged;
    using SynapseGroup::canPSBeMerged;
    using SynapseGroup::canPSBeLinearlyCombined;
    using SynapseGroup::canWUInitBeMerged;
    using SynapseGroup::canWUPreInitBeMerged;
    using SynapseGroup::canWUPostInitBeMerged;
    using SynapseGroup::canPSInitBeMerged;
    using SynapseGroup::canConnectivityInitBeMerged;
    using SynapseGroup::canConnectivityHostInitBeMerged;
};
