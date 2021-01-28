#include "customUpdate.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"
#include "currentSource.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//------------------------------------------------------------------------
// CustomUpdateBase
//------------------------------------------------------------------------
void CustomUpdateBase::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
void CustomUpdateBase::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    const size_t extraGlobalParamIndex = getCustomUpdateModel()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(getCustomUpdateModel()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_ExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateBase::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)];
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateBase::getExtraGlobalParamLocation(const std::string &varName) const
{
    return m_ExtraGlobalParamLocation[getCustomUpdateModel()->getExtraGlobalParamIndex(varName)];
}
void CustomUpdateBase::initDerivedParams(double dt)
{
    auto derivedParams = getCustomUpdateModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.func(getParams(), dt));
    }

    // Initialise derived parameters for variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.initDerivedParams(dt);
    }
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isInitRNGRequired() const
{
    // If initialising the neuron variables require an RNG, return true
    if(Utils::isRNGRequired(getVarInitialisers())) {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isZeroCopyEnabled() const
{
    // If there are any variables implemented in zero-copy mode return true
    return std::any_of(m_VarLocation.begin(), m_VarLocation.end(),
                       [](VarLocation loc) { return (loc & VarLocation::ZERO_COPY); });
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::canBeMerged(const CustomUpdateBase &other) const
{
    return getCustomUpdateModel()->canBeMerged(other.getCustomUpdateModel());
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::canInitBeMerged(const CustomUpdateBase &other) const
{
     // If both groups have the same number of variables
    if(getVarInitialisers().size() == other.getVarInitialisers().size()) {
        // if any of the variable's initialisers can't be merged, return false
        for(size_t i = 0; i < getVarInitialisers().size(); i++) {
            if(!getVarInitialisers()[i].canBeMerged(other.getVarInitialisers()[i])) {
                return false;
            }
        }
        
        return true;
    }
    else {
        return false;
    }
}

//----------------------------------------------------------------------------
CustomUpdateWU::CustomUpdateWU(const std::string &name, const std::string &updateGroupName, Operation operation,
                   const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                   const std::vector<Models::VarInit> &varInitialisers, const std::vector<WUVarReference> &varReferences,
                   VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, defaultVarLocation, defaultExtraGlobalParamLocation),
    m_Operation(operation), m_VarReferences(varReferences)
{
    // Loop through all variable references
    // **TODO** move to helper function
    const auto varRefs = getCustomUpdateModel()->getVarRefs();
    for(size_t i = 0; i < varReferences.size(); i++) {
        const auto varRef = m_VarReferences.at(i);

        // Check types of variable references against those specified in model
        // **THINK** due to GeNN's current string-based type system this is rather conservative
        if(varRef.getVar().type != varRefs.at(i).type) {
            throw std::runtime_error("Incompatible type for variable reference '" + getCustomUpdateModel()->getVarRefs().at(i).name + "'");
        }
    }

    // If there are any variable references
    /*if(!m_VarReferences.empty()) {
        // If this is a weight update variable custom update
        if(wuCustomUpdate) {
            const SynapseGroupInternal *firstSG = static_cast<const SynapseGroupInternal *>(m_VarReferences.front().getSynapseGroup());
            const size_t preSize = firstSG->getSrcNeuronGroup()->getNumNeurons();
            const size_t maxRowLength = firstSG->getMaxConnections();
            const bool sparse = (firstSG->getMatrixType() & SynapseMatrixConnectivity::SPARSE);

            for(const auto &v : m_VarReferences) {
                // Check that connectivity types match
                const SynapseGroupInternal *sg =  static_cast<const SynapseGroupInternal*>(v.getSynapseGroup());
                if((sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) != sparse) {
                    throw std::runtime_error("Variable references to weight update model variables must all be to populations with the same connectivity type.");
                }
                
                // Check that presynaptic population size and maximum row lengths match
                if(sg->getSrcNeuronGroup()->getNumNeurons() != preSize || sg->getMaxConnections() != maxRowLength) {
                    throw std::runtime_error("Variable references must all be to populations of the same size.");
                }

                // Check that, if connectivity is sparse, all referenced variables belong to the same population
                // **NOTE** this is because sparse connectivity on two differnt populations 
                // might have the same max row length but totally different connectivity
                if(sparse && firstSG != sg) {
                    throw std::runtime_error("Variable references to sparse weight update model variables must all be in the same synapse group.");
                }
            }
        }
    }*/

    // If this is a transpose operation
    if(m_Operation == Operation::UPDATE_TRANSPOSE) {
        // Give error if any of the variable references aren't dense
        if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [](const WUVarReference &v) { return !(v.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE); }))
        {
            throw std::runtime_error("Custom updates that perform a transpose operation can currently only be used on DENSE synaptic matrices.");
        }
    }
}