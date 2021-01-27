#pragma once

// GeNN includes
#include "customUpdate.h"

//------------------------------------------------------------------------
// CustomUpdateInternal
//------------------------------------------------------------------------
class CustomUpdateInternal : public CustomUpdate
{
public:
    CustomUpdateInternal(const std::string &name, const std::string &updateGroupName, Operation operation, 
                         const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params, 
                         const std::vector<Models::VarInit> &varInitialisers, const std::vector<Models::VarReference> &varReferences, 
                         VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   CustomUpdate(name, updateGroupName, operation, customUpdateModel, params, varInitialisers, varReferences, 
                     defaultVarLocation, defaultExtraGlobalParamLocation)
    {
    }

    using CustomUpdate::initDerivedParams;
    using CustomUpdate::getDerivedParams;
    using CustomUpdate::isInitRNGRequired;
    using CustomUpdate::canBeMerged;
    using CustomUpdate::canInitBeMerged;
};
