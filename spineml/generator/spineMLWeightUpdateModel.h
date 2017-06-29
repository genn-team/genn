#pragma once

#pragma once

// Standard includes
#include <set>
#include <string>

// GeNN includes
#include "newWeightUpdateModels.h"

// Spine ML generator includes
#include "spineMLModelCommon.h"

// Forward declarations
namespace SpineMLGenerator
{
    namespace ModelParams
    {
        class WeightUpdate;
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLWeightUpdateModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class SpineMLWeightUpdateModel : public WeightUpdateModels::Base
{
public:
    SpineMLWeightUpdateModel(const ModelParams::WeightUpdate &params);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef SpineMLGenerator::ParamValues ParamValues;
    typedef SpineMLGenerator::VarValues VarValues;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const std::string &getSendPortSpikeImpulse() const
    {
        return m_SendPortSpikeImpulse;
    }

    //------------------------------------------------------------------------
    // PostsynapticModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getSimCode() const{ return m_SimCode; }
    virtual std::string getSynapseDynamicsCode() const{ return m_SynapseDynamicsCode; }

    virtual NewModels::Base::StringVec getParamNames() const{ return m_ParamNames; }
    virtual NewModels::Base::StringPairVec getVars() const{ return m_Vars; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_SimCode;
    std::string m_SynapseDynamicsCode;

    // How are send ports mapped to GeNN?
    std::string m_SendPortSpikeImpulse;

    NewModels::Base::StringVec m_ParamNames;
    NewModels::Base::StringPairVec m_Vars;
};
}   // namespace SpineMLGenerator