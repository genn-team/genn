#pragma once

#pragma once

// Standard includes
#include <set>
#include <string>

// GeNN includes
#include "newWeightUpdateModels.h"

// Spine ML generator includes
#include "modelCommon.h"

// Forward declarations
namespace SpineMLGenerator
{
    class NeuronModel;

    namespace ModelParams
    {
        class WeightUpdate;
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::WeightUpdateModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class WeightUpdateModel : public WeightUpdateModels::Base
{
public:
    WeightUpdateModel(const ModelParams::WeightUpdate &params,
                      const NeuronModel *srcNeuronModel,
                      const NeuronModel *trgNeuronModel);

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

    const std::string &getSendPortAnalogue() const
    {
        return m_SendPortAnalogue;
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
    std::string m_SendPortAnalogue;

    NewModels::Base::StringVec m_ParamNames;
    NewModels::Base::StringPairVec m_Vars;
};
}   // namespace SpineMLGenerator