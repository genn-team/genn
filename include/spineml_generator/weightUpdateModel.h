#pragma once

// Standard includes
#include <set>
#include <string>

// GeNN includes
#include "weightUpdateModels.h"

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

namespace pugi
{
    class xml_node;
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
                      const pugi::xml_node &componentClass,
                      const NeuronModel *srcNeuronModel,
                      const NeuronModel *trgNeuronModel);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef SpineMLGenerator::ParamValues ParamValues;
    typedef SpineMLGenerator::VarValues<WeightUpdateModel> VarValues;
    typedef Models::VarInitContainerBase<0> PreVarValues;
    typedef Models::VarInitContainerBase<0> PostVarValues;

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

    unsigned int getInitialRegimeID() const
    {
        return m_InitialRegimeID;
    }

    //------------------------------------------------------------------------
    // WeightUpdateModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getSimCode() const override{ return m_SimCode; }
    virtual std::string getSynapseDynamicsCode() const override{ return m_SynapseDynamicsCode; }

    virtual Models::Base::StringVec getParamNames() const override{ return m_ParamNames; }
    virtual Models::Base::VarVec getVars() const override{ return m_Vars; }
    virtual Models::Base::DerivedParamVec getDerivedParams() const override{ return m_DerivedParams; }

    //------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------
    static const char *componentClassName;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_SimCode;
    std::string m_SynapseDynamicsCode;

    // How are send ports mapped to GeNN?
    std::string m_SendPortSpikeImpulse;
    std::string m_SendPortAnalogue;

    Models::Base::StringVec m_ParamNames;
    Models::Base::VarVec m_Vars;
    Models::Base::DerivedParamVec m_DerivedParams;

    unsigned int m_InitialRegimeID;
};
}   // namespace SpineMLGenerator
