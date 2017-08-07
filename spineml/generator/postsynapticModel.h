#pragma once

#pragma once

// Standard includes
#include <set>
#include <string>

// GeNN includes
#include "newPostsynapticModels.h"

// Spine ML generator includes
#include "modelCommon.h"

// Forward declarations
namespace SpineMLGenerator
{
    class NeuronModel;
    class WeightUpdateModel;

    namespace ModelParams
    {
        class Postsynaptic;
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::PostsynapticModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class PostsynapticModel : public PostsynapticModels::Base
{
public:
    PostsynapticModel(const ModelParams::Postsynaptic &params,
                      const NeuronModel *trgNeuronModel,
                      const WeightUpdateModel *weightUpdateModel);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef SpineMLGenerator::ParamValues ParamValues;
    typedef SpineMLGenerator::VarValues VarValues;

    //------------------------------------------------------------------------
    // PostsynapticModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getDecayCode() const{ return m_DecayCode; }
    virtual std::string getApplyInputCode() const{ return m_ApplyInputCode; }
    virtual std::string getUpdateLinSynCode() const{ return m_UpdateLinSynCode; }

    virtual NewModels::Base::StringVec getParamNames() const{ return m_ParamNames; }
    virtual NewModels::Base::StringPairVec getVars() const{ return m_Vars; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_DecayCode;
    std::string m_ApplyInputCode;
    std::string m_UpdateLinSynCode;

    NewModels::Base::StringVec m_ParamNames;
    NewModels::Base::StringPairVec m_Vars;
};
}   // namespace SpineMLGenerator