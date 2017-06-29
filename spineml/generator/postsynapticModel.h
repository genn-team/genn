#pragma once

#pragma once

// Standard includes
#include <set>
#include <string>

// GeNN includes
#include "newPostsynapticModels.h"

// Spine ML generator includes
#include "spineMLModelCommon.h"

// Forward declarations
namespace SpineMLGenerator
{
    class SpineMLNeuronModel;
    class SpineMLWeightUpdateModel;

    namespace ModelParams
    {
        class Postsynaptic;
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLPostsynapticModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class SpineMLPostsynapticModel : public PostsynapticModels::Base
{
public:
    SpineMLPostsynapticModel(const ModelParams::Postsynaptic &params,
                             const SpineMLNeuronModel *trgNeuronModel,
                             const SpineMLWeightUpdateModel *weightUpdateModel);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef SpineMLGenerator::ParamValues ParamValues;
    typedef SpineMLGenerator::VarValues VarValues;

    //------------------------------------------------------------------------
    // PostsynapticModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getDecayCode() const{ return m_DecayCode; }
    virtual std::string getCurrentConverterCode() const{ return m_CurrentConverterCode; }
    virtual NewModels::Base::StringVec getParamNames() const{ return m_ParamNames; }
    virtual NewModels::Base::StringPairVec getVars() const{ return m_Vars; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_DecayCode;
    std::string m_CurrentConverterCode;

    NewModels::Base::StringVec m_ParamNames;
    NewModels::Base::StringPairVec m_Vars;
};
}   // namespace SpineMLGenerator