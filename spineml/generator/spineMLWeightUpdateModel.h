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
    class ModelParams;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLWeightUpdateModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class SpineMLWeightUpdateModel : public WeightUpdateModels::Base
{
public:
    SpineMLWeightUpdateModel(const ModelParams &params);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef SpineMLGenerator::ParamValues ParamValues;
    typedef SpineMLGenerator::VarValues VarValues;

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

    NewModels::Base::StringVec m_ParamNames;
    NewModels::Base::StringPairVec m_Vars;
};
}   // namespace SpineMLGenerator