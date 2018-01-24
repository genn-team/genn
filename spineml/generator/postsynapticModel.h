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

namespace pugi
{
    class xml_node;
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
                      const pugi::xml_node &componentClass,
                      const NeuronModel *trgNeuronModel,
                      const WeightUpdateModel *weightUpdateModel);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef SpineMLGenerator::ParamValues ParamValues;
    typedef SpineMLGenerator::VarValues<PostsynapticModel> VarValues;

    //------------------------------------------------------------------------
    // PostsynapticModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getDecayCode() const{ return m_DecayCode; }
    virtual std::string getApplyInputCode() const{ return m_ApplyInputCode; }
    virtual NewModels::Base::StringVec getParamNames() const{ return m_ParamNames; }
    virtual NewModels::Base::StringPairVec getVars() const{ return m_Vars; }
    virtual NewModels::Base::DerivedParamVec getDerivedParams() const{ return m_DerivedParams; }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    unsigned int getInitialRegimeID() const
    {
        return m_InitialRegimeID;
    }

    //------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------
    static const char *componentClassName;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_DecayCode;
    std::string m_ApplyInputCode;

    NewModels::Base::StringVec m_ParamNames;
    NewModels::Base::StringPairVec m_Vars;
    NewModels::Base::DerivedParamVec m_DerivedParams;

    unsigned int m_InitialRegimeID;
};
}   // namespace SpineMLGenerator