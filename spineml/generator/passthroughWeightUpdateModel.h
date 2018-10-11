#pragma once

// Standard includes
#include <string>

// GeNN includes
#include "newWeightUpdateModels.h"

// Forward declarations
namespace SpineMLGenerator
{
    class NeuronModel;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::PassThroughWeightUpdateModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class PassthroughWeightUpdateModel : public WeightUpdateModels::Base
{
public:
    PassthroughWeightUpdateModel(const std::string &srcPortName,
                                 const NeuronModel *srcNeuronModel);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef Snippet::ValueBase<0> ParamValues;
    typedef NewModels::VarInitContainerBase<0> VarValues;
    typedef NewModels::VarInitContainerBase<0> PreVarValues;
    typedef NewModels::VarInitContainerBase<0> PostVarValues;

    //------------------------------------------------------------------------
    // WeightUpdateModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getSimCode() const{ return m_SimCode; }
    virtual std::string getSynapseDynamicsCode() const override{ return m_SynapseDynamicsCode; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_SimCode;
    std::string m_SynapseDynamicsCode;
};
}   // namespace SpineMLGenerator