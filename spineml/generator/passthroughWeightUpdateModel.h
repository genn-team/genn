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
    typedef NewModels::ValueBase<0> ParamValues;
    typedef NewModels::ValueBase<0> VarValues;

    //------------------------------------------------------------------------
    // WeightUpdateModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getSynapseDynamicsCode() const override{ return m_SynapseDynamicsCode; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_SynapseDynamicsCode;
};
}   // namespace SpineMLGenerator