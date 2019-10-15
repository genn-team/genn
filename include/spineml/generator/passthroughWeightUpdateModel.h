#pragma once

// Standard includes
#include <string>

// GeNN includes
#include "weightUpdateModels.h"

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
    PassthroughWeightUpdateModel(const std::string &srcPortName, const NeuronModel *srcNeuronModel,
                                 bool heterogeneousDelay);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef Snippet::ValueBase<0> ParamValues;
    typedef Models::VarInitContainerBase<0> VarValues;
    typedef Models::VarInitContainerBase<0> PreVarValues;
    typedef Models::VarInitContainerBase<0> PostVarValues;

    //------------------------------------------------------------------------
    // WeightUpdateModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getSimCode() const override{ return m_SimCode; }
    virtual std::string getSynapseDynamicsCode() const override{ return m_SynapseDynamicsCode; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_SimCode;
    std::string m_SynapseDynamicsCode;
};
}   // namespace SpineMLGenerator
