#pragma once

// Standard includes
#include <string>

// GeNN includes
#include "postsynapticModels.h"

// Forward declarations
namespace SpineMLGenerator
{
    class NeuronModel;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::PassthroughPostsynapticModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class PassthroughPostsynapticModel : public PostsynapticModels::Base
{
public:
    PassthroughPostsynapticModel(const std::string &trgPortName,
                                 const NeuronModel *trgNeuronModel);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef Snippet::ValueBase<0> ParamValues;
    typedef Models::VarInitContainerBase<0> VarValues;

    //------------------------------------------------------------------------
    // PostsynapticModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getApplyInputCode() const override{ return m_ApplyInputCode; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_ApplyInputCode;
};
}   // namespace SpineMLGenerator