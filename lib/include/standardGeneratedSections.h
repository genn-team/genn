#pragma once

// Standard includes
#include <string>

// GeNN includes
#include "codeGenUtils.h"
#include "newNeuronModels.h"
#include "standardSubstitutions.h"

// Forward declarations
class NeuronGroup;

//----------------------------------------------------------------------------
// Functions to generate standard sections of code for use across backends
//----------------------------------------------------------------------------
namespace StandardGeneratedSections
{
void neuronOutputInit(
    std::ostream &os,
    const NeuronGroup &ng,
    const std::string &devPrefix);

void neuronLocalVarInit(
    std::ostream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &devPrefix,
    const std::string &localID);

void neuronLocalVarWrite(
    std::ostream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &devPrefix,
    const std::string &localID);

void neuronSpikeEventTest(
    std::ostream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &localID,
    const std::string &ftype);
}