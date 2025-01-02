#pragma once

// GeNN includes
#include "gennExport.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::BackendBase
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class GENN_EXPORT BackendCLike : public BackendBase
{
public:
    using BackendBase::BackendBase;
};
}