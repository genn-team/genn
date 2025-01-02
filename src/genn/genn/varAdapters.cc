#include "adapters.h"

// GeNN runtime includes
#include "runtime/runtime.h"

//----------------------------------------------------------------------------
// CurrentSourceVarAdapter
//----------------------------------------------------------------------------
namespace GeNN
{
const Runtime::ArrayBase *CurrentSourceVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_CS, varName);
}

}