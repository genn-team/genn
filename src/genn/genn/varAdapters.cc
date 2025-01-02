#include "varAdapters.h"

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
//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarAdapter
//----------------------------------------------------------------------------
const Runtime::ArrayBase *CustomConnectivityUpdateVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_CU, varName);
}

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdapter
//----------------------------------------------------------------------------
const Runtime::ArrayBase *CustomConnectivityUpdatePreVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_CU, varName);
}

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePostVarAdapter
//----------------------------------------------------------------------------
const Runtime::ArrayBase *CustomConnectivityUpdatePostVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_CU, varName);
}

//----------------------------------------------------------------------------
// NeuronVarAdapter
//----------------------------------------------------------------------------
const Runtime::ArrayBase *NeuronVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_NG, varName);
}

//----------------------------------------------------------------------------
// SynapsePSMVarAdapter
//----------------------------------------------------------------------------
const Runtime::ArrayBase *SynapsePSMVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_SG.getFusedPSTarget(), varName);
}

//----------------------------------------------------------------------------
// SynapseWUVarAdapter
//----------------------------------------------------------------------------
const Runtime::ArrayBase *SynapseWUVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_SG, varName);
}

//----------------------------------------------------------------------------
// SynapseWUPreVarAdapter
//----------------------------------------------------------------------------
const Runtime::ArrayBase *SynapseWUPreVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_SG.getFusedWUPreTarget(), varName);
}

//----------------------------------------------------------------------------
// SynapseWUPostVarAdapter
//----------------------------------------------------------------------------
const Runtime::ArrayBase *SynapseWUPostVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_SG.getFusedWUPostTarget(), varName);
}

//----------------------------------------------------------------------------
// CustomUpdateVarAdapter
//----------------------------------------------------------------------------
const Runtime::ArrayBase *CustomUpdateVarAdapter::getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const
{
    return runtime.getArray(m_CU, varName);
}
}