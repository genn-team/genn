#include "code_generator/mergedRunnerMap.h"

// GeNN code generator includes
#include "code_generator/runnerGroupMerged.h"

//--------------------------------------------------------------------------
// CodeGenerator::MergedRunnerMap
//--------------------------------------------------------------------------
namespace CodeGenerator
{
std::string MergedRunnerMap::getStruct(const NeuronGroup &ng) const 
{ 
    return getStruct<NeuronRunnerGroupMerged>(ng.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::getStruct(const SynapseGroup &sg) const 
{ 
    return getStruct<SynapseRunnerGroupMerged>(sg.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::getStruct(const CurrentSource &cs) const 
{ 
    return getStruct<CurrentSourceRunnerGroupMerged>(cs.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::getStruct(const CustomUpdate &cu) const 
{ 
    return getStruct<CustomUpdateRunnerGroupMerged>(cu.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::getStruct(const CustomUpdateWU &cu) const
{ 
    return getStruct<CustomUpdateWURunnerGroupMerged>(cu.getName()); 
}
}
