#include "code_generator/mergedRunnerMap.h"

// GeNN code generator includes
#include "code_generator/runnerGroupMerged.h"

//--------------------------------------------------------------------------
// CodeGenerator::MergedRunnerMap
//--------------------------------------------------------------------------
namespace CodeGenerator
{
std::string MergedRunnerMap::findGroup(const NeuronGroup &ng) const 
{ 
    return findGroup<NeuronRunnerGroupMerged>(ng.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::findGroup(const SynapseGroup &sg) const 
{ 
    return findGroup<SynapseRunnerGroupMerged>(sg.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::findGroup(const CurrentSource &cs) const 
{ 
    return findGroup<CurrentSourceRunnerGroupMerged>(cs.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::findGroup(const CustomUpdate &cu) const 
{ 
    return findGroup<CustomUpdateRunnerGroupMerged>(cu.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::findGroup(const CustomUpdateWU &cu) const
{ 
    return findGroup<CustomUpdateWURunnerGroupMerged>(cu.getName()); 
}
}