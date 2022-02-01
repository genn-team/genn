#include "code_generator/mergedRunnerMap.h"

// GeNN code generator includes
#include "code_generator/runnerGroupMerged.h"

//--------------------------------------------------------------------------
// CodeGenerator::MergedRunnerMap
//--------------------------------------------------------------------------
namespace CodeGenerator
{
std::string MergedRunnerMap::getStruct(const std::string &name) const
{
    // Find group by name
    const auto m = m_MergedRunnerGroups.at(name);

    // Return structure
    return "merged" + std::get<2>(m) + "Group" + std::to_string(std::get<0>(m)) + "[" + std::to_string(std::get<1>(m)) + "]";
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::getStruct(const NeuronGroup &ng) const 
{ 
    return getStruct(ng.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::getStruct(const SynapseGroup &sg) const 
{ 
    return getStruct(sg.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::getStruct(const CurrentSource &cs) const 
{ 
    return getStruct(cs.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::getStruct(const CustomUpdate &cu) const 
{ 
    return getStruct(cu.getName()); 
}
//--------------------------------------------------------------------------
std::string MergedRunnerMap::getStruct(const CustomUpdateWU &cu) const
{ 
    return getStruct(cu.getName()); 
}
}
