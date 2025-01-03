#pragma once

// Standard C++ includes
#include <algorithm>

// GeNN code generator includes
#include "code_generator/groupMerged.h"

template<typename G>
bool hasField(const GeNN::CodeGenerator::GroupMerged<G> &group, const std::string &name)
{
    const auto fields = group.getFields();

    const auto f = std::find_if(fields.cbegin(), fields.cend(), [name](const auto &f){ return f.name == name; });
    return (f != fields.cend());
}