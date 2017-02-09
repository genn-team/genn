#pragma once

// GeNN includes
#include "newModels.h"
#include "postSynapseModels.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_DECAY_CODE(DECAY_CODE) virtual std::string GetDecayCode() const{ return DECAY_CODE; }
#define SET_CURRENT_CONVERTER_CODE(CURRENT_CONVERTER_CODE) virtual std::string GetCurrentConverterCode() const{ return CURRENT_CONVERTER_CODE; }
#define SET_SUPPORT_CODE(SUPPORT_CODE) virtual std::string GetSupportCode() const{ return SUPPORT_CODE; }

//----------------------------------------------------------------------------
// PostsynapticModels::Base
//----------------------------------------------------------------------------
namespace PostsynapticModels
{
class Base : public NewModels::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string GetDecayCode() const{ return ""; }
    virtual std::string GetCurrentConverterCode() const{ return ""; }
    virtual std::string GetSupportCode() const{ return ""; }
};

//----------------------------------------------------------------------------
// PostsynapticModels::LegacyWrapper
//----------------------------------------------------------------------------
class LegacyWrapper : public NewModels::LegacyWrapper<Base, postSynModel, postSynModels>
{
public:
    LegacyWrapper(unsigned int legacyTypeIndex) : NewModels::LegacyWrapper<Base, postSynModel, postSynModels>(legacyTypeIndex)
    {
    }

    //----------------------------------------------------------------------------
    // Base virtuals
    //----------------------------------------------------------------------------
    virtual std::string GetDecayCode() const;
    virtual std::string GetCurrentConverterCode() const;
    virtual std::string GetSupportCode() const;
};

}