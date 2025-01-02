#pragma once

// Standard C++ includes
#include <map>
#include <optional>
#include <string>
#include <vector>

// GeNN includes
#include "models.h"
#include "varLocation.h"

//----------------------------------------------------------------------------
// GeNN::VarAdapterBase
//----------------------------------------------------------------------------
namespace GeNN
{
class VarAdapterBase
{
public:
    virtual ~VarAdapterBase() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &name) const = 0;
   
    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const = 0;

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const = 0;
};

//----------------------------------------------------------------------------
// GeNN::VarAdapter
//----------------------------------------------------------------------------
class VarAdapter : public VarAdapterBase
{
public:
    virtual ~VarAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Models::Base::Var> getDefs() const = 0;
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const = 0;
};

//----------------------------------------------------------------------------
// GeNN::CUVarAdapter
//----------------------------------------------------------------------------
class CUVarAdapter : public VarAdapterBase
{
public:
    virtual ~CUVarAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Models::Base::CustomUpdateVar> getDefs() const = 0;
    virtual VarAccessDim getVarDims(const Models::Base::CustomUpdateVar &var) const = 0;
};

//----------------------------------------------------------------------------
// GeNN::VarRefAdapterBase
//----------------------------------------------------------------------------
class VarRefAdapterBase
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals 
    //------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const = 0;
};

//----------------------------------------------------------------------------
// GeNN::VarRefAdapter
//----------------------------------------------------------------------------
class VarRefAdapter : public VarRefAdapterBase
{
public:
    virtual ~VarRefAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const = 0;

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varRefName) const = 0;
};

//----------------------------------------------------------------------------
// GeNN::WUVarRefAdapter
//----------------------------------------------------------------------------
class WUVarRefAdapter : public VarRefAdapterBase
{
public:
    virtual ~WUVarRefAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual const std::map<std::string, Models::WUVarReference> &getInitialisers() const = 0;
};

//----------------------------------------------------------------------------
// GeNN::EGPAdapter
//----------------------------------------------------------------------------
class EGPAdapter
{
public:
    virtual ~EGPAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &name) const = 0;
    
    virtual Snippet::Base::EGPVec getDefs() const = 0;
};
}