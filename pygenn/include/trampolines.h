#pragma once

// PyBind11 includes
#include <pybind11/pybind11.h>

// GeNN includes
#include "models.h"
#include "snippet.h"

//----------------------------------------------------------------------------
// PySnippet
//----------------------------------------------------------------------------
template <class SnippetBase = Snippet::Base> 
class PySnippet : public SnippetBase 
{   
public:
    //using SnippetBase::SnippetBase;
    
    virtual Snippet::Base::StringVec getParamNames() const override{ PYBIND11_OVERRIDE_NAME(Snippet::Base::StringVec, SnippetBase, "get_param_names", getParamNames); }
    virtual Snippet::Base::DerivedParamVec getDerivedParams() const override{ PYBIND11_OVERRIDE_NAME(Snippet::Base::DerivedParamVec, SnippetBase, "get_derived_params", getDerivedParams); }
    virtual Snippet::Base::EGPVec getExtraGlobalParams() const override{ PYBIND11_OVERRIDE_NAME(Snippet::Base::EGPVec, SnippetBase, "get_extra_global_params", getExtraGlobalParams); }    
};

//----------------------------------------------------------------------------
// PyModel
//----------------------------------------------------------------------------
template <class ModelBase = Models::Base> 
class PyModel : public PySnippet<ModelBase> 
{
public:
    //using PyAnimal<DogBase>::PyAnimal;
    
    virtual Models::Base::VarVec getVars() const override{ PYBIND11_OVERRIDE_NAME(Models::Base::VarVec, ModelBase, "get_vars", getVars); }
};
