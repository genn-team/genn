#include "newNeuronModels.h"

//----------------------------------------------------------------------------
// NeuronModels::Izhikevich
//----------------------------------------------------------------------------
NeuronModels::Izhikevich::s_SimCode =
"    if ($(V) >= 30.0){\n"
"      $(V)=$(c);\n"
"                  $(U)+=$(d);\n"
"    } \n"
"    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability\n"
"    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;\n"
"    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
"   //if ($(V) > 30.0){   //keep this only for visualisation -- not really necessaary otherwise \n"
"   //  $(V)=30.0; \n"
"   //}\n";
NeuronModels::Izhikevich::s_ThresholdConditionCode = "$(V) >= 29.99";
NeuronModels::Izhikevich::s_ParamNames[] = {"a", "b", "c", "d"};
NeuronModels::Izhikevich::s_InitValueNames[] = {"V", "U"};
NeuronModels::Izhikevich::s_InitValueTypes[] = {"scalar", "scalar"};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// NeuronModels::IzhikevichV
//----------------------------------------------------------------------------
NeuronModels::IzhikevichV::s_SimCode =
"    if ($(V) >= 30.0){\n"
"      $(V)=$(c);\n"
"                  $(U)+=$(d);\n"
"    } \n"
"    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset))*DT; //at two times for numerical stability\n
"    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset))*DT;\n"
"    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
"    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
"   //if ($(V) > 30.0){   //keep this only for visualisation -- not really necessaary otherwise \n"
"   //  $(V)=30.0; \n"
"   //}\n";
NeuronModels::IzhikevichV::s_ThresholdConditionCode = "$(V) >= 29.99";
NeuronModels::IzhikevichV::s_ParamNames[] = {"Ioffset"};
NeuronModels::IzhikevichV::s_InitValueNames[] = {"V", "U", "a", "b", "c", "d"};
NeuronModels::IzhikevichV::s_InitValueTypes[] = {"scalar", "scalar", "scalar", "scalar", "scalar", "scalar"};
//----------------------------------------------------------------------------