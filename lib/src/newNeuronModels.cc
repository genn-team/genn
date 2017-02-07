#include "newNeuronModels.h"

//----------------------------------------------------------------------------
// NeuronModels::Izhikevich
//----------------------------------------------------------------------------
NeuronModels::Izhikevich *NeuronModels::Izhikevich::s_Instance = NULL;
std::string NeuronModels::Izhikevich::s_SimCode =
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
std::string NeuronModels::Izhikevich::s_ThresholdConditionCode = "$(V) >= 29.99";
std::string NeuronModels::Izhikevich::s_ResetCode = "";
std::vector<std::string> NeuronModels::Izhikevich::s_ParamNames = {"a", "b", "c", "d"};
std::vector<std::pair<std::string, std::string>> NeuronModels::Izhikevich::s_InitVals = {{"V","scalar"}, {"U", "scalar"}};
//-------------------------------------------------------------------------------------------------------------------------------------