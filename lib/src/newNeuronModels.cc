#include "newNeuronModels.h"

//----------------------------------------------------------------------------
// NeuronModels::Izhikevich
//----------------------------------------------------------------------------
IMPLEMENT_NEURON(NeuronModels::Izhikevich,
  "    if ($(V) >= 30.0){\n"
  "      $(V)=$(c);\n"
  "                  $(U)+=$(d);\n"
  "    } \n"
  "    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability\n"
  "    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;\n"
  "    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
  "   //if ($(V) > 30.0){   //keep this only for visualisation -- not really necessaary otherwise \n"
  "   //  $(V)=30.0; \n"
  "   //}\n",
  "$(V) >= 29.99",
  "",
  ARRAY_PROTECT({"a", "b", "c", "d"}),
  ARRAY_PROTECT({{"V","scalar"}, {"U", "scalar"}}));
//-------------------------------------------------------------------------------------------------------------------------------------