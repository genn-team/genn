#include "generateSynapseUpdate.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

// NuGeNN includes
#include "tempSubstitutions.h"
#include "substitution_stack.h"
#include "backends/base.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateSynapseUpdate(CodeStream &os, const NNmodel &model, const Backends::Base &backend)
{
    // Presynaptic update kernel
    backend.genSynapseUpdate(os, model,
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            // code substitutions ----
            const WeightUpdateModels::Base *wu = sg.getWUModel();
            std::string code = wu->getEventThresholdConditionCode();
            applyWeightUpdateModelSubstitutions(code, sg, backend.getVarPrefix(),
                                                sg.getName() + "[" + baseSubs.getVarSubstitution("id_syn") + "]", "");
            neuron_substitutions_in_synaptic_code(code, &sg, baseSubs.getVarSubstitution("id_pre"),
                                                  baseSubs.getVarSubstitution("id_post"), backend.getVarPrefix(),
                                                  model.getDT());
            baseSubs.apply(code);
            code= ensureFtype(code, model.getPrecision());
            checkUnreplacedVariables(code, sg.getName() + " : evntThreshold");
            os << code;
        },
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            const WeightUpdateModels::Base *wu = sg.getWUModel();
            std::string code = wu->getSimCode(); //**TODO** pass through truespikeness
            baseSubs.apply(code);

            applyWeightUpdateModelSubstitutions(code, sg, backend.getVarPrefix(),
                                                sg.getName() + "[" + baseSubs.getVarSubstitution("id_syn") + "]", "");
            neuron_substitutions_in_synaptic_code(code, &sg, baseSubs.getVarSubstitution("id_pre"),
                                                  baseSubs.getVarSubstitution("id_post"), backend.getVarPrefix(),
                                                  model.getDT());
            baseSubs.apply(code);
            code= ensureFtype(code, model.getPrecision());
            checkUnreplacedVariables(code, sg.getName() + " : simCode");
            os << code;
        }
    );
}