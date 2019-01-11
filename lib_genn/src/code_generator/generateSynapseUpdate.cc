#include "code_generator/generateSynapseUpdate.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "modelSpec.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/tempSubstitutions.h"
#include "code_generator/substitutions.h"
#include "code_generator/backendBase.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void applySynapseSubstitutions(CodeGenerator::CodeStream &os, std::string code, const std::string &errorSuffix, const SynapseGroup &sg,
                               const CodeGenerator::Substitutions &baseSubs, const NNmodel &model, const CodeGenerator::BackendBase &backend)
{
    using namespace CodeGenerator;
    applyWeightUpdateModelSubstitutions(code, sg, backend.getVarPrefix(),
                                        sg.getName() + "[" + baseSubs.getVarSubstitution("id_syn") + "]", "");
    neuronSubstitutionsInSynapticCode(code, &sg, baseSubs.getVarSubstitution("id_pre"),
                                      baseSubs.getVarSubstitution("id_post"), backend.getVarPrefix(),
                                      model.getDT());
    baseSubs.apply(code);
    code= ensureFtype(code, model.getPrecision());
    checkUnreplacedVariables(code, sg.getName() + errorSuffix);
    os << code;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateSynapseUpdate(CodeStream &os, const NNmodel &model, const BackendBase &backend)
{
    os << "#include \"definitions.h\"" << std::endl;

    // Synaptic update kernels
    backend.genSynapseUpdate(os, model,
        // Presynaptic weight update threshold
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getWUModel()->getEventThresholdConditionCode(), " : evntThreshold",
                                      sg, baseSubs, model, backend);
        },
        // Presynaptic simcode
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            applySynapseSubstitutions(os, sg.getWUModel()->getSimCode(), " : simCode",
                                      sg, baseSubs, model, backend);
        },
        // Postsynaptic learning code
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            if (!sg.getWUModel()->getLearnPostSupportCode().empty()) {
                os << " using namespace " << sg.getName() << "_weightupdate_simLearnPost;" << std::endl;
            }

            applySynapseSubstitutions(os, sg.getWUModel()->getLearnPostCode(), " : simLearnPost",
                                      sg, baseSubs, model, backend);
        },
        // Synapse dynamics
        [&backend, &model](CodeStream &os, const SynapseGroup &sg, const Substitutions &baseSubs)
        {
            if (!sg.getWUModel()->getSynapseDynamicsSuppportCode().empty()) {
                os << " using namespace " << sg.getName() << "_weightupdate_synapseDynamics;" << std::endl;
            }

            applySynapseSubstitutions(os, sg.getWUModel()->getSynapseDynamicsCode(), " : synapseDynamics",
                                      sg, baseSubs, model, backend);
        }
    );
}