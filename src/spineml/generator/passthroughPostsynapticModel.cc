#include "passthroughPostsynapticModel.h"

// Standard C++ includes
#include <iostream>

// SpineML common includes
#include "spineMLLogging.h"

// Spine ML generator includes
#include "neuronModel.h"

//----------------------------------------------------------------------------
// SpineMLGenerator::PassthroughPostsynapticModel
//----------------------------------------------------------------------------
SpineMLGenerator::PassthroughPostsynapticModel::PassthroughPostsynapticModel(const std::string &trgPortName,
                                                                             const NeuronModel *trgNeuronModel)
{
    // If the target neuron model has a additional input var with the specified
    // name, create apply input code to add it to the neurons input
    if(trgNeuronModel->hasAdditionalInputVar(trgPortName)) {
        LOGD_SPINEML << "\t\tPassing through input to postsynaptic neuron port '" << trgPortName << "'";

        m_ApplyInputCode = trgPortName + " += $(inSyn); $(inSyn) = 0;\n";
    }
    else {
        throw std::runtime_error("Passthrough post synaptic models can only provide input to impulse receive ports");
    }
}
