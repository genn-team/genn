#include "passthroughPostsynapticModel.h"

// Standard C++ includes
#include <iostream>

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
        std::cout << "\t\tPassing through input to postsynaptic neuron port '" << trgPortName << "'" << std::endl;

        m_ApplyInputCode = trgPortName + " += $(inSyn);\n";
    }
    else {
        throw std::runtime_error("Passthrough post synaptic models can only provide input to impulse receive ports");
    }
}