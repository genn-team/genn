import numpy as np

from ._dvs import DVS, Polarity
from .genn_model import GeNNModel

class DVSMixin(object):
    """Mixin added to DVS objects
    It provides additional functionality for interfacing with GeNN models
    
    Attributes:
        pop:                 :class:`pygenn.NeuronGroup` that,
                             if spike tikes are required, will provide
                             interface for pushing, pulling and accessing them
    """
    
    def add_to_model(self, genn_model: GeNNModel, name: str = "DVS"):
        """Add a neuron population to a GeNN model 
        for interfacing with this DVS

        Args:
            genn_model: GeNN model to add DVS population to
            name:       Name to give DVS population    
        """
        # Create population using built in model
        self.pop = genn_model.add_neuron_population(
            name, self.output_width * self.output_height * self.output_channels,
            "EventCamera")
        
        # Create correctly sized spike vector
        self.pop.extra_global_params["spikeVector"].set_init_values(
            np.empty(self.output_array_words, dtype=np.uint32))
        
        # Return population
        return self.pop
    
    def copy_spikes(self):
        """Transfer spikes from DVS to GeNN neuron population
        """
        # Zero spike vector, read events into it and push to GPU
        spike_vector = self.pop.extra_global_params["spikeVector"]
        spike_vector.view[:] = 0
        self.read_events(spike_vector._array)
        spike_vector.push_to_device()

# Dynamically add Python mixin to wrapped class
DVS.__bases__ += (DVSMixin,)
