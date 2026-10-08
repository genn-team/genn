"""
This module provides functionality for streaming data from iniVation 
USB DVS cameras into GeNN models using libcaer on Linux and Mac. 
    
To use this functionality, please install libcaer using the instructions 
at https://gitlab.com/inivation/dv/libcaer to install libcaer 
before installing GeNN.

Instantiation
^^^^^^^^^^^^^
To connect a DVS to a GeNN model, you do this:

..  code-block:: python

    model = GeNNModel("float", "my_model")

    dvs = DVS.create_davis()
    dvs.add_to_model(model)

:meth:`DVS.create_dvs128` and :meth:`DVS.create_dvxplorer` also exist.

Usage
^^^^^
Spikes need to be copied from the DVS to the GeNN model
every timestep like this:

..  code-block:: python

    while True:
        dvs.copy_spikes()
        model.step_time()

If you need it elsewhere in your model, the _output_ resolution of the 
camera can be obtained with :attr:`DVS.output_width`, 
:attr:`DVS.output_height` and :attr:`DVS.output_channels`.

Advanced functionality
^^^^^^^^^^^^^^^^^^^^^^
Copying spikes from a DVS camera via the CPU is not particularly 
efficient so this class also implements various functionality for 
saving bandwidth:

Cropping
--------
A crop rectangle can be provided when you create the DVS object to 
filter out a region of interest:

..  code-block:: python

    dvs = DVS.create_davis(crop_rect=(0,0, 100, 100))

Scaling
-------
Even a moderate resolution translated into a very large number of neurons 
e.g. :math:`640 \\times 480 \\times 2 = 614400` so you can also spatially 
downsample the event camera output before it gets to the camera:

..  code-block:: python
    
    dvs = DVS.create_davis(scale=0.5)

"""

import numpy as np

from ._dvs import DVS, Polarity
from .genn_model import GeNNModel


class DVSMixin(object):
    """Mixin added to DVS objects to provide easier integration with PyGeNN
    
    Attributes:
        pop:    :class:`pygenn.NeuronGroup` that, if spike tikes are required,
                will provide interface for pushing, pulling and accessing them
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

__all__ = ["DVS", "DVSMixin", "Polarity"]
