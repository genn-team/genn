import numpy as np
import pytest
from pygenn import types

from pygenn import GeNNModel

@pytest.mark.parametrize("backend, batch_size", [("single_threaded_cpu", 1), 
                                                 ("cuda", 1), ("cuda", 5)])
@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_spike_recording(backend, precision, batch_size):
    model = GeNNModel(precision, "test_spike_recording", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size
    
    # Loop through batches
    ss_end_spikes = np.empty((batch_size, 100), dtype=int)
    ss_start_spikes = np.empty((batch_size, 100), dtype=int)
    neuron_ids = np.arange(100)
    spike_ids = []
    spike_times = []
    id_offset = 0
    for b in range(batch_size):
        # Define input spike pattern with each neuron 
        # firing twice: once at b + id ms and once at b + 99 - id ms
        batch_spike_ids = np.tile(neuron_ids, 2)
        batch_spike_times = np.concatenate((b + neuron_ids, 
                                            b + 99.0 - neuron_ids))

        # Remove spikes that occur before or after simulation
        valid = (batch_spike_times >= 0.0) & (batch_spike_times < 100.0)
        batch_spike_ids = batch_spike_ids[valid]
        batch_spike_times = batch_spike_times[valid]

        # Sort spike IDs and times into spike source array order
        batch_spike_ordering = np.lexsort((batch_spike_times, batch_spike_ids))
        batch_spike_ids = batch_spike_ids[batch_spike_ordering]
        batch_spike_times = batch_spike_times[batch_spike_ordering]

        # Add batch spike ids and times to lists
        spike_ids.append(batch_spike_ids)
        spike_times.append(batch_spike_times)

        # Build start and end spike indices
        ss_end_spikes[b,:] = id_offset + np.cumsum(np.bincount(batch_spike_ids, minlength=100))
        ss_start_spikes[b,0] = id_offset
        ss_start_spikes[b,1:] = ss_end_spikes[b,:-1]
        
        # Advance offset
        id_offset += len(batch_spike_ids)

    ss = model.add_neuron_population("SpikeSource", 100, "SpikeSourceArray",
                                     {}, {"startSpike": ss_start_spikes, "endSpike": ss_end_spikes})
    ss.extra_global_params["spikeTimes"].set_init_values(np.concatenate(spike_times))
    ss.spike_recording_enabled = True
    
    # Build model and load
    model.build()
    model.load(num_recording_timesteps=100)
    
    # Simulate 100 timesteps
    while model.timestep < 100:
        model.step_time()
    
    # Download recording data and decode
    model.pull_recording_buffers_from_device()
    rec_spikes = ss.spike_recording_data

    # Loop through batches
    for b, (batch_rec_spike_times, batch_rec_spike_ids) in enumerate(rec_spikes):
        # Re-order batch spikes to match spike source array order
        batch_rec_spike_ordering = np.lexsort((batch_rec_spike_times, batch_rec_spike_ids))
        batch_rec_spike_ids = batch_rec_spike_ids[batch_rec_spike_ordering]
        batch_rec_spike_times = batch_rec_spike_times[batch_rec_spike_ordering]

        # Check that recorded spikes match input
        assert np.allclose(batch_rec_spike_times, spike_times[b])
        assert np.array_equal(batch_rec_spike_ids, spike_ids[b])

if __name__ == '__main__':
    test_spike_recording("cuda", types.Float, 1)
