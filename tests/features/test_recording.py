import numpy as np
import pytest
from pygenn import types

from pygenn import VarAccess, VarAccessMode

from pygenn import (create_neuron_model, create_var_ref, 
                    create_weight_update_model, init_postsynaptic,
                    init_weight_update)

# Neuron model which does nothing
empty_neuron_model = create_neuron_model("empty")
    
spike_event_source_array_model = create_neuron_model(
    "spike_event_source_array",
    sim_code=
    """
    if(startSpike != endSpike && t >= spikeTimes[startSpike]) {
        output = true;
        startSpike++;
    }
    else {
        output = false;
    }
    """,
    vars=[("startSpike", "unsigned int"), ("endSpike", "unsigned int", VarAccess.READ_ONLY_DUPLICATE),
                    ("output", "bool")],
    extra_global_params=[("spikeTimes", "scalar*")])

static_event_pulse_model = create_weight_update_model(
    "static_event_pulse",
    params=[("g", "scalar")],
    pre_neuron_var_refs=[("output", "bool", VarAccessMode.READ_ONLY)],
    pre_event_threshold_condition_code=
    """
    output
    """,
    pre_event_syn_code=
    """
    addToPost(g);
    """)

def compare_events(rec_events, event_times, event_ids):
    # Loop through batches
    for b, (batch_rec_event_times, batch_rec_event_ids) in enumerate(rec_events):
        # Re-order batch events to match spike source array order
        batch_rec_event_ordering = np.lexsort((batch_rec_event_times, batch_rec_event_ids))
        batch_rec_event_ids = batch_rec_event_ids[batch_rec_event_ordering]
        batch_rec_event_times = batch_rec_event_times[batch_rec_event_ordering]

        # Check that recorded events match input
        assert np.allclose(batch_rec_event_times, event_times[b])
        assert np.array_equal(batch_rec_event_ids, event_ids[b])


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_event_recording(make_model, backend, precision, batch_size):
    model = make_model(precision, "test_event_recording", backend=backend)
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

    # Add spike source to test spike recording
    ss = model.add_neuron_population("SpikeSource", 100, "SpikeSourceArray",
                                     {}, {"startSpike": ss_start_spikes, "endSpike": ss_end_spikes})
    ss.extra_global_params["spikeTimes"].set_init_values(np.concatenate(spike_times))
    ss.spike_recording_enabled = True

    # Add spike-event source
    es = model.add_neuron_population("SpikeEventSource", 100, spike_event_source_array_model,
                                     {}, {"startSpike": ss_start_spikes, "endSpike": ss_end_spikes, "output": False})
    es.extra_global_params["spikeTimes"].set_init_values(np.concatenate(spike_times))
    es.spike_event_recording_enabled = True

    # Because spike events are recorded per synapse group, add 
    post = model.add_neuron_population("Post", 1, empty_neuron_model)
    sg = model.add_synapse_population(
        "Synapses", "DENSE",
        es, post,
        init_weight_update(static_event_pulse_model, {"g": 1.0},
                           pre_var_refs={"output": create_var_ref(es, "output")}),
        init_postsynaptic("DeltaCurr"))

    # Build model and load
    model.build()
    model.load(num_recording_timesteps=100)

    # Simulate 100 timesteps
    while model.timestep < 100:
        model.step_time()

    # Download recording data and decode
    model.pull_recording_buffers_from_device()
    rec_spikes = ss.spike_recording_data
    rec_spike_events = sg.pre_spike_event_recording_data

    # Verify spikes and spike_events are recorded correctly
    compare_events(rec_spikes, spike_times, spike_ids)
    compare_events(rec_spike_events, spike_times, spike_ids)

@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_spike_recording_does_not_enable_event_recording(make_model, backend, precision, batch_size):
    model = make_model(precision, "test_spike_recording_no_event", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size

    ss_end_spikes = np.empty((batch_size, 10), dtype=int)
    ss_start_spikes = np.empty((batch_size, 10), dtype=int)

    spike_times = []
    id_offset = 0
    for b in range(batch_size):
        ids = np.arange(10)
        times = b + ids.astype(float)

        spike_times.append(times)

        ss_end_spikes[b, :] = id_offset + np.cumsum(np.ones(10, dtype=int))
        ss_start_spikes[b, 0] = id_offset
        ss_start_spikes[b, 1:] = ss_end_spikes[b, :-1]

        id_offset += 10

    # Spike source (recording ENABLED)
    ss = model.add_neuron_population(
        "SpikeSource", 10, "SpikeSourceArray",
        {}, {"startSpike": ss_start_spikes, "endSpike": ss_end_spikes}
    )
    ss.extra_global_params["spikeTimes"].set_init_values(np.concatenate(spike_times))
    ss.spike_recording_enabled = True

    # Spike-event source (recording NOT enabled)
    es = model.add_neuron_population(
        "SpikeEventSource", 10, spike_event_source_array_model,
        {}, {"startSpike": ss_start_spikes, "endSpike": ss_end_spikes, "output": False}
    )
    es.extra_global_params["spikeTimes"].set_init_values(np.concatenate(spike_times))

    post = model.add_neuron_population("Post", 1, empty_neuron_model)
    sg = model.add_synapse_population(
        "Synapses", "DENSE",
        es, post,
        init_weight_update(static_event_pulse_model, {"g": 1.0},
                           pre_var_refs={"output": create_var_ref(es, "output")}),
        init_postsynaptic("DeltaCurr")
    )

    model.build()
    model.load(num_recording_timesteps=20)

    while model.timestep < 20:
        model.step_time()

    model.pull_recording_buffers_from_device()

    # Spike recording should work
    _ = ss.spike_recording_data

    # Spike-event recording should NOT be enabled
    with pytest.raises(Exception):
        _ = sg.pre_spike_event_recording_data
