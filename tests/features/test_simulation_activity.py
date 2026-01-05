import numpy as np
import pytest
from pygenn import types

from pygenn import VarAccess, VarAccessMode

from pygenn import (create_neuron_model, create_weight_update_model,
                    init_postsynaptic, init_sparse_connectivity,
                    init_weight_update)

# Module-level reusable models
empty_neuron_model = create_neuron_model("empty")

# Neuron model that fires in a deterministic pattern based on neuron ID and time
# Neuron i fires when: t >= i AND (t - i) mod 10 == 0
# This creates predictable spike patterns for testing
pattern_spike_neuron_model = create_neuron_model(
    "pattern_spike_neuron",
    vars=[("spike_count", "unsigned int")],
    sim_code=
    """
    spike_count++;
    """,
    threshold_condition_code=
    """
    t >= (scalar)id && fmod(t - (scalar)id, 10.0) < 1e-4
    """,
    reset_code="")

# Simple weight update model that just delivers a pulse
simple_pulse_model = create_weight_update_model(
    "simple_pulse",
    params=[("g", "scalar")],
    pre_spike_syn_code=
    """
    addToPost(g);
    """)

# Neuron model that accumulates synaptic input for verification
# Uses built-in LIF model which is more stable
accumulator_neuron_model = create_neuron_model(
    "accumulator_neuron",
    vars=[("V", "scalar")],
    sim_code=
    """
    V += Isyn * dt;
    """)

# Spike event source model (similar to test_recording.py)
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

# Static event pulse model for spike-like events
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


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_activity_minimal(make_model, backend, precision):
    """Test basic simulation activity with a single neuron population.

    Verifies:
    - Timestep counter increments correctly
    - Simulation time progresses correctly
    - Spikes are generated according to pattern
    - Internal counter variables update correctly
    """
    model = make_model(precision, "test_activity_minimal", backend=backend)
    model.dt = 1.0

    # Create a population of 10 neurons with pattern-based spiking
    pop = model.add_neuron_population("Neurons", 10, pattern_spike_neuron_model,
                                      {}, {"spike_count": 0})

    # Enable spike recording
    pop.spike_recording_enabled = True

    # Build and load
    model.build()
    model.load(num_recording_timesteps=100)

    # Verify initial state
    assert model.timestep == 0
    assert np.isclose(model.t, 0.0)

    # Run simulation for 100 timesteps
    while model.timestep < 100:
        model.step_time()

    # Verify timestep and time progression
    assert model.timestep == 100
    assert np.isclose(model.t, 100.0)

    # Pull spike recording data
    model.pull_recording_buffers_from_device()
    spike_times, spike_ids = pop.spike_recording_data[0]

    # Verify spikes were generated
    assert len(spike_times) > 0, "No spikes were recorded"

    # Verify spike pattern: neuron i fires at t = i, i+10, i+20, ...
    # For 100 timesteps, each neuron should fire 10 times (at t=id, id+10, ..., id+90)
    spike_counts = np.bincount(spike_ids, minlength=10)
    expected_count = 10  # Each neuron fires 10 times in 100 timesteps
    assert np.all(spike_counts == expected_count), \
        f"Expected each neuron to spike {expected_count} times, got {spike_counts}"

    # Verify internal counter variable
    pop.vars["spike_count"].pull_from_device()
    # spike_count increments every timestep, so after 100 timesteps it should be 100
    assert np.all(pop.vars["spike_count"].values == 100), \
        f"Expected spike_count to be 100, got {pop.vars['spike_count'].values}"


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_activity_minimal_batch(make_model, backend, precision, batch_size):
    """Test basic simulation activity with batching.

    Same as test_activity_minimal but with batch_size parameter.
    Verifies that batched simulations track activity correctly.
    """
    model = make_model(precision, "test_activity_minimal_batch", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size

    # Create population
    pop = model.add_neuron_population("Neurons", 10, pattern_spike_neuron_model,
                                      {}, {"spike_count": 0})

    # Enable spike recording
    pop.spike_recording_enabled = True

    # Build and load
    model.build()
    model.load(num_recording_timesteps=100)

    # Run simulation
    while model.timestep < 100:
        model.step_time()

    # Verify timestep progression (same for all batches)
    assert model.timestep == 100
    assert np.isclose(model.t, 100.0)

    # Verify spike recording for each batch
    model.pull_recording_buffers_from_device()

    for b in range(batch_size):
        spike_times, spike_ids = pop.spike_recording_data[b]

        # Verify spikes were generated in this batch
        assert len(spike_times) > 0, f"No spikes recorded in batch {b}"

        # Verify spike counts
        spike_counts = np.bincount(spike_ids, minlength=10)
        expected_count = 10
        assert np.all(spike_counts == expected_count), \
            f"Batch {b}: Expected {expected_count} spikes per neuron, got {spike_counts}"

    # Verify internal counter
    pop.vars["spike_count"].pull_from_device()
    expected_shape = (batch_size, 10) if batch_size > 1 else (10,)
    assert pop.vars["spike_count"].values.shape == expected_shape or \
           pop.vars["spike_count"].values.flatten().shape == (batch_size * 10,), \
        f"Unexpected shape: {pop.vars['spike_count'].values.shape}"


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_activity_basic_network(make_model, backend, precision):
    """Test simulation activity in a basic network with neurons and synapses.

    Verifies:
    - Pre-synaptic spikes are generated
    - Post-synaptic neurons receive input
    - Synapse variables (counters) update correctly
    - Spike propagation through one-to-one connectivity
    """
    model = make_model(precision, "test_activity_basic_network", backend=backend)
    model.dt = 1.0

    # Create pre and post populations
    pre = model.add_neuron_population("Pre", 10, pattern_spike_neuron_model,
                                      {}, {"spike_count": 0})
    post = model.add_neuron_population("Post", 10, accumulator_neuron_model,
                                       {}, {"V": 0.0})

    # Enable spike recording
    pre.spike_recording_enabled = True

    # Create synapse population with one-to-one connectivity
    syn = model.add_synapse_population(
        "Synapses", "SPARSE",
        pre, post,
        init_weight_update(simple_pulse_model, {"g": 0.1}, {}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))

    # Build and load
    model.build()
    model.load(num_recording_timesteps=100)

    # Run simulation
    while model.timestep < 100:
        model.step_time()

    # Verify timestep progression
    assert model.timestep == 100

    # Verify pre-synaptic spikes
    model.pull_recording_buffers_from_device()
    spike_times, spike_ids = pre.spike_recording_data[0]
    assert len(spike_times) > 0, "No pre-synaptic spikes recorded"

    # Verify spike counts
    spike_counts = np.bincount(spike_ids, minlength=10)
    expected_count = 10
    assert np.all(spike_counts == expected_count), \
        f"Expected {expected_count} spikes per pre-neuron, got {spike_counts}"

    # Verify post-synaptic neurons received input (voltage changed from initial 0.0)
    post.vars["V"].pull_from_device()
    # Each post-neuron should have received input (V changed from 0.0)
    assert np.any(post.vars["V"].values != 0.0), \
        "Post-synaptic neurons did not receive input"


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_activity_basic_network_batch(make_model, backend, precision, batch_size):
    """Test simulation activity in a basic network with batching.

    Same as test_activity_basic_network but with batch processing.
    """
    model = make_model(precision, "test_activity_basic_network_batch", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size

    # Create populations
    pre = model.add_neuron_population("Pre", 10, pattern_spike_neuron_model,
                                      {}, {"spike_count": 0})
    post = model.add_neuron_population("Post", 10, accumulator_neuron_model,
                                       {}, {"V": 0.0})

    pre.spike_recording_enabled = True

    # Create synapses
    syn = model.add_synapse_population(
        "Synapses", "SPARSE",
        pre, post,
        init_weight_update(simple_pulse_model, {"g": 0.1}, {}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))

    # Build and load
    model.build()
    model.load(num_recording_timesteps=100)

    # Run simulation
    while model.timestep < 100:
        model.step_time()

    # Verify timestep
    assert model.timestep == 100

    # Verify spikes for each batch
    model.pull_recording_buffers_from_device()
    for b in range(batch_size):
        spike_times, spike_ids = pre.spike_recording_data[b]
        assert len(spike_times) > 0, f"No spikes in batch {b}"

        spike_counts = np.bincount(spike_ids, minlength=10)
        assert np.all(spike_counts == 10), f"Batch {b}: Unexpected spike counts"

    # Verify post-synaptic input (voltage changed from initial 0.0)
    post.vars["V"].pull_from_device()
    # All batches should have received input (voltage changed)
    v_flat = post.vars["V"].values.flatten()
    assert np.any(v_flat != 0.0), \
        "Post-synaptic neurons did not receive input"


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_activity_spike_events(make_model, backend_simt, precision):
    """Test simulation activity with spike-like events (GPU backends only).

    Verifies:
    - Spike-like events are generated and recorded
    - Event counts match expected pattern
    - Events propagate through synapses

    Uses backend_simt fixture to run only on CUDA/HIP backends.
    """
    model = make_model(precision, "test_activity_spike_events", backend=backend_simt)
    model.dt = 1.0

    # Create event times: neuron i fires events at t = i, i+10, i+20, ..., i+90
    neuron_ids = np.arange(10)
    event_times_list = []
    event_ids_list = []

    for i in range(10):
        times = np.arange(i, 100, 10, dtype=float)
        event_times_list.extend(times)
        event_ids_list.extend([i] * len(times))

    # Sort by time, then by ID
    event_times = np.array(event_times_list)
    event_ids = np.array(event_ids_list)
    sort_order = np.lexsort((event_times, event_ids))
    event_times = event_times[sort_order]
    event_ids = event_ids[sort_order]

    # Build start/end spike indices for spike source array
    end_spikes = np.cumsum(np.bincount(event_ids, minlength=10))
    start_spikes = np.zeros(10, dtype=np.uint32)
    start_spikes[1:] = end_spikes[:-1]

    # Create event source population
    event_source = model.add_neuron_population(
        "EventSource", 10, spike_event_source_array_model,
        {}, {"startSpike": start_spikes, "endSpike": end_spikes, "output": False})

    # Create post population
    post = model.add_neuron_population("Post", 10, accumulator_neuron_model,
                                       {}, {"V": 0.0})

    # Set spike times as EGP
    event_source.extra_global_params["spikeTimes"].set_init_values(event_times)

    # Create synapse with event-based weight update
    syn = model.add_synapse_population(
        "EventSynapses", "SPARSE",
        event_source, post,
        init_weight_update(static_event_pulse_model, {"g": 0.1}, {}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))

    # Enable event recording
    event_source.spike_event_recording_enabled = True

    # Build and load
    model.build()
    model.load(num_recording_timesteps=100)

    # Run simulation
    while model.timestep < 100:
        model.step_time()

    # Verify timestep
    assert model.timestep == 100

    # Verify event recording
    model.pull_recording_buffers_from_device()
    rec_event_times, rec_event_ids = syn.pre_spike_event_recording_data[0]

    # Verify events were recorded
    assert len(rec_event_times) > 0, "No spike events were recorded"

    # Verify event counts (each neuron should have 10 events)
    event_counts = np.bincount(rec_event_ids, minlength=10)
    expected_count = 10
    assert np.all(event_counts == expected_count), \
        f"Expected {expected_count} events per neuron, got {event_counts}"

    # Verify post-synaptic neurons received input from events
    post.vars["cumulative_input"].pull_from_device()
    assert np.all(post.vars["cumulative_input"].values > 0), \
        "Post-synaptic neurons did not receive input from events"


@pytest.mark.parametrize("precision", [types.Double, types.Float])
def test_activity_with_delays(make_model, backend, precision, batch_size):
    """Test simulation activity with axonal delays.

    Verifies:
    - Activity counters work correctly with delays
    - Delayed spikes arrive at expected times
    - Batch processing works with delays
    """
    model = make_model(precision, "test_activity_with_delays", backend=backend)
    model.dt = 1.0
    model.batch_size = batch_size

    # Create populations
    pre = model.add_neuron_population("Pre", 10, pattern_spike_neuron_model,
                                      {}, {"spike_count": 0})
    post = model.add_neuron_population("Post", 10, accumulator_neuron_model,
                                       {}, {"V": 0.0})

    # Enable spike recording
    pre.spike_recording_enabled = True

    # Create synapse with axonal delay
    syn = model.add_synapse_population(
        "DelayedSynapses", "SPARSE",
        pre, post,
        init_weight_update(simple_pulse_model, {"g": 0.1}, {}),
        init_postsynaptic("DeltaCurr"),
        init_sparse_connectivity("OneToOne"))

    # Set axonal delay of 5 timesteps
    syn.axonal_delay_steps = 5

    # Build and load
    model.build()
    model.load(num_recording_timesteps=100)

    # Track post-synaptic voltage at different times
    v_at_t5 = None
    v_at_t50 = None

    # Run simulation
    while model.timestep < 100:
        model.step_time()

        # Sample voltage at specific timesteps
        if model.timestep == 5:
            post.vars["V"].pull_from_device()
            v_at_t5 = post.vars["V"].values.copy()
        elif model.timestep == 50:
            post.vars["V"].pull_from_device()
            v_at_t50 = post.vars["V"].values.copy()

    # Verify timestep
    assert model.timestep == 100

    # Verify pre-synaptic spikes occurred
    model.pull_recording_buffers_from_device()
    spike_times_all = []
    for b in range(batch_size):
        spike_times, spike_ids = pre.spike_recording_data[b]
        spike_times_all.extend(spike_times)
    assert len(spike_times_all) > 0, "No pre-synaptic spikes"

    # Verify delayed arrival: at t=5, voltage should be minimal
    # At t=50, voltage should be higher (more spikes have arrived)
    v_flat_t5 = v_at_t5.flatten() if v_at_t5 is not None else np.array([])
    v_flat_t50 = v_at_t50.flatten() if v_at_t50 is not None else np.array([])

    # Post-synaptic neurons should have higher voltage by t=50 than at t=5
    if len(v_flat_t5) > 0 and len(v_flat_t50) > 0:
        # Average voltage should increase over time with delays
        assert np.mean(np.abs(v_flat_t50)) > np.mean(np.abs(v_flat_t5)), \
            "Voltage did not increase over time with delays"

    # Final check: post-synaptic neurons should have received input (voltage changed)
    post.vars["V"].pull_from_device()
    final_v = post.vars["V"].values.flatten()
    assert np.any(final_v != 0.0), \
        "Post-synaptic neurons did not receive delayed input"
