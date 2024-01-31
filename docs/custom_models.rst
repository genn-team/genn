.. py:currentmodule:: pygenn

=============
Custom models
=============
--------
GeNNCode
--------

Random number generation
------------------------
Random numbers are useful in many forms of custom model. For example as a source of noise or a probabilistic spiking mechanism. 
In GeNN this can be implemented by using the following functions within GeNNCode:

- ``gennrand_uniform()`` returns a number drawn uniformly from the interval :math:`[0.0, 1.0]`
- ``gennrand_normal()`` returns a number drawn from a normal distribution with a mean of 0 and a standard deviation of 1.
- ``gennrand_exponential()`` returns a number drawn from an exponential distribution with \f$\lambda=1\f$.
- ``gennrand_log_normal, MEAN, STDDEV)`` returns a number drawn from a log-normal distribution with the specified mean and standard deviation.
- ``gennrand_gamma, ALPHA)`` returns a number drawn from a gamma distribution with the specified shape.
- ``gennrand_binomial, N, P)`` returns a number drawn from a binomial distribution with the specified shape.

Literals
--------
0.0d 0.0f 0.0

-----
Types
-----
pass

---------------
Variable access
---------------
pass

-----------------------
Initialisation snippets
-----------------------

Variable initialisation
-----------------------
pass

Sparse connectivity initialisation
----------------------------------
pass

Toeplitz connectivity initialisation
------------------------------------
pass

------
Models
------

Neuron models
-------------
In order to define a new neuron type, it is necessary to define a new class derived from NeuronModels::Base.
However, for convenience, convenience, 

.. autofunction:: pygenn.create_neuron_model

For example, we can define a leaky integrator :math:`\tau\frac{dV}{dt}= -V + I_{{\rm syn}}` solved using Euler's method:

..  code-block:: python

    leaky_integrator_model = pygenn.create_neuron_model(
        "leaky_integrator",

        sim_code="V += (-V + Isyn) * (dt / tau);",
        threshold_condition_code="V >= 1.0",
        reset_code="V = 0.0;",

        params=["tau"],
        vars=[("V", "scalar", pygenn.VarAccess.READ_WRITE)])


Derived parameters
^^^^^^^^^^^^^^^^^^
'Derived parameters' are a mechanism for enhanced efficiency when running neuron models. 
If parameters with model-side meaning, such as time constants or conductances always appear in a certain combination in the model, then it is more efficient to pre-compute this combination and define it as a dependent parameter.

For example, because the equation defining the previous leaky integrator example has an algebraic solution, it can be more accurately solved as follows - using a derived parameter to calculate :math:`\exp\left(\frac{-t}{\tau}\right)`:

..  code-block:: python

    leaky_integrator_2_model = pygenn.create_neuron_model(
        "leaky_integrator_2",

        sim_code="V = Isyn - ExpTC * (Isyn - V);",
        threshold_condition_code="V >= 1.0",
        reset_code="V = 0.0;",

        params=["tau"],
        vars=[("V", "scalar", VarAccess_READ_WRITE)],
        derived_params=[("ExpTC", lambda pars, dt: np.exp(-dt / pars["tau"]))])

Additional input variables
^^^^^^^^^^^^^^^^^^^^^^^^^^
Normally, neuron models receive the linear sum of the inputs coming from all of their synaptic inputs through the ``Isyn`` variable. 
However neuron models can define additional input variables - allowing input from different synaptic inputs to be combined non-linearly.
For example, if we wanted our leaky integrator to operate on the the product of two input currents, we could modify our model as follows:


..  code-block:: python

    additional_input_vars=[("Isyn2", "scalar", 1.0)],
    sim_code=
        """
        const scalar input = Isyn * Isyn2;
        V = input - ExpTC * (input - V);
        """,


Weight update models
--------------------

.. autofunction:: pygenn.create_weight_update_model

For example, we can define a simple additive STDP rule with nearest-neighbour spike pairing and the following time-dependence:

..  math::

    \Delta w_{ij} & = \
        \begin{cases}
            A_{+}\exp\left(-\frac{\Delta t}{\tau_{+}}\right) & if\, \Delta t>0\\
            A_{-}\exp\left(\frac{\Delta t}{\tau_{-}}\right) & if\, \Delta t\leq0
        \end{cases}

in a fully event-driven manner as follows:

..  code-block:: python

    stdp_additive_model = pygenn.create_weight_update_model(
        "stdp_additive",
        params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
        vars=[("g", "scalar")],

        pre_spike_syn_code="""
            addToPost(g);
            const scalar dt = t - st_post;
            if (dt > 0) {
                const scalar timing = exp(-dt / tauMinus);
                const scalar newWeight = g - (Aminus * timing);
                g = fmax(Wmin, fmin(Wmax, newWeight));
            }
            """,
        post_spike_syn_code="""
            const scalar dt = t - st_pre;
            if (dt > 0) {
                const scalar timing = exp(-dt / tauPlus);
                const scalar newWeight = g + (Aplus * timing);
                g = fmax(Wmin, fmin(Wmax, newWeight));
            }
            """)

Pre and postsynaptic dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The memory required for synapse variables and the computational cost of updating them tends to grow with \f$O(N^2)\f$ with the number of neurons.
Therefore, if it is possible, implementing synapse variables on a per-neuron rather than per-synapse basis is a good idea. 
The \add_cpp_python_text{SET_PRE_VARS() and SET_POST_VARS() macros, `pre_var_name_types` and `post_var_name_types` keyword arguments} are used to define any pre or postsynaptic state variables.
Then the \add_cpp_python_text{SET_PRE_DYNAMICS_CODE() and SET_POST_DYNAMICS_CODE() macros, `pre_dynamics_code` and `post_dynamics_code` keyword arguments} can be used to define any time-driven updates to these variables and the \add_cpp_python_text{SET_PRE_SPIKE_CODE() and SET_POST_SPIKE_CODE() macros, `pre_spike_code` and `post_spike_code` keyword arguments} any spike-driven updates.
For example, using pre and postsynaptic variables, our event-driven STDP rule can be extended to use all-to-all spike pairing using pre and postsynaptic <i>trace</i> variables \cite Morrison2008 :
\note
These pre and postsynaptic code snippets can only access the corresponding pre and postsynaptic variables as well as those associated with the pre or postsynaptic neuron population. Like other state variables, variables defined here as `NAME` can be accessed in weight update model code strings using the \$(NAME) syntax. 

..  code-block:: python

    stdp_additive_2_model = genn_model.create_custom_weight_update_class(
        "stdp_additive_2",
        params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
        vars=[("g", "scalar")],
        pre_vars=[("preTrace", "scalar")],
        post_vars=[("postTrace", "scalar")],
        
        pre_spike_syn_code="""
            addToPost(g);
            const scalar dt = t - st_post;
            if(dt > 0) {
                const scalar newWeight = g - (aMinus * postTrace);
                g = fmin(wMax, fmax(wMin, newWeight));
            }
            """,
        post_spike_syn_code="""
            const scalar dt = t - st_pre;
            if(dt > 0) {
                const scalar newWeight = g + (aPlus * preTrace);
                g = fmin(wMax, fmax(wMin, newWeight));
            }
            """,

        pre_spike_code="""
            preTrace += 1.0;
            """,
        pre_dynamics_code="""
            preTrace *= tauPlusDecay;
            """,
        post_spike_code="""
            postTrace += 1.0;
            """,
        post_dynamics_code="""
            postTrace *= tauMinusDecay;
            """)

Synapse dynamics
^^^^^^^^^^^^^^^^
Unlike the event-driven updates previously described, synapse dynamics code is run for each synapse, each timestep i.e. unlike the others it is time-driven. 
This can be used where synapses have internal variables and dynamics that are described in continuous time, e.g. by ODEs.
However using this mechanism is typically computationally very costly because of the large number of synapses in a typical network. 
By using the ``addToPost()`` and ``addToPostDelay()`` functions discussed in the context of ``pre_spike_syn_code``, the synapse dynamics can also be used to implement continuous synapses for rate-based models.
For example a continous synapse could be added to a weight update model definition as follows:

..  code-block:: python

    synapse_dynamics_code="addToPost(g * V_pre);",

where ``V_pre`` is a presynaptic variable reference.

Spike-like events
^^^^^^^^^^^^^^^^^
As well as time-driven synapse dynamics and spike event-driven updates, GeNN weight update models also support "spike-like events". These are triggerd by a threshold condition -- implemented with the code string specified using the  \add_cpp_python_text{SET_EVENT_THRESHOLD_CONDITION_CODE() macro,`event_threshold_condition_code` keyword argument}. This typically involves the pre-synaptic variables, e.g. the membrane potential: 
\add_toggle_code_cpp
SET_EVENT_THRESHOLD_CONDITION_CODE("$(V_pre) > -0.02");
\end_toggle_code
\add_toggle_code_python
event_threshold_condition_code="$(V_pre) > -0.02"
\end_toggle_code
Whenever this expression evaluates to true, the event code set using the \add_cpp_python_text{SET_EVENT_CODE() macro,`event_code` keyword argument} is executed. For an example, see WeightUpdateModels::StaticGraded.
Weight update models can indicate whether they require the times of these spike-like-events using the \add_cpp_python_text{SET_NEEDS_PRE_SPIKE_EVENT_TIME() and SET_NEEDS_PREV_PRE_SPIKE_EVENT_TIME() macros, ``is_pre_spike_event_time_required`` and ``is_prev_pre_spike_event_time_required`` keyword arguments}.
These times can then be accessed through the \$(seT_pre) and \$(prev_seT_pre) variables.

Postsynaptic update models
--------------------------
pass

Current source models
---------------------
pass

Custom update models
---------------------
pass

Custom connectivity update models
---------------------------------
pass
