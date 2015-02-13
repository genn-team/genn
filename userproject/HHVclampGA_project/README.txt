
  Genetic algorithm for tracking parameters in a HH model cell
  =============================================

This example simulates a population of HH neuron models on the GPU and evolves them with a simple 
guided random search (simple GA) to mimic the dynamics of a separate HH
neuron that is simulated on the CPU. The parameters of the CPU simulated "true cell" are drifting 
according to a user-chosen protocol: Either one of the parameters gNa, ENa, gKd, EKd, gleak,
Eleak, Cmem are modified by a sinusoidal addition (voltage parameters) or factor (conductance or capacitance) - 
protocol 0-6. For protocol 7 all 7 parameters undergo a random walk concurrently.

  USAGE
  -----

generate_run <CPU=0, GPU=1> <protocol> <nPop> <totalT> <outdir> <debug mode? (0/1)> <GPU choice>

An example invocation of generate_run is:

./generate_run 1 -1 12 200000 test1 0 0

This will simulate nPop= 5000 HH neurons on the GPU which will for 1000 ms be matched to a HH neuron where the parameter
gKd is sinusoidally modulated. The output files will be written into a directory of the name test1_output, which will
be created if it does not yet exist.
