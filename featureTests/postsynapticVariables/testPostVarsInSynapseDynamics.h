
#ifndef TESTPREVARSINSYNAPSEDYNAMICS_H
#define TESTPREVARSINSYNAPSEDYNAMICS_H

#define DT 0.1f
#define TOTAL_TIME 20.0f
#define REPORT_TIME 1.0f

class postVarsInSynapseDynamics
{

public:
  postVarsInSynapseDynamics();
  ~postVarsInSynapseDynamics();
  void init_synapses();
  void init_neurons();
  void run(int);

  float **theW;
};

#endif // TESTPREVARSINSYNAPSEDYNAMICS_H
