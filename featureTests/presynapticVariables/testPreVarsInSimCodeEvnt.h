
#ifndef TESTPREVARSINSIMCODEEVNT_H
#define TESTPREVARSINSIMCODEEVNT_H

#define DT 0.1f
#define TOTAL_TIME 20.0f
#define REPORT_TIME 1.0f

class preVarsInSimCodeEvnt
{

public:
  preVarsInSimCodeEvnt();
  ~preVarsInSimCodeEvnt();
  void init_synapses();
  void init_neurons();
  void run(int);

  float **theW;
};

#endif // TESTPREVARSINSIMCODEEVNT_H
