
#ifndef TESTPREVARSINSIMCODEEVNT_SPARSEINV_H
#define TESTPREVARSINSIMCODEEVNT_SPARSEINV_H

#define DT 0.1f
#define TOTAL_TIME 20.0f
#define REPORT_TIME 1.0f

class preVarsInSimCodeEvnt_sparseInv
{

public:
  preVarsInSimCodeEvnt_sparseInv();
  ~preVarsInSimCodeEvnt_sparseInv();
  void init_synapses();
  void init_neurons();
  void run(int);

  float **theW;
};

#endif // TESTPREVARSINSIMCODEEVNT_SPARSEINV_H
