
#ifndef TESTEGSPINSIMCODEEVNT_SPARSEINV_H
#define TESTEGSPINSIMCODEEVNT_SPARSEINV_H

#define DT 0.1f
#define TOTAL_TIME 20.0f
#define REPORT_TIME 1.0f

class EGSPInSimCodeEvnt_sparseInv
{

public:
  EGSPInSimCodeEvnt_sparseInv();
  ~EGSPInSimCodeEvnt_sparseInv();
  void init_synapses();
  void init_neurons();
  void run(int);

  float **theW;
  float **theThresh;
};

#endif // TESTEGSPINSIMCODEEVNT_SPARSEINV_H
