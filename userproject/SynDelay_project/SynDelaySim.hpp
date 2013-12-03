
#ifndef SYNDELAYSIM_HPP
#define SYNDELAYSIM_HPP

#define DT 1.0f
#define TOTAL_TIME 5000.0f
#define REPORT_TIME 1000.0f

class SynDelay
{

private:
  float t;
  bool usingGPU;
  int sumInput;
  int sumInterneuron;
  int sumOutput;

public:
  SynDelay(bool usingGPU_arg);
  ~SynDelay();
  void run();

};

#endif // SYNDELAYSIM_HPP
