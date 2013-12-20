
#ifndef SYNDELAYSIM_HPP
#define SYNDELAYSIM_HPP

#define DT 1.0f
#define TOTAL_TIME 5000.0f
#define REPORT_TIME 1000.0f

class SynDelay
{
private:
  bool usingGPU;

public:
  SynDelay(bool usingGPU);
  ~SynDelay();
  void run(float t);
};

#endif // SYNDELAYSIM_HPP
