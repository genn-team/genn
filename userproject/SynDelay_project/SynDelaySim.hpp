
#ifndef SYNDELAYSIM_HPP
#define SYNDELAYSIM_HPP

#define DT 1.0f
#define TOTAL_TIME 5000.0f
#define REPORT_TIME 1000.0f

class SynDelay
{

private:
  int which;
  float t;
  int sumInputIzh;
  int sumOutputIzh;

public:
  SynDelay(int whichArg);
  ~SynDelay();
  void run();

};

#endif // SYNDELAYSIM_HPP
