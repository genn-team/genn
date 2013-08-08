/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

using namespace std;

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "randomGen.h"
#include "randomGen.cc"

typedef float real;

randomGen R;

int main(int argc, char *argv[])
{
  if (argc != 9)
  {
    cerr << "usage: gen_input_structured ";
    cerr << "<nAL> <# classes> <# pattern/ input class> ";
    cerr << "<prob. to be active> <perturbation prob. in class> ";
    cerr << "<'on' rate> <baseline rate> ";
    cerr << "<outfile> ";
    cerr << endl;
    exit(1);
  }

  int nNo= atoi(argv[1]);
  int classNo= atoi(argv[2]);
  int single_pNo= atoi(argv[3]);
  real pact= atof(argv[4]);
  real pperturb= atof(argv[5]);
  unsigned int lambdaOn= atoi(argv[6]);
  unsigned int lambdaOff= atoi(argv[7]);
  int patternNo= single_pNo*classNo;
  int pat[nNo], patp[nNo];
  int nact= (int) (pact*nNo);
  int patns[nact];
  int npert= (int) (pperturb*nact);
  int theno, newno;
  ofstream os(argv[8], ios::binary);
  unsigned int *fpt= new unsigned int[patternNo*nNo];

  cerr << "# call was: ";
  for (int i= 0; i < argc; i++) cerr << argv[i] << " ";
  cerr << endl;
  
  for (int c= 0; c < classNo; c++)
  {
    for (int x= 0; x < nNo; x++) pat[x]= 0;
    for (int x= 0; x < nact; x++)
    {
      // get exactly nact active neurons in mother pattern
      do {
	theno= (int) (R.n()*nNo);
      } while (pat[theno] == 1);
      pat[theno]= 1;
      patns[x]= theno;
    }

    for (int n= 0; n < single_pNo; n++)
    {
      for (int x= 0; x < nNo; x++) {
	patp[x]= pat[x];
      }
      for (int x= 0; x < npert; x++)
      {
	theno= R.n()*nact;  // number of switched off one;
	do {
	  newno= (int) (R.n()*nNo);
	} while (pat[newno] == 1);
	patp[patns[theno]]= 0;
	patp[newno]= 1;
      }
      // note: Only risk remaining is that a switched-off neuron is 
      // switched back on as replacement of another one (fat chance!)
    
      for (int x=0; x < nNo; x++) {
	if (patp[x] == 1) {
	  fpt[(c*single_pNo+n)*nNo+x]= lambdaOn;
	}
	else {
	  fpt[(c*single_pNo+n)*nNo+x]= lambdaOff;
	}
      }
    }
  }

  os.write((char *)fpt, patternNo*nNo*sizeof(unsigned int));
  os.close();
  delete[] fpt;

  return 0;
}

