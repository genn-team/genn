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
#include "gauss.h"

#include "randomGen.cc"

randomGen R;
randomGauss RG;

int main(int argc, char *argv[])
{
  if (argc != 7)
  {
    cerr << "usage: gen_pnkc_syns <nAL> <nMB> ";
    cerr << "<prob. of PN-KC syn> ";
    cerr << "<mean strength> ";
    cerr << "<strength jitter> <outfile>" << endl;
    exit(1);
  }

  int nAL= atoi(argv[1]);
  int nMB= atoi(argv[2]);
  float psyn= atof(argv[3]);
  float pnkc_gsyn= atof(argv[4]);
  float pnkc_jitter= atof(argv[5]);
  ofstream os(argv[6], ios::binary);
  float gsyn;
  float *g= new float[nAL*nMB];

  cerr << "# call was: ";
  for (int i= 0; i < argc; i++) cerr << argv[i] << " ";
  cerr << endl;

  for (int i= 0; i < nAL; i++) {
    for (int j= 0; j < nMB; j++) {
      if (R.n() < psyn) {
	gsyn= pnkc_gsyn+pnkc_jitter*RG.n();
	g[i*nMB+j]= gsyn;
      }
      else {
	g[i*nMB+j]= 0.0f;
      }
    }
  }
  
  os.write((char *)g, nAL*nMB*sizeof(float));
  os.close();
  delete[] g;
  
  return 0;
}

