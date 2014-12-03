/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file gen_pnkc_syns.cc

\brief This file is part of a tool chain for running the classol/MBody1 example model.

This file compiles to a tool to generate appropriate connectivity patterns between PNs and KCs in the model. The connectivity is saved to file and can then be read by the classol method for reading this connectivity.
*/ 
//--------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

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
  double psyn= atof(argv[3]);
  double pnkc_gsyn= atof(argv[4]);
  double pnkc_jitter= atof(argv[5]);
  ofstream os(argv[6], ios::binary);
  double gsyn;
  double *g= new double[nAL*nMB];

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
  
  os.write((char *)g, nAL*nMB*sizeof(double));
  os.close();
  delete[] g;
  
  return 0;
}

