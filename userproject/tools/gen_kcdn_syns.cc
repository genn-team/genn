/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file gen_kcdn_syns.cc

\brief This file is part of a tool chain for running the classol/MBody1 example model.

This file compiles to a tool to generate appropriate connectivity patterns between KCs and DNs (detector neurons) in the model. The connectivity is saved to file and can then be read by the classol method for reading this connectivity.
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
    cerr << "usage: gen_kcdn_syns <nMB> <nLobes> ";
    cerr << "<mean strength> ";
    cerr << "<strength jitter> <min strength> <outfile>" << endl;
    exit(1);
  }

  int nMB= atoi(argv[1]);
  int nLB= atoi(argv[2]);
  double pnkc_gsyn= atof(argv[3]);
  double pnkc_jitter= atof(argv[4]);
  double EPS= atof(argv[5]);
  ofstream os(argv[6],ios::binary);
  double gsyn;
  double *g= new double[nMB*nLB];

  cerr << "# call was: ";
  for (int i= 0; i < argc; i++) cerr << argv[i] << " ";
  cerr << endl;

  for (int i= 0; i < nMB; i++) {
    for (int j= 0; j < nLB; j++) {
      gsyn= pnkc_gsyn+pnkc_jitter*RG.n();
      if (gsyn < EPS) gsyn= EPS;
      g[i*nLB+j]= gsyn;
    }
  }
  os.write((char *)g, nMB*nLB*sizeof(double));
  os.close();
  delete[] g;
 
  return 0;
}

