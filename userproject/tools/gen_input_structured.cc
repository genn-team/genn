/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Science
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file userproject/tools/gen_input_structured.cc

\brief This file is part of a tool chain for running the classol/MBody1 example model.

This file compiles to a tool to generate appropriate input patterns for the antennal lobe in the model. The triple "fix" in the filename refers to a three-fold control for having the same number of active inputs for each pattern, even if changing patterns by adding noise.
*/ 
//--------------------------------------------------------------------------



#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>

using namespace std;

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
        return EXIT_FAILURE;
    }

    const int nNo= atoi(argv[1]);
    const int classNo= atoi(argv[2]);
    const int single_pNo= atoi(argv[3]);
    const double pact= atof(argv[4]);
    const double pperturb= atof(argv[5]);
    const double lambdaOn= atof(argv[6]);
    const double lambdaOff= atof(argv[7]);
    const int patternNo= single_pNo*classNo;
    int *pat = new int[nNo];
    int *patp = new int[nNo];
    const int nact= (int) (pact*nNo);
    int *patns = new int[nact];
    const int npert= (int) (pperturb*nact);
    ofstream os(argv[8], ios::binary);
    double *fpt= new double[patternNo*nNo];

    cerr << "# call was: ";
    for (int i = 0; i < argc; i++) {
        cerr << argv[i] << " ";
    }
    cerr << endl;
  
    std::mt19937 rng;
    std::uniform_int_distribution<int> nNoDist(0, nNo - 1);
    std::uniform_int_distribution<int> nActDist(0, nact - 1);

    for (int c= 0; c < classNo; c++)
    {
        for (int x = 0; x < nNo; x++) {
            pat[x] = 0;
        }
        for (int x= 0; x < nact; x++)
        {
            // get exactly nact active neurons in mother pattern
            int theno;
            do {
                theno= nNoDist(rng);
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
                const int theno= nActDist(rng);  // number of switched off one;
                int newno;
                do {
	                newno= nNoDist(rng);
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

    os.write((char *)fpt, patternNo*nNo*sizeof(double));
    os.close();
    delete[] fpt;

    return EXIT_SUCCESS;
}

