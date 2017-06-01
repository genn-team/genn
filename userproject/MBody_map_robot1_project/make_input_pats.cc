#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
      cerr << "usage: make_input_pats ";
      cerr << "<input fname> <nAL> <# classes> <# patterns> ";
      cerr << "<p_perturb> ";
      cerr << "<outfile> ";
      cerr << endl;
      exit(1);
    }
    
    int nNo= atoi(argv[2]);
    int classNo= atoi(argv[3]);
    int patternNo= atoi(argv[4]);
    double pperturb= atof(argv[5]);
    double lambdaOn= 1;
    double lambdaOff= 2e-4;

    ifstream ifs(argv[1]);
    ofstream ofs(argv[6],ios::binary);
    
    double *pics= new double[classNo*nNo];
    double *patterns= new double[classNo*patternNo*nNo];

    for (int i= 0; i < classNo*nNo; i++) {
      ifs >> pics[i];
    }
    srand(111);
    for (int c= 0; c < classNo; c++) {
      for (int ex= 0; ex < patternNo; ex++) {
	for (int k= 0; k < nNo; k++) {
	  if ((pics[c*nNo+k] > 1e-4) || (rand() < pperturb))  {
	    patterns[(c*patternNo+ex)*nNo+k]= lambdaOn;
	  }
	  else {
	    patterns[(c*patternNo+ex)*nNo+k]= lambdaOff;
	  }
	}
      }
    }

    ofs.write((char *) patterns,classNo*patternNo*nNo*sizeof(double));
    ifs.close();
    ofs.close();
    delete[] pics;
    delete[] patterns;
}
