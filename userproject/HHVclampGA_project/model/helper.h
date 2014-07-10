/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Informatics
              University of Sussex 
              Brighton BN1 9QJ, UK
  
   email to:  t.nowotny@sussex.ac.uk
  
   initial version: 2014-06-26
  
--------------------------------------------------------------------------*/

#include <vector>

typedef struct {
  double t;
  double baseV;
  int N;
  vector<double> st;
  vector<double> V;
} inputSpec;

double sigGNa= 0.1;
double sigENa= 10.0;
double sigGK= 0.1;
double sigEK= 10.0;
double sigGl= 0.1;
double sigEl= 10.0;
double sigC= 0.1;

ostream &operator<<(ostream &os, inputSpec &I)
{
  os << " " << I.t << "  ";
  os << " " << I.baseV << "    ";
  os << " " << I.N << "    ";  
  for (int i= 0; i < I.N; i++) {
    os << I.st[i] << " ";
    os << I.V[i] << "  ";
  }
  return os;
}

void write_para() 
{
  fprintf(stderr, "# DT %f \n", DT);
}

void single_var_reinit(int n, double fac) 
{
  gNaHH[n]*= (1.0+fac*sigGNa*RG.n()); // multiplicative Gaussian noise
  ENaHH[n]+= fac*sigENa*RG.n(); // additive Gaussian noise
  gKHH[n]*= (1.0+fac*sigGK*RG.n()); // multiplicative Gaussian noise
  EKHH[n]+= fac*sigEK*RG.n(); // additive Gaussian noise
  glHH[n]*= (1.0+fac*sigGl*RG.n()); // multiplicative Gaussian noise
  ElHH[n]+= fac*sigEl*RG.n(); // additive Gaussian noise
  CHH[n]*= (1.0+fac*sigC*RG.n()); // multiplicative Gaussian noise
}

void copy_var(int src, int trg)
{
  gNaHH[trg]= gNaHH[src];
  ENaHH[trg]= ENaHH[src];
  gKHH[trg]= gKHH[src];
  EKHH[trg]=EKHH[src];
  glHH[trg]= glHH[src];
  ElHH[trg]= ElHH[src];
  CHH[trg]= CHH[src];
}

void var_reinit(double fac) 
{
  // add noise to the parameters
  for (int n= 0; n < NPOP; n++) {
    single_var_reinit(n, fac);
  }
  copyStateToDevice();	
}

void truevar_init()
{
  for (int n= 0; n < NPOP; n++) {  
    VHH[n]= myHH_ini[0];
    mHH[n]= myHH_ini[1];
    hHH[n]= myHH_ini[2];
    nHH[n]= myHH_ini[3];
    errHH[n]= 0.0;
  }
  copyStateToDevice();	  
}


double Vexp;
double mexp;
double hexp;
double nexp;
double gNaexp;
double ENaexp;
double gKexp;
double EKexp;
double glexp;
double Elexp;
double Cexp;

void initexpHH()
{
  Vexp= myHH_ini[0];
  mexp= myHH_ini[1];
  hexp= myHH_ini[2];
  nexp= myHH_ini[3];
  gNaexp= myHH_ini[4];
  ENaexp= myHH_ini[5];
  gKexp= myHH_ini[6];
  EKexp= myHH_ini[7];
  glexp= myHH_ini[8];
  Elexp= myHH_ini[9];
  Cexp= myHH_ini[10]; 
}

void truevar_initexpHH()
{
  Vexp= myHH_ini[0];
  mexp= myHH_ini[1];
  hexp= myHH_ini[2];
  nexp= myHH_ini[3];
}


void runexpHH(float t)
{
  // calculate membrane potential
  double Imem;
  unsigned int mt;
  double mdt= DT/100.0;
  for (mt=0; mt < 100; mt++) {
    IsynGHH= 200.0*(stepVGHH-Vexp);
    //    cerr << IsynGHH << " " << Vexp << endl;
    Imem= -(mexp*mexp*mexp*hexp*gNaexp*(Vexp-(ENaexp))+
	    nexp*nexp*nexp*nexp*gKexp*(Vexp-(EKexp))+
	    glexp*(Vexp-(Elexp))-IsynGHH);
    double _a= (3.5+0.1*Vexp) / (1.0-exp(-3.5-0.1*Vexp));
    double _b= 4.0*exp(-(Vexp+60.0)/18.0);
    mexp+= (_a*(1.0-mexp)-_b*mexp)*mdt;
    _a= 0.07*exp(-Vexp/20.0-3.0);
    _b= 1.0 / (exp(-3.0-0.1*Vexp)+1.0);
    hexp+= (_a*(1.0-hexp)-_b*hexp)*mdt;
    _a= (-0.5-0.01*Vexp) / (exp(-5.0-0.1*Vexp)-1.0);
    _b= 0.125*exp(-(Vexp+60.0)/80.0);
    nexp+= (_a*(1.0-nexp)-_b*nexp)*mdt;
    Vexp+= Imem/Cexp*mdt;
  }
}


void initI(inputSpec &I) 
{
  I.t= 200.0;
  I.baseV= -60.0;
  I.N= 12;
  I.st.push_back(10.0);
  I.V.push_back(-30.0);
  I.st.push_back(20.0);
  I.V.push_back(-60.0);

  I.st.push_back(40.0);
  I.V.push_back(-20.0);
  I.st.push_back(50.0);
  I.V.push_back(-60.0);

  I.st.push_back(70.0);
  I.V.push_back(-10.0);
  I.st.push_back(80.0);
  I.V.push_back(-60.0);

  I.st.push_back(100.0);
  I.V.push_back(0.0);
  I.st.push_back(110.0);
  I.V.push_back(-60.0);

  I.st.push_back(130.0);
  I.V.push_back(10.0);
  I.st.push_back(140.0);
  I.V.push_back(-60.0);

  I.st.push_back(160.0);
  I.V.push_back(20.0);
  I.st.push_back(170.0);
  I.V.push_back(-60.0);
  assert((I.N == I.V.size()) && (I.N == I.st.size()));
}
      
