/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Informatics
              University of Sussex 
              Brighton BN1 9QJ, UK
  
   email to:  t.nowotny@sussex.ac.uk
  
   initial version: 2014-06-26
  
--------------------------------------------------------------------------*/
#pragma once

#include <ostream>
#include <vector>

#include <cassert>
#include <cmath>

#include "HHNeuronParameters.h"
#include "HHVClampParameters.h"

struct inputSpec
{
    double t;
    double baseV;
    int N;
    std::vector<double> st;
    std::vector<double> V;
};

double sigGNa= 0.1;
double sigENa= 10.0;
double sigGK= 0.1;
double sigEK= 10.0;
double sigGl= 0.1;
double sigEl= 10.0;
double sigC= 0.1;

std::ostream &operator<<(std::ostream &os, inputSpec &I)
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


void runexpHH()
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
      
