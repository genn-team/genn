/*
 * This code follows Marsaglia & Tsang 2000, but with a passed-in data
 * structure for mutable variables. (RngData modifications authored by
 * Seb James, August 2014).
 *
 * Example usage:
 *
 * #include <stdio.h>
 * #include "rng.h"
 *
 * int main()
 * {
 *     RngData rd;
 *     float rn;
 *     int i;
 *
 *     rngDataInit (&rd);
 *     zigset(&rd, 11);
 *     rd.seed = 102;
 *
 *     while (i < 10) {
 *         rn = randomNormal ((&rd));
 *         printf ("%f\n", rn);
 *         i++;
 *     }
 *     return i;
 * }
 *
 * g++ -o testrng testrng.cpp -lm
 */

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <sys/time.h>

/*
 * Definition of a data storage class for use with this code. Each
 * thread wishing to call functions from this random number generator
 * must manage an instance of RngData. All functions operate by taking
 * a pointer to an instance of RngData.
 *
 * Prior to using one of the random number generators here, first call
 * rngDataInit to set up your RngData instance (if you're compiling
 * this with a c++-11 compatible compiler, you can move the
 * initialisation into the struct).
 */
struct RngData {
    // Some constants, which could go at global scope, but which I
    // prefer to have in here.
    const static int a_RNG = 1103515245;
    const static int c_RNG = 12345;
    int seed, hz;
    unsigned int iz,jz,jsr,kn[128],ke[256];
    float wn[128],fn[128], we[256],fe[256];
    float qBinVal,sBinVal,rBinVal,aBinVal;
};

// An initialiser function for RngData
void rngDataInit (RngData* rd)
{
    rd->seed = 0;
    rd->jsr = 123456789;
    rd->qBinVal = -1;
}

int getTime(void)
{
    struct timeval currTime;
    gettimeofday(&currTime, NULL);
    return time(0) | currTime.tv_usec;
}

float uniformGCC(RngData* rd)
{
    rd->seed = abs(rd->seed * rd->a_RNG + rd->c_RNG);
    float seed2 = rd->seed/2147483648.0;
    return seed2;
}

// RANDOM NUMBER GENERATOR
#define SHR3(rd) ((rd)->jz=(rd)->seed,          \
                  (rd)->seed^=((rd)->seed<<13), \
                  (rd)->seed^=((rd)->seed>>17), \
                  (rd)->seed^=((rd)->seed<<5),  \
                  (rd)->jz+(rd)->seed)
#define UNI(rd) uniformGCC(rd)
#define RNOR(rd) ((rd)->hz=SHR3(rd),                                    \
                  (rd)->iz=(rd)->hz&127,                                \
                  (abs((rd)->hz) < (rd)->kn[(rd)->iz]) ? (rd)->hz*(rd)->wn[(rd)->iz] : nfix(rd))
#define REXP(rd) ((rd)->jz=SHR3(rd),                                    \
                  (rd)->iz=(rd)->jz&255,                                \
                  ((rd)->jz < (rd)->ke[(rd)->iz]) ? (rd)->jz*(rd)->we[(rd)->iz] : efix(rd))
#define RPOIS(rd) -log(1.0-UNI(rd))

float nfix (RngData* rd) /*provides RNOR if #define cannot */
{
    const float r = 3.442620f;
    static float x, y;
    for (;;) {
        x=rd->hz*rd->wn[rd->iz];
        if (rd->iz==0) {
            do {
                x = -log(UNI(rd))*0.2904764;
                y = -log(UNI(rd));
            } while (y+y<x*x);
            return (rd->hz>0) ? r+x : -r-x;
        }

        if (rd->fn[rd->iz]+UNI(rd)*(rd->fn[rd->iz-1]-rd->fn[rd->iz]) < exp(-.5*x*x)) {
            return x;
        }

        rd->hz=SHR3(rd);
        rd->iz=rd->hz&127;
        if (abs(rd->hz)<(int)rd->kn[rd->iz]) {
            return (rd->hz*rd->wn[rd->iz]);
        }
    }
}

float efix (RngData* rd) /*provides REXP if #define cannot */
{
    float x;
    for (;;) {
        if (rd->iz==0) {
            return (7.69711-log(UNI(rd)));
        }
        x=rd->jz*rd->we[rd->iz];
        if (rd->fe[rd->iz]+UNI(rd)*(rd->fe[rd->iz-1]-rd->fe[rd->iz]) < exp(-x)) {
            return (x);
        }
        rd->jz=SHR3(rd);
        rd->iz=(rd->jz&255);
        if (rd->jz<rd->ke[rd->iz]) {
            return (rd->jz*rd->we[rd->iz]);
        }
    }
}

// == This procedure sets the seed and creates the tables ==
void zigset (RngData* rd, unsigned int jsrseed)
{
    clock();

    const double m1 = 2147483648.0, m2 = 4294967296.;
    double dn=3.442619855899,tn=dn,vn=9.91256303526217e-3, q;
    double de=7.697117470131487, te=de, ve=3.949659822581572e-3;
    int i;
    rd->jsr=jsrseed;

    /* Tables for RNOR: */
    q=vn/exp(-.5*dn*dn);
    rd->kn[0]=(dn/q)*m1; rd->kn[1]=0;
    rd->wn[0]=q/m1; rd->wn[127]=dn/m1;
    rd->fn[0]=1.; rd->fn[127]=exp(-.5*dn*dn);
    for (i=126;i>=1;i--) {
        dn=sqrt(-2.*log(vn/dn+exp(-.5*dn*dn)));
        rd->kn[i+1]=(dn/tn)*m1; tn=dn;
        rd->fn[i]=exp(-.5*dn*dn); rd->wn[i]=dn/m1;
    }
    /* Tables for REXP */
    q = ve/exp(-de);
    rd->ke[0]=(de/q)*m2; rd->ke[1]=0;
    rd->we[0]=q/m2; rd->we[255]=de/m2;
    rd->fe[0]=1.; rd->fe[255]=exp(-de);
    for (i=254;i>=1;i--) {
        de=-log(ve/de+exp(-de));
        rd->ke[i+1]= (de/te)*m2; te=de;
        rd->fe[i]=exp(-de); rd->we[i]=de/m2;
    }
}

int slowBinomial(RngData* rd, int N, float p)
{
    int num = 0;
    for (int i = 0; i < N; ++i) {
        if (UNI(rd) < p) {
            ++num;
        }
    }
    return num;
}

int fastBinomial(RngData* rd, int N, float p)
{
    // setup the computationally intensive vals
    if (rd->qBinVal == -1) {
        rd->qBinVal = 1-p;
        rd->sBinVal = p/rd->qBinVal;
        rd->aBinVal = (N+1)*rd->sBinVal;
        rd->rBinVal = pow(rd->qBinVal,N);
    }

    float r = rd->rBinVal;
    float u = UNI(rd);
    int x = 0;

    while (u>r) {
        u=u-rd->rBinVal;
        x=x+1;
        r=((rd->aBinVal/float(x))-rd->sBinVal)*r;
    }

    return x;
}

#define _randomUniform(rd) uniformGCC(rd)
#define _randomNormal(rd) RNOR(rd)
#define _randomExponential(rd) REXP(rd)
#define _randomPoisson(rd) RPOIS(rd)
#define HACK_MACRO(rd,N,p) 1;                                           \
    int spks=fastBinomial(rd,N,p);                                      \
    for(unsigned int i=0;i<spks;++i) {DATAOutspike.push_back(num);}
