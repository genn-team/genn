#ifndef __ISAAC_HPP
#define __ISAAC_HPP

#include <stdlib.h>

/*

    C++ TEMPLATE VERSION OF Robert J. Jenkins Jr.'s
    ISAAC Random Number Generator.

    Ported from vanilla C to to template C++ class
    by Quinn Tyler Jackson on 16-23 July 1998.

        qjackson@wave.home.com

    The function for the expected period of this
    random number generator, according to Jenkins is:

        f(a,b) = 2**((a+b*(3+2^^a)-1)

        (where a is ALPHA and b is bitwidth)
        
    So, for a bitwidth of 32 and an ALPHA of 8,
    the expected period of ISAAC is:

        2^^(8+32*(3+2^^8)-1) = 2^^8295

    Jackson has been able to run implementations
    with an ALPHA as high as 16, or

        2^^2097263

*/


#ifndef __ISAAC64
typedef unsigned long int UINT32;
const UINT32 GOLDEN_RATIO = UINT32(0x9e3779b9);
typedef UINT32 ISAAC_INT;
#else   // __ISAAC64
typedef unsigned __int64 UINT64;
const UINT64 GOLDEN_RATIO = UINT64(0x9e3779b97f4a7c13);
typedef UINT64 ISAAC_INT;
#endif  // __ISAAC64


template <int ALPHA = (8), class T = ISAAC_INT> 
class QTIsaac
{

  //    friend int main (void);

public:

    typedef unsigned char byte;

    struct randctx
    {
        randctx(void)
        {
            randrsl = (T*)::malloc(N * sizeof(T));
            randmem = (T*)::malloc(N * sizeof(T));
        }

        ~randctx(void)
        {
            ::free((void*)randrsl);
            ::free((void*)randmem);
        }

        T randcnt;
        T* randrsl;
        T* randmem;
        T randa;
        T randb;
        T randc;
    };

    QTIsaac(T a = 0, T b = 0, T c = 0);
    virtual ~QTIsaac(void);

    T rand(void);
    virtual void randinit(randctx* ctx, bool bUseSeed);
    virtual void srand(T a = 0, T b = 0, T c = 0, T* s = NULL);

    enum {N = (1<<ALPHA)};

protected:

    virtual void isaac(randctx* ctx);

    T ind(T* mm, T x);
    void rngstep(T mix, T& a, T& b, T*& mm, T*& m, T*& m2, T*& r, T& x, T& y);
    virtual void shuffle(T& a, T& b, T& c, T& d, T& e, T& f, T& g, T& h);

private:

    randctx m_rc;   

};


template<int ALPHA, class T>
QTIsaac<ALPHA,T>::QTIsaac(T a, T b, T c)
{
    srand();
}


template<int ALPHA, class T>
QTIsaac<ALPHA,T>::~QTIsaac(void)
{
    // DO NOTHING
}


template<int ALPHA, class T>
void QTIsaac<ALPHA,T>::srand(T a, T b, T c, T* s)
{
    for(int i = 0; i < N; i++)
    {
        m_rc.randrsl[i] = s != NULL ? s[i] : 0;
    }

    m_rc.randa = a;
    m_rc.randb = b;
    m_rc.randc = c;
    
    randinit(&m_rc, true);
}


template<int ALPHA, class T>
inline T QTIsaac<ALPHA,T>::rand(void)
{
    return(!m_rc.randcnt-- ? (isaac(&m_rc), m_rc.randcnt=(N-1), m_rc.randrsl[m_rc.randcnt]) : m_rc.randrsl[m_rc.randcnt]);
}


template<int ALPHA, class T>
void QTIsaac<ALPHA,T>::randinit(randctx* ctx, bool bUseSeed)
{
    T a,b,c,d,e,f,g,h;
    
    a = b = c = d = e = f = g = h = (T) GOLDEN_RATIO;

    T* m = (ctx->randmem);
    T* r = (ctx->randrsl);

    if(!bUseSeed)
    {
        ctx->randa = 0;
        ctx->randb = 0;
        ctx->randc = 0;
    }

    // scramble it
    for(int i=0; i < 4; ++i)         
    {
        shuffle(a,b,c,d,e,f,g,h);
    }

    if(bUseSeed) 
    {
        // initialize using the contents of r[] as the seed

        for(int i=0; i < N; i+=8)
        {
            a+=r[i  ]; b+=r[i+1]; c+=r[i+2]; d+=r[i+3];
            e+=r[i+4]; f+=r[i+5]; g+=r[i+6]; h+=r[i+7];

            shuffle(a,b,c,d,e,f,g,h);

            m[i  ]=a; m[i+1]=b; m[i+2]=c; m[i+3]=d;
            m[i+4]=e; m[i+5]=f; m[i+6]=g; m[i+7]=h;
        }           
        
        //do a second pass to make all of the seed affect all of m

        for(int i=0; i < N; i += 8)
        {
            a+=m[i  ]; b+=m[i+1]; c+=m[i+2]; d+=m[i+3];
            e+=m[i+4]; f+=m[i+5]; g+=m[i+6]; h+=m[i+7];
            
            shuffle(a,b,c,d,e,f,g,h);

            m[i  ]=a; m[i+1]=b; m[i+2]=c; m[i+3]=d;
            m[i+4]=e; m[i+5]=f; m[i+6]=g; m[i+7]=h;
        }
    }
    else
    {
        // fill in mm[] with messy stuff

        for(int i=0; i < N; i += 8)
        {
	  shuffle(a,b,c,d,e,f,g,h);
	  
	  m[i  ]=a; m[i+1]=b; m[i+2]=c; m[i+3]=d;
	  m[i+4]=e; m[i+5]=f; m[i+6]=g; m[i+7]=h;
	}
    }

    isaac(ctx);         // fill in the first set of results 
    ctx->randcnt = N;   // prepare to use the first set of results 
}


template<int ALPHA, class T>
inline T QTIsaac<ALPHA,T>::ind(T* mm, T x)
{
    #ifndef __ISAAC64
    return (*(T*)((byte*)(mm) + ((x) & ((N-1)<<2))));
    #else   // __ISAAC64
    return (*(T*)((byte*)(mm) + ((x) & ((N-1)<<3))));
    #endif  // __ISAAC64
}


template<int ALPHA, class T>
inline void QTIsaac<ALPHA,T>::rngstep(T mix, T& a, T& b, T*& mm, T*& m, T*& m2, T*& r, T& x, T& y)
{
    x = *m;  
    a = (a^(mix)) + *(m2++); 
    *(m++) = y = ind(mm,x) + a + b; 
    *(r++) = b = ind(mm,y>>ALPHA) + x; 
}


template<int ALPHA, class T>
void QTIsaac<ALPHA,T>::shuffle(T& a, T& b, T& c, T& d, T& e, T& f, T& g, T& h)
{ 
    #ifndef __ISAAC64
    a^=b<<11; d+=a; b+=c; 
    b^=c>>2;  e+=b; c+=d; 
    c^=d<<8;  f+=c; d+=e; 
    d^=e>>16; g+=d; e+=f; 
    e^=f<<10; h+=e; f+=g; 
    f^=g>>4;  a+=f; g+=h; 
    g^=h<<8;  b+=g; h+=a; 
    h^=a>>9;  c+=h; a+=b; 
    #else // __ISAAC64
    a-=e; f^=h>>9;  h+=a;
    b-=f; g^=a<<9;  a+=b;
    c-=g; h^=b>>23; b+=c;
    d-=h; a^=c<<15; c+=d;
    e-=a; b^=d>>14; d+=e;
    f-=b; c^=e<<20; e+=f;
    g-=c; d^=f>>17; f+=g;
    h-=d; e^=g<<14; g+=h;
    #endif // __ISAAC64
}


template<int ALPHA, class T>
void QTIsaac<ALPHA,T>::isaac(randctx* ctx)
{
    T x,y;
    
    T* mm = ctx->randmem;
    T* r  = ctx->randrsl;
    
    T a = (ctx->randa);
    T b = (ctx->randb + (++ctx->randc));

    T* m    = mm; 
    T* m2   = (m+(N/2));
    T* mend = m2;
    
    for(; m<mend; )
    {
        #ifndef __ISAAC64
        rngstep((a<<13), a, b, mm, m, m2, r, x, y);
        rngstep((a>>6) , a, b, mm, m, m2, r, x, y);
        rngstep((a<<2) , a, b, mm, m, m2, r, x, y);
        rngstep((a>>16), a, b, mm, m, m2, r, x, y);
        #else   // __ISAAC64
        rngstep(~(a^(a<<21)), a, b, mm, m, m2, r, x, y);
        rngstep(  a^(a>>5)  , a, b, mm, m, m2, r, x, y);
        rngstep(  a^(a<<12) , a, b, mm, m, m2, r, x, y);
        rngstep(  a^(a>>33) , a, b, mm, m, m2, r, x, y);
        #endif  // __ISAAC64
    }

    m2 = mm;

    for(; m2<mend; )
    {
        #ifndef __ISAAC64
        rngstep((a<<13), a, b, mm, m, m2, r, x, y);
        rngstep((a>>6) , a, b, mm, m, m2, r, x, y);
        rngstep((a<<2) , a, b, mm, m, m2, r, x, y);
        rngstep((a>>16), a, b, mm, m, m2, r, x, y);
        #else   // __ISAAC64
        rngstep(~(a^(a<<21)), a, b, mm, m, m2, r, x, y);
        rngstep(  a^(a>>5)  , a, b, mm, m, m2, r, x, y);
        rngstep(  a^(a<<12) , a, b, mm, m, m2, r, x, y);
        rngstep(  a^(a>>33) , a, b, mm, m, m2, r, x, y);
        #endif  // __ISAAC64
    }
    
    ctx->randb = b;
    ctx->randa = a;
}


#endif // __ISAAC_HPP
