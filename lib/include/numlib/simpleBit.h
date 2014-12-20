#ifndef SIMPLEBIT_H
#define SIMPLEBIT_H //!< macro for avoiding multiple inclusion during compilation

#include <cassert>
#include <cmath>

//-----------------------------------------------------------------------
/*! \file simpleBit.h

\brief Contains three macros that allow simple bit manipulations on an (presumably unsigned) 32 bit integer
*/
//-----------------------------------------------------------------------

// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x

#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1 

#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0 

#endif
