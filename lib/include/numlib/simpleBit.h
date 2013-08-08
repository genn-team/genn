#ifndef SIMPLEBIT_H
#define SIMPLEBIT_H

#include <cassert>
#include <cmath>

// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i)))

#define setB(x,i) x= ((x) | (0x80000000 >> (i)))

#define delB(x,i) x= ((x) & (~(0x80000000 >> (i))))

#endif
