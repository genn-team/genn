
#ifndef DPCLASS_H
#define DPCLASS_H

#include <vector>

using namespace std;


class dpclass {
public:
    virtual double calculateDerivedParameter(int, vector<double>, double = 0.5) {return -1;}
};

#endif // DPCLASS_H
