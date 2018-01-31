#pragma once

const double initialHHValues[12] = {
    -60.0,         // 0 - membrane potential E
    0.0529324,     // 1 - prob. for Na channel activation m
    0.3176767,     // 2 - prob. for not Na channel blocking h
    0.5961207,      // 3 - prob. for K channel activation n
    120.0,         // 4 - gNa: Na conductance in 1/(mOhms * cm^2)
    55.0,          // 5 - ENa: Na equi potential in mV
    36.0,          // 6 - gK: K conductance in 1/(mOhms * cm^2)
    -72.0,         // 7 - EK: K equi potential in mV
    0.3,           // 8 - gl: leak conductance in 1/(mOhms * cm^2)
    -50.0,         // 9 - El: leak equi potential in mV
    1.0,           // 10 - Cmem: membr. capacity density in muF/cm^2
    0.0
};