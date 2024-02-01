.. py:currentmodule:: pygenn

=====================
Upgrading from GeNN 4
=====================
GeNN 5 does not aim to be a backward-compatible upgrade.
In PyGeNN we have strived to add backward compatibility where possible but most models will need to be updated.

--------
GeNNCode
--------
In GeNN 4 GeNNCode was passed directly to the underlying compiler whereas, in GeNN 5, it is parsed and only a subset of C99 language features are supported.
In PyGeNN,
- The & operator isn't supported - on the target GPU hardware, local variables are assumed to be stored in registers and not addressable. The only time this is limiting is when dealing with extra global parameter arrays as you can no longer do stuff like ``const int *egpSubset = &egp[offset];`` and instead have to do ``const int *egpSubset = egp + offset;``.
- The old ``$(xx)`` syntax for referencing GeNN stuff is no longer necessary at all. 

The most