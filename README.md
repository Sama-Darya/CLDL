# CLDL: Closed-Loop Deep Learning
 This is a flexible low level library that allows for development of innovative update rules in the context of
 closed-loop deep learning. It can be used with the conventional back-propagation algorithm or the newly developed
 'local propagation of global (closed-loop) error' algorithm. This repository is intended for use as an external
 library to any learning applications.

## Doxygen output
you can find descriptions of all functions in the doxygen output file ``refman.pdf``

## Building CLDL
CLDL uses cmake. just enter the CLDL directory from the root:
- ``cd CLDL``
and type:
- ``mkdir build && cd build``
- ``cmake ..``
- ``make``

record the path to both the generated library file (``libCLDL.a``) and of the ``include`` directory for external use in other projects.
