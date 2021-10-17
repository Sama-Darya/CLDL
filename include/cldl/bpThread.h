//
// Created by sama on 16/10/2021.
//

#ifndef CLDL_BPTHREAD_H
#define CLDL_BPTHREAD_H

#include "CppThread.h"

class bpThread : public CppThread {

public:
    bpThread(int _offset) {
        offset = _offset;
    }

private:
    void run();

private:
    int offset;
};

#endif