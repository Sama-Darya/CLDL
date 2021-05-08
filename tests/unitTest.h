//
// Created by sama on 07/05/2021.
//

#ifndef CLDL_UNITTEST_H
#define CLDL_UNITTEST_H

#include "../include/cldl/Net.h"


class unitTest {
public:
    unitTest();
    void test_net_setInputs();
    void test_net_porpInputs();
    void test_net_masterPropagation();
    void printInputs();
    void printWeights();
    void printIntErrors();

private:
    Net *net;
    int nLayers = 3;
    int* nNeurons = nullptr;
    int nInputs = 2;
    double* inputs = nullptr;
    double controlError = 1;
    double learningRate = 1;
    double output = 0;

};


#endif //CLDL_UNITTEST_H
