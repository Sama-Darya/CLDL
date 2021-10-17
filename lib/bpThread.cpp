//
// Created by sama on 16/10/2021.
//

#include "../include/cldl/bpThread.h"

#include <cstdio>
#include <chrono>
#include <thread>
#include <iostream>

using namespace std;

void bpThread::run() {
   cout << "inside of the thread: " <<offset << endl;
    //        for (int wn_index = 0; wn_index < layers[L_index-1]->getnNeurons(); wn_index++){ //iterate through the inputs to each layer
//            double thisSum = 0.00;
//            for (int n_index = 0; n_index < layers[L_index]->getnNeurons(); n_index++){ //iterate through the neurons of each layer
//                if( L_index == nextInjectionLayerIndex){
//                    assert((injectionCount<=nLayers)&&(injectionCount>=0) && "NET failed");
//                    tempError = controlError;
//                    injectionCount += 1;
//                    nextInjectionLayerIndex = injectionLayerIndex[injectionCount];
//                }else{
//                    tempError = layers[L_index]->getInternalErrors(_internalErrorIndex, n_index);
//                }
//                tempWeight = layers[L_index]->getWeights(n_index, wn_index);
//                thisSum += (tempError * tempWeight);
//            }
//            assert(std::isfinite(thisSum) && "NET failed");
//            layers[L_index-1]->setInternalErrors(_internalErrorIndex, thisSum,
//                                                 wn_index, _errorMethod);
//        }
}
