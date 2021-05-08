//
// Created by sama on 07/05/2021.
//

#include "unitTest.h"
#include "../include/cldl/Net.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

unitTest::unitTest(){
    nNeurons= new int[nLayers];
    inputs = new double[nInputs];
    for(int i = 0; i<nLayers; i++){
        nNeurons[i] = nLayers - i;
    }
    for(int i = 0; i<nInputs; i++){
        inputs[i] = 1;
    }
    net = new Net(nLayers, nNeurons, nInputs, 1);
    net->initNetwork(Neuron::W_ONES, Neuron::B_NONE,
                     Neuron::Act_NONE);
    net->setLearningRate(learningRate);
}

void unitTest::test_net_setInputs(){
    cout << "testing setInputs: \n\t inputs control: ";
    for(int i = 0; i < nInputs; i++){
        cout << inputs[i] << ", ";
    }
    double* inputsP = &inputs[0];
    net->setInputs(inputsP);
    cout << "result: ";
    for(int i = 0; i < net->getnInputs(); i++){
        cout << net->getInputs(i) << ", ";
    }
    assert(nInputs == net->getnInputs);
    int match = 0;
    for(int i = 0; i < net->getnInputs(); i++){
        if(inputs[i] !=  net->getInputs(i)){
            cout <<  "\n\t Discrepancy in input index " << i << endl;
            match = 1;
        }
    }
    if(match == 0){
        cout << "\n\t The inputs match" <<endl;
    }
}

void unitTest::test_net_porpInputs(){
    net->propInputs();
    for(int i = 0; i < nInputs; i++){
        output += inputs[i] * 1;
    }
    cout << "testing porpInputs: \n\t output control: " << output << ", result: " <<
    net->getLayer(0)->getOutput(0) << endl;
    if(output  !=  net->getLayer(0)->getOutput(0)){
        cout <<  "\t Discrepancy in the output" << endl;
    }else{
        cout << "\t The outputs match" <<endl;
    }
    printInputs();
}

void unitTest::test_net_masterPropagation(){
    std::vector<int> injectionLayers;
    injectionLayers.reserve(1);
    injectionLayers = {nLayers - 1};
    printIntErrors();
    net->masterPropagate(injectionLayers, 0,
                                     Net::BACKWARD, controlError,
                                     Neuron::Sign);
    printIntErrors();
    printWeights();
    net->updateWeights();
    printWeights();
}

void unitTest::printInputs(){
    cout << "\t Inputs are: " ;
    for(int i = 0; i < nLayers; i++){
        cout << "L" << i ;
        for(int j = 0; j < nNeurons[i]; j++){
            cout << "N" << j << ": (";
            for(int k = 0; k < net->getLayer(i)->getNeuron(j)->getnInputs(); k++){
                cout << net->getLayer(i)->getNeuron(j)->getInputs(k) << ",";
            }
            cout << ") ";
        }
    }
    cout<<endl;
}

void unitTest::printIntErrors(){
    cout << "\t Internal Errors are: " ;
    for(int i = 0; i < nLayers; i++){
        cout << "L" << i ;
        for(int j = 0; j < nNeurons[i]; j++){
            cout << "N" << j << ": (" << net->getLayer(i)->getNeuron(j)->getInternalErrors(0)<< ",";
            cout << ") ";
        }
    }
    cout<<endl;
}

void unitTest::printWeights(){
    cout << "\t Weights are: " ;
    for(int i = 0; i < nLayers; i++){
        cout << "L" << i ;
        for(int j = 0; j < nNeurons[i]; j++){
            cout << "N" << j << ": (";
            for(int k = 0; k < net->getLayer(i)->getNeuron(j)->getnInputs(); k++){
                cout << net->getLayer(i)->getNeuron(j)->getWeights(k) << ",";
            }
            cout << ") ";
        }
    }
    cout<<endl;
}