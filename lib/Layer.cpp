#include "../include/cldl/Layer.h"
#include "../include/cldl/Neuron.h"

#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <numeric>
#include <vector>
#include <fstream>

//*************************************************************************************
// constructor de-constructor
//*************************************************************************************

Layer::Layer(int _nNeurons, int _nInputs, int _nInternalErrors){
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron
    neurons = new Neuron*[nNeurons];
    nInternalErrors = _nInternalErrors;
    /* dynamic allocation of memory to n number of
     * neuron-pointers and returning a pointer, "neurons",
     * to the first element */
    for (int i=0;i<nNeurons;i++){
        neurons[i]=new Neuron(nInputs, nInternalErrors);
    }
    /* each element of "neurons" pointer is itself a pointer
     * to a neuron object with specific no. of inputs*/
     //cout << "layer" << endl;
}

Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        delete neurons[i];
    }
    delete[] neurons;
    //delete[] inputs;
    /* it is important to delete any dynamic
     * memory allocation created by "new" */
}

//*************************************************************************************
//initialisation:
//*************************************************************************************

void Layer::initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    myLayerIndex = _layerIndex;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->initNeuron(i, myLayerIndex, _wim, _bim, _am);
    }
}

void Layer::setLearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setLearningRate(learningRate);
    }
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

void Layer::setInputs(const double* _inputs){
    /*this is only for the first layer*/
    inputs=_inputs;
    for (int j=0; j<nInputs; j++){
        Neuron** neuronsp = neurons;//point to the 1st neuron
        /* sets a temporarily pointer to neuron-pointers
         * within the scope of this function. this is inside
         * the loop, so that it is set to the first neuron
         * everytime a new value is distributed to neurons */
        double input= *inputs; //take this input value
        for (int i=0; i<nNeurons; i++){
            (*neuronsp)->setInput(j,input);
            //set this input value for this neuron
            neuronsp++; //point to the next neuron
        }
        inputs++; //point to the next input value
    }
}

void Layer::propInputs(int _index, double _value){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->propInputs(_index, _value);
    }
}

void Layer::calcOutputs(){
    for (int i=0; i<nNeurons; i++){
        layerHasReported = neurons[i]->calcOutput(layerHasReported);
    }
}

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

void Layer::setErrorInputsAndCalculateInternalError(int _index,
                                                    double _value,int _internalErrorIndex,
                                                    Neuron::errorMethod _errorMethod){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setErrorInputsAndCalculateInternalError(_index, _value,
                                                            _internalErrorIndex,_errorMethod);
    }
}
//*************************************************************************************
//back propagation of error:
//*************************************************************************************
void Layer::setInternalErrors(int _internalErrorIndex, double _sumValue,
                              int _neuronIndex, Neuron::errorMethod _errorMethod){
    assert(isfinite(_sumValue) && (_internalErrorIndex >= 0)
                && (_internalErrorIndex < nInternalErrors) &&
           (_neuronIndex >= 0) && (_neuronIndex < nNeurons) &&"Layer failed");
    neurons[_neuronIndex]->setInternalError(_internalErrorIndex, _sumValue,
                                            _errorMethod);
}

double Layer::getInternalErrors(int _internalErrorIndex, int _neuronIndex){
    assert((_neuronIndex<nNeurons) && (_neuronIndex>=0) && "Layer failed");
    return neurons[_neuronIndex]->getRawInternalErrors(_internalErrorIndex);
}

void Layer::updateWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->updateWeights();
    }
}


//*************************************************************************************
//getters:
//*************************************************************************************

Neuron* Layer::getNeuron(int _neuronIndex){
    assert(_neuronIndex < nNeurons && "Layer failed");
    return (neurons[_neuronIndex]);
}

double Layer::getSumOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getSumOutput());
}

double Layer::getWeights(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getWeights(_weightIndex));
}

double Layer::getInitWeight(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getInitWeights(_weightIndex));
}

double Layer::getWeightChange(){
    weightChange=0;
    for (int i=0; i<nNeurons; i++){
        weightChange += neurons[i]->getWeightChange();
    }
    //cout<< "Layer: WeightChange is: " << weightChange << endl;
    return (weightChange);
}

double Layer::getWeightDistance(){
    getWeightChange();
    return sqrt(weightChange);
}

double Layer::getOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getOutput());
}

int Layer::getnNeurons(){
    return (nNeurons);
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

void Layer::saveWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->saveWeights();
    }
}

void Layer::snapWeights(){
    std::ofstream wfile;
    char l = '0';
    l += myLayerIndex + 1;
    string name = "layerWeight";
    name += l;
    name += ".csv";
    wfile.open(name);
    for (int i=0; i<nNeurons; i++){
        for (int j=0; j<nInputs; j++){
            wfile << neurons[i]->getWeights(j) << " ";
        }
        wfile << "\n";
    }
    wfile.close();
}

void Layer::printLayer(){
    cout<< "\t This layer has " << nNeurons << " Neurons" <<endl;
    cout<< "\t There are " << nInputs << " inputs to this layer" <<endl;
    for (int i=0; i<nNeurons; i++){
        cout<< "\t Neuron number " << i << ":" <<endl;
        neurons[i]->printNeuron();
    }

}
