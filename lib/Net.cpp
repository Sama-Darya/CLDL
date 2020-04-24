#include "clbp/Net.h"
#include "clbp/Layer.h"
#include "clbp/Neuron.h"

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>

using namespace std;

//*************************************************************************************
//initialisation:
//*************************************************************************************

Net::Net(int _nLayers, int* _nNeurons, int _nInputs){
    nLayers = _nLayers; //no. of layers including inputs and outputs layers
    layers= new Layer*[nLayers];
    int* nNeuronsp = _nNeurons; //number of neurons in each layer
    nInputs=_nInputs; // the no. of inputs to the network (i.e. the first layer)
    //cout << "nInputs: " << nInputs << endl;
    int nInput = 0; //temporary variable to use within the scope of for loop
    for (int i=0; i<nLayers; i++){
        int numNeurons= *nNeuronsp; //no. neurons in this layer
        if (i==0){nInput=nInputs;}
        /* no. inputs to the first layer is equal to no. inputs to the network */
        layers[i]= new Layer(numNeurons, nInput);
        nNeurons += numNeurons;
        nWeights += (numNeurons * nInput);
        nInput=numNeurons;
        /*no. inputs to the next layer is equal to the number of neurons
         * in the current layer. */
        nNeuronsp++; //point to the no. of neurons in the next layer
    }
    nOutputs=layers[nLayers-1]->getnNeurons();
    errorGradient= new double[nLayers];
    //cout << "net" << endl;
}

Net::~Net(){
    for (int i=0; i<nLayers; i++){
        delete layers[i];
    }
    delete[] layers;
    delete[] errorGradient;
}

void Net::initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    for (int i=0; i<nLayers; i++){
        layers[i]->initLayer(i, _wim, _bim, _am);
    }
}

void Net::setLearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nLayers; i++){
        layers[i]->setlearningRate(learningRate);
    }
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

void Net::setInputs(const double* _inputs){
    inputs=_inputs;
    layers[0]->setInputs(inputs); //sets the inputs to the first layer only
}

void Net::propInputs(){
    for (int i=0; i<nLayers-1; i++){
        layers[i]->calcOutputs();
        for (int j=0; j<layers[i]->getnNeurons(); j++){
            double inputOuput = layers[i]->getOutput(j);
            layers[i+1]->propInputs(j, inputOuput);
        }
    }
    layers[nLayers-1]->calcOutputs();
    /* this calculates the final outoup of the network,
     * i.e. the output of the final layer
     * but this is not fed into any further layer*/
}

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

void Net::setForwardError(double _leadForwardError){
    leadForwardError=_leadForwardError;
    layers[0]->setForwardError(leadForwardError);
}

void Net::propErrorForward(){
    for (int i=0; i<nLayers-1; i++){
        layers[i]->calcForwardError();
        for (int j=0; j<layers[i]->getnNeurons(); j++){
            double inputOuput = layers[i]->getForwardError(j);
            layers[i+1]->propErrorForward(j, inputOuput);
        }
    }
    layers[nLayers-1]->calcForwardError();
}

//*************************************************************************************
//back propagation of error
//*************************************************************************************

void Net::setBackwardError(double _leadError){
    /* this is only for the final layer */
    theLeadError = _leadError;
    //cout<< "lead Error: " << theLeadError << endl;
    layers[nLayers-1]->setBackwardError(theLeadError);
    /* if the leadError was diff. for each output neuron
     * then it would be implemented in a for-loop */
}

void Net::propErrorBackward(){
    double tempError = 0;
    double tempWeight = 0;
    for (int i = nLayers-1; i > 0 ; i--){
        for (int k = 0; k < layers[i-1]->getnNeurons(); k++){
            double sum = 0.0;
            double weightSumer = 0.0;
            int counter = 0;
            for (int j = 0; j < layers[i]->getnNeurons(); j++){
                tempError = layers[i]->getBackwardError(j);
                tempWeight = layers[i]->getWeights(j,k);
                sum += (tempError * tempWeight);
                weightSumer += fabs(tempWeight);
                counter += 1;
            }
            double normSum = sum ; // / weightSumer;
            assert(std::isfinite(sum));
            assert(std::isfinite(weightSumer));
            assert(std::isfinite(counter));
            assert(std::isfinite(normSum));
            layers[i-1]->propErrorBackward(k, normSum);
          }
    }
    //cout << "--------------------------------------------------" << endl;
}

//*************************************************************************************
//MID propagation of error
//*************************************************************************************

void Net::setMidError(int _layerIndex, double _leadMidError){
    midLayerIndex = _layerIndex;
    theLeadMidError = _leadMidError;
    layers[midLayerIndex]->setMidError(theLeadMidError);
}

void Net::propMidErrorForward(){
    for (int i= midLayerIndex; i<nLayers-1; i++){
        layers[i]->calcMidError();
        for (int j=0; j<layers[i]->getnNeurons(); j++){
            double inputOuput = layers[i]->getMidError(j);
            layers[i+1]->propMidErrorForward(j, inputOuput);
        }
    }
    layers[nLayers-1]->calcMidError();
}

void Net::propMidErrorBackward(){
    double tempError = 0;
    double tempWeight = 0;
    for (int i = midLayerIndex; i > 0 ; i--){
        for (int k = 0; k < layers[i-1]->getnNeurons(); k++){
            double sum = 0.0;
            double weightSumer = 0.0;
            int counter = 0;
            for (int j = 0; j < layers[i]->getnNeurons(); j++){
                tempError = layers[i]->getMidError(j);
                tempWeight = layers[i]->getWeights(j,k);
                sum += (tempError * tempWeight);
                weightSumer += fabs(tempWeight);
                counter += 1;
            }
            double normSum = sum ; // / weightSumer;
            assert(std::isfinite(sum));
            assert(std::isfinite(weightSumer));
            assert(std::isfinite(counter));
            assert(std::isfinite(normSum));
            layers[i-1]->propMidErrorBackward(k, normSum);
        }
    }
    //cout << "--------------------------------------------------" << endl;
}

//*************************************************************************************
//exploding/vanishing gradient:
//*************************************************************************************

double Net::getGradient(Neuron::whichError _whichError, Layer::whichGradient _whichGradient) {
    for (int i=0; i<nLayers; i++) {
        errorGradient[i] = layers[i]->getGradient(_whichError, _whichGradient);
    }
    double gradientRatio = errorGradient[0] ; ///errorGradient[0];
    assert(std::isfinite(gradientRatio));
    return gradientRatio;
}

//*************************************************************************************
//learning:
//*************************************************************************************

void Net::setErrorCoeff(double _globalCoeff, double _backwardsCoeff, double _midCoeff, double _forwardCoeff, double _localCoeff, double  _echoCoeff){
    for (int i=0; i<nLayers; i++){
        layers[i]->setErrorCoeff(_globalCoeff, _backwardsCoeff, _midCoeff, _forwardCoeff, _localCoeff, _echoCoeff);
    }
}

void Net::updateWeights(){
    for (int i=nLayers-1; i>=0; i--){
        layers[i]->updateWeights();
    }
}

//*************************************************************************************
//global settings:
//*************************************************************************************

void Net::setGlobalError(double _globalError){
  globalError = _globalError;
    for (int i=nLayers-1; i>=0; i--){
        layers[i]->setGlobalError(globalError);
    }
}

//*************************************************************************************
//error echo
//*************************************************************************************

void Net::setEchoError(double _echoError) {
    echoError = _echoError;
    layers[nLayers-1]->setEchoError(echoError);
}

void Net::echoErrorBackward(){
    double tempError = 0;
    double tempWeight = 0;
    for (int i = nLayers-1; i > 0 ; i--){
        for (int k = 0; k < layers[i-1]->getnNeurons(); k++){
            double sum = 0.0;
            double weightSumer = 0.0;
            int counter = 0;
            for (int j = 0; j < layers[i]->getnNeurons(); j++){
                tempError = layers[i]->getEchoError(j);
                tempWeight = layers[i]->getWeights(j,k);
                sum += (tempError * tempWeight);
                weightSumer += fabs(tempWeight);
                counter += 1;
            }
            double normSum = sum ; // / weightSumer;
            assert(std::isfinite(sum));
            assert(std::isfinite(weightSumer));
            assert(std::isfinite(counter));
            assert(std::isfinite(normSum));
            layers[i-1]->echoErrorBackward(k, normSum);
        }
    }
    //cout << "--------------------------------------------------" << endl;
}

void Net::echoErrorForward(){
    for (int i=1; i<nLayers-1; i++){
        for (int j=0; j<layers[i]->getnNeurons(); j++){
            double inputOuput = layers[i]->getEchoError(j);
            layers[i+1]->echoErrorForward(j, inputOuput);
        }
        layers[i+1]->calcEchoError();
    }
    layers[nLayers-1]->calcEchoError();
}

void Net::doEchoError(double _theError){
    setEchoError(_theError);
    while (layers[nLayers-1]->getEchoError(0) != 0){
        echoErrorBackward();
        updateWeights();
        echoErrorForward();
        cout << "Echo Error is: " << layers[nLayers-1]->getEchoError(0) << endl;
    }
}

//*************************************************************************************
//local backpropagation of error
//*************************************************************************************

void Net::setLocalError(double _leadLocalError){
    /* this is only for the final layer */
    theLeadLocalError = _leadLocalError;
    //cout<< "lead Error: " << theLeadError << endl;
    layers[nLayers-1]->setLocalError(theLeadLocalError);
    /* if the leadError was diff. for each output neuron
     * then it would be implemented in a for-loop */
}

void Net::propGlobalErrorBackwardLocally(){
    double tempWeight = 0;
    for (int i = nLayers-1; i > 0 ; i--){
        for (int k = 0; k < layers[i-1]->getnNeurons(); k++){
            double sum = 0.0;
            double weightSumer = 0.0;
            int counter = 0;
            for (int j = 0; j < layers[i]->getnNeurons(); j++){
                tempWeight = layers[i]->getWeights(j,k);
                sum += (globalError * tempWeight);
                weightSumer += fabs(tempWeight);
                counter += 1;
            }
            double normSum = sum; //  / weightSumer;
            assert(std::isfinite(sum));
            assert(std::isfinite(weightSumer));
            assert(std::isfinite(counter));
            assert(std::isfinite(normSum));
            layers[i-1]->propGlobalErrorBackwardLocally(k, normSum);
        }
    }
    //cout << "--------------------------------------------------" << endl;
}


//*************************************************************************************
// getters:
//*************************************************************************************

double Net::getOutput(int _neuronIndex){
    return (layers[nLayers-1]->getOutput(_neuronIndex));
}

double Net::getSumOutput(int _neuronIndex){
    return (layers[nLayers-1]->getSumOutput(_neuronIndex));
}

int Net::getnLayers(){
    return (nLayers);
}

int Net::getnInputs(){
    return (nInputs);
}

Layer* Net::getLayer(int _layerIndex){
    assert(_layerIndex<nLayers);
    return (layers[_layerIndex]);
}

double Net::getWeightDistance(){
    double weightChange = 0 ;
    double weightDistance =0;
    for (int i=0; i<nLayers; i++){
        weightChange += layers[i]->getWeightChange();
    }
    weightDistance=sqrt(weightChange);
    // cout<< "Net: WeightDistance is: " << weightDistance << endl;
    return (weightDistance);
}

double Net::getLayerWeightDistance(int _layerIndex){
    return layers[_layerIndex]->getWeightDistance();
}

double Net::getWeights(int _layerIndex, int _neuronIndex, int _weightIndex){
    double weight=layers[_layerIndex]->getWeights(_neuronIndex, _weightIndex);
    return (weight);
}

int Net::getnNeurons(){
    return (nNeurons);
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

void Net::saveWeights(){
    for (int i=0; i<nLayers; i++){
        layers[i]->saveWeights();
    }
}

void Net::snapWeights(){
  layers[0]->snapWeights();
    // for (int i=0; i<nLayers; i++){
    //     layers[i]->snapWeights();
    // }
}

void Net::printNetwork(){
    cout<< "This network has " << nLayers << " layers" <<endl;
    for (int i=0; i<nLayers; i++){
        cout<< "Layer number " << i << ":" <<endl;
        layers[i]->printLayer();
    }
    cout<< "The output(s) of the network is(are):";
    for (int i=0; i<nOutputs; i++){
        cout<< " " << this->getOutput(i);
    }
    cout<<endl;
}
