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

Net::Net(int _nLayers, int* _nNeurons, int _nInputs)

{
    nLayers = _nLayers; //no. of layers including inputs and ouputs layers
    layers= new Layer*[nLayers];
    int* nNeuronsp = _nNeurons; //number of neurons in each layer expect input
    nInputs=_nInputs; // the no. of inputs to the network (i.e. the first layer)
    int nInput = 0; //temporary variable to use within the scope of for loop
    for (int i=0; i<nLayers; i++){
        int numNeurons= *nNeuronsp; //no. neurons in this layer
        if (i==0){nInput=nInputs;}
        /* no. inputs to the first layer is equal to no. inputs to the network */
        layers[i]= new Layer(numNeurons, nInput);
        nNeurons += numNeurons;
        nWeights += (numNeurons * nInput);
        nInput=numNeurons;
        /*no. inputs to the next layer becomes is equal to the number of neurons
         * in the current layer. */
        nNeuronsp++; //point to the no. of neurons in the next layer
    }
    nOutputs=layers[nLayers-1]->getnNeurons();
    //inputs= new double[nInputs];

    nNeurons=0;
    for (int i=0; i<nLayers; i++){
        nNeurons += layers[i]->getnNeurons();
    }

    //cout << "number of inputs are: " << nInputs << endl;
}

Net::~Net(){
    for (int i=0; i<nLayers; i++){
        delete layers[i];
    }
    delete[] layers;
    //delete[] inputs;
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

void Net::setErrorAtInput(double _leadForwardError){
    leadForwardError=_leadForwardError;
    layers[0]->setErrorAtInput(leadForwardError);
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
            double_t sum = 0.0;
            double_t normSum = 0.0;
            double_t weightSumer = 0.0;
            int counter = 0;
            for (int j = 0; j < layers[i]->getnNeurons(); j++){
                tempError = layers[i]->getBackwardError(j);
                tempWeight = layers[i]->getWeights(j,k);
                sum += (tempError * tempWeight);
                weightSumer += fabs(tempWeight);
                counter += 1;
            }
            normSum = sum ; // / weightSumer;
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
            double_t sum = 0.0;
            double_t normSum = 0.0;
            double_t weightSumer = 0.0;
            int counter = 0;
            for (int j = 0; j < layers[i]->getnNeurons(); j++){
                tempError = layers[i]->getMidError(j);
                tempWeight = layers[i]->getWeights(j,k);
                sum += (tempError * tempWeight);
                weightSumer += fabs(tempWeight);
                counter += 1;
            }
            normSum = sum ; // / weightSumer;
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
//learning:
//*************************************************************************************

void Net::setErrorCoeff(double _globalCoeff, double _backwardsCoeff, double _midCoeff, double _forwardCoeff){
    backwardsCoeff = _backwardsCoeff;
    midCoeff = _midCoeff;
    forwardCoeff =_forwardCoeff;
    globalCoeff = _globalCoeff;
    for (int i=0; i<nLayers; i++){
        layers[i]->setErrorCoeff(globalCoeff, backwardsCoeff, midCoeff, forwardCoeff);
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
