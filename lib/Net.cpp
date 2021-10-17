#include "../include/cldl/Net.h"
#include "../include/cldl/Layer.h"
#include "../include/cldl/Neuron.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
using namespace std;



//*************************************************************************************
//initialisation:
//*************************************************************************************

Net::Net(int _nLayers, int* _nNeurons, int _nInputs, int _nInternalErrors){
    cout << "*******************************************************************************************************" << endl;
    nLayers = _nLayers; //no. of layers including inputs and outputs layers
    layers= new Layer*[nLayers];
    nInternalErrors = _nInternalErrors;
    int* nNeuronsp = _nNeurons; //number of neurons in each layer
    nInputs=_nInputs; // the no. of inputs to the network (i.e. the first layer)
    //cout << "nInputs: " << nInputs << endl;
    int nInput = 0; //temporary variable to use within the scope of for loop
    for (int i=0; i<nLayers; i++){
        int numNeurons= *nNeuronsp; //no.
        // neurons in this layer
        if (i==0){nInput=nInputs;}
        /* no. inputs to the first layer is equal to no. inputs to the network */
        layers[i]= new Layer(numNeurons, nInput, nInternalErrors);
        nNeurons += numNeurons;
        nWeights += (numNeurons * nInput);
        nInput=numNeurons;
        /*no. inputs to the next layer is equal to the number of neurons
         * in the current layer. */
        nNeuronsp++; //point to the no. of neurons in the next layer
    }
    nOutputs=layers[nLayers-1]->getnNeurons();
}

Net::~Net(){
    for (int i=0; i<nLayers; i++){
        delete layers[i];
    }
    delete[] layers;
}

void Net::initNetwork(Neuron::weightInitMethod _wim,
                      Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    for (int i=0; i<nLayers; i++){
        layers[i]->initLayer(i, _wim, _bim, _am);
    }
}

void Net::setLearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nLayers; i++){
        layers[i]->setLearningRate(learningRate);
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
    /* this calculates the final output of the network,
     * i.e. the output of the final layer
     * but this is not fed into any further layer*/
}

void Net::masterPropagate(std::vector<int> &injectionLayerIndex,
                          int _internalErrorIndex, propagationDirection _propDir,
                          double _controlError, Neuron::errorMethod _errorMethod){
    switch(_propDir){
        case BACKWARD:
            std::sort(injectionLayerIndex.rbegin(), injectionLayerIndex.rend());
            customBackProp(injectionLayerIndex, _internalErrorIndex,
                           _controlError, _errorMethod);
            break;
        case FORWARD:
            std::sort(injectionLayerIndex.begin(), injectionLayerIndex.end());
            customForwardProp(injectionLayerIndex, _internalErrorIndex,
                              _controlError, _errorMethod);
            break;
    }
}

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************
void Net::customForwardProp(std::vector<int> &injectionLayerIndex,
                            int _internalErrorIndex, double _controlError,
                            Neuron::errorMethod _errorMethod){
    assert(injectionLayerIndex[0] == 0 && "Forward propagation must start form the first layer, include (0) in your array");
    int injectionCount = 0;
    int nextInjectionLayerIndex = injectionLayerIndex[0];
    controlError = _controlError;
    for(int i = 0; i < layers[nextInjectionLayerIndex]->getnNeurons(); i++){
        layers[nextInjectionLayerIndex]->setInternalErrors(_internalErrorIndex,
                                                           controlError, i, _errorMethod); // setting the internal errors in the first layer
    }
    double inputOutput = 0.00;
    for (int L_index=nextInjectionLayerIndex; L_index<nLayers-1; L_index++){
        for (int N_index=0; N_index<layers[L_index]->getnNeurons(); N_index++){
            if(L_index == nextInjectionLayerIndex){
                assert((injectionCount<nLayers)&&(injectionCount>=0) && "NET failed");
                inputOutput = controlError;
                injectionCount += 1;
                nextInjectionLayerIndex = injectionLayerIndex[injectionCount];
            }else{
                inputOutput = layers[L_index]->getInternalErrors(_internalErrorIndex, N_index);
            }
            layers[L_index+1]->setErrorInputsAndCalculateInternalError(N_index,
                                                                       inputOutput, _internalErrorIndex,
                                                                       _errorMethod);
        }
    }
}

void Net::customBackProp(std::vector<int> &injectionLayerIndex,
                         int _internalErrorIndex, double _controlError,
                         Neuron::errorMethod _errorMethod){
    assert(injectionLayerIndex[0] == nLayers-1 && "Backpropagation must start form the last layer, include (Nlayers - 1) in your array");
    double tempError = 0;
    double tempWeight = 0;
    int nextInjectionLayerIndex = injectionLayerIndex[0];
    int injectionCount = 0;
    controlError = _controlError;
    for(int neuronIndex=0; neuronIndex < layers[nextInjectionLayerIndex]->getnNeurons(); neuronIndex++){ //set the internal error in the final layer
        layers[nextInjectionLayerIndex]->setInternalErrors(_internalErrorIndex,
                                                           controlError, neuronIndex, _errorMethod);
    }

    for (int L_index = nextInjectionLayerIndex; L_index > 0 ; L_index--){ //iterate through the layers
        cout << "for loop: " <<  L_index << endl;
        bpThread **myBPThread = nullptr;
        int wn_index = layers[L_index-1]->getnNeurons();
        cout << wn_index << endl;
        myBPThread = new bpThread*[wn_index];
        for (int i = 0; i < wn_index; i++){
//            cout << "i: " << i << endl;
            myBPThread[i] = new bpThread(i);
        }
        for (int i = 0; i < wn_index; i++ ){
//            cout << "i: " << i << endl;
            myBPThread[i]->start();
        }
        for (int i = 0; i < wn_index; i++ ){
            myBPThread[i]->join();
            delete myBPThread[i];
        }
        delete myBPThread;
    }

}

//void Net::customBackProp(std::vector<int> &injectionLayerIndex,
//                         int _internalErrorIndex, double _controlError,
//                         Neuron::errorMethod _errorMethod){
//    assert(injectionLayerIndex[0] == nLayers-1 && "Backpropagation must start form the last layer, include (Nlayers - 1) in your array");
//    double tempError = 0;
//    double tempWeight = 0;
//    int nextInjectionLayerIndex = injectionLayerIndex[0];
//    int injectionCount = 0;
//    controlError = _controlError;
//    for(int neuronIndex=0; neuronIndex < layers[nextInjectionLayerIndex]->getnNeurons(); neuronIndex++){ //set the internal error in the final layer
//        layers[nextInjectionLayerIndex]->setInternalErrors(_internalErrorIndex,
//                                                           controlError, neuronIndex, _errorMethod);
//    }
//    for (int L_index = nextInjectionLayerIndex; L_index > 0 ; L_index--){ //iterate through the layers
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
//          }
//    }
//}

void Net::updateWeights(){
    for (int i=nLayers-1; i>=0; i--){
        layers[i]->updateWeights();
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
    assert(_layerIndex<nLayers && "NET failed");
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

double Net::getInputs(int _inputIndex){
    return inputs[_inputIndex];
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

void Net::saveWeights(){
    for (int i=0; i<nLayers; i++){
        layers[i]->saveWeights();
    }
}

void Net::snapFistLayerWeights(){
        layers[0]->snapWeights();
}

void Net::snapWeights(){
    for (int i=0; i<nLayers; i++){
        layers[i]->snapWeights();
    }
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
