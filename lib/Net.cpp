#include "Net.h"
#include "Layer.h"
#include "Neuron.h"

#include <iostream>

using namespace std;

IcoNet::IcoNet(int _nLayers, int* _nNeurons, int _nInputs)
{
    nLayers = _nLayers; //no. of layers including inputs and ouputs layers
    layers= new IcoLayer*[nLayers];
    int* nNeuronsp = _nNeurons; //number of neurons in each layer expect input
    nInputs=_nInputs; // the no. of inputs to the network (i.e. the first layer)
    int nInput = 0; //temporary variable to use within the scope of for loop
    for (int i=0; i<nLayers; i++){
        int nNeurons= *nNeuronsp; //no. neurons in this layer
        if (i==0){nInput=nInputs;}
        /* no. inputs to the first layer is equal to no. inputs to the network */
        layers[i]= new IcoLayer(nNeurons, nInput);
        nInput=nNeurons;
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
}

IcoNet::~IcoNet(){
    for (int i=0; i<nLayers; i++){
        delete layers[i];
    }
    delete[] layers;
    //delete[] inputs;
}

void IcoNet::setInputs(double* _inputs){
    inputs=_inputs;
    layers[0]->setInputs(inputs); //sets the inputs to the first layer only
}

void IcoNet::initWeights(IcoNeuron::weightInitMethod _wim, IcoNeuron::biasInitMethod _bim){
    for (int i=0; i<nLayers; i++){
        layers[i]->initWeights(_wim, _bim);
    }
}

void IcoNet::propInputs(){
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

double IcoNet::getOutput(int _neuronIndex){
    return (layers[nLayers-1]->getOutput(_neuronIndex));
}

double IcoNet::getSumOutput(int _neuronIndex){
    return (layers[nLayers-1]->getSumOutput(_neuronIndex));
}

int IcoNet::getnLayers(){
    return (nLayers);
}

int IcoNet::getnInputs(){
    return (nInputs);
}

IcoLayer* IcoNet::getLayer(int _layerIndex){
    assert(_layerIndex<nLayers);
    return (layers[_layerIndex]);
}

void IcoNet::propError(){
    double sum=0;
    double tempError=0;
    double tempWeight=0;
    for (int i=nLayers-1; i>0 ; i--){
        for (int k=0; k<layers[i-1]->getnNeurons();k++){
            for (int j=0; j<layers[i]->getnNeurons(); j++){
                tempError=layers[i]->getError(j);
                tempWeight=layers[i]->getWeights(j,k);
                sum+=tempError * tempWeight;
            }
            layers[i-1]->propError(k, sum);
        }
    }
}

void IcoNet::setError(double _leadError){
    /* this is only for the final layer */
    layers[nLayers-1]->setError(_leadError);
    /* if the leadError was diff. for each output neuron
     * then it would be implemented in a for-loop */
}

void IcoNet::updateWeights(){
    for (int i=nLayers-1; i>=0; i--){
        layers[i]->updateWeights();
    }
}

double IcoNet::getWeightDistance(){
    for (int i=0; i<nLayers; i++){
        weightChange += layers[i]->getWeightChange();
    }
    weightDistance=sqrt(weightChange);
    cout<< "IcoNet: WeightDistance is: " << weightDistance << endl;

    return (weightDistance);
}

double IcoNet::getWeights(int _layerIndex, int _neuronIndex, int _weightIndex){
    double weight=layers[_layerIndex]->getWeights(_neuronIndex, _weightIndex);
    return (weight);
}

void IcoNet::setLearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nLayers; i++){
        layers[i]->setlearningRate(learningRate);
    }
}

int IcoNet::getnNeurons(){
    return (nNeurons);
}

void IcoNet::saveWeights(){
    int neuronCount = 0;
    for (int i=0; i<nLayers; i++){
        neuronCount += layers[i]->saveWeights(i, neuronCount);
        layers[i]->snapWeights(i);
    }
}

void IcoNet::printNetwork(){
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
