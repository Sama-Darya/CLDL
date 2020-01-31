#include "clbp/Neuron.h"

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
// constructor de-constructor
//*************************************************************************************

Neuron::Neuron(int _nInputs)
{
    nInputs=_nInputs;
    weights = new double[nInputs];
    initialWeights = new double[nInputs];
    inputs = new double[nInputs];
    inputErrors = new double[nInputs];
}

Neuron::~Neuron(){
    delete [] weights;
    delete [] initialWeights;
    delete [] inputs;
    delete [] inputErrors;
}

//*************************************************************************************
//initialisation:
//*************************************************************************************

void Neuron::initNeuron(int _neuronIndex, int _layerIndex, weightInitMethod _wim, biasInitMethod _bim, Neuron::actMethod _am){
    myLayerIndex = _layerIndex;
    myNeuronIndex = _neuronIndex;
    for (int i=0; i<nInputs; i++){
        switch (_wim){
            case W_ZEROS:
                weights[i]=0;
                break;
            case W_ONES:
                weights[i]=1;
                break;
            case W_RANDOM:
                weights[i]=(((double)rand()/(RAND_MAX)));// * 2)-1;
                break;
                //cout << " Neuron: weight is: " << weights[i] << endl;
                /* rand function generates a random function between
                 * 0 and RAND_MAX, after the devision the weights are
                 * set to a value between 0 and 1 */
        }
        initialWeights[i] = weights[i];
        weightSum = 0;
        for (int i=0; i<nInputs; i++){
            weightSum += fabs(weights[i]);
            maxWeight = max(maxWeight, weights[i]);
            minWeight = min (minWeight, weights[i]);
        }

    }
    switch (_bim){
        case B_NONE:
            bias=0;
            break;
        case B_RANDOM:
            bias=((double)rand()/RAND_MAX);
            break;
    }
    switch(_am){
        case Act_Sigmoid:
            actMet = 0;
            break;
        case Act_Tanh:
            actMet = 1;
            break;
        case Act_NONE:
            actMet = 2;
            break;
    }
}

void Neuron::setLearningRate(double _learningRate){
    learningRate=_learningRate;
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

void Neuron::setInput(int _index,  double _value) {
    /* the seInput function sets one input value at the given index,
     * it has to be implemented in a loop inside the layer class to set
     * all the inputs associated with all the neurons in that layer*/
    assert((_index>=0)&&(_index<nInputs));
    /*checking _index is a valid int, non-negative and within boundary*/
    inputs[_index] = _value;
    //cout << "Neuron the input is: " << _value << endl;
}

void Neuron::propInputs(int _index,  double _value){
    /*works like setInput function expect it only applies
     * to the neurons in the hidden and output layers
     * and not the input layer*/
    assert((_index>=0)&&(_index<nInputs));
    inputs[_index] = _value;
}

void Neuron::calcOutput(){
    sum=0;
    for (int i=0; i<nInputs; i++){
        sum += inputs[i] * weights[i];
    }
    sum += bias;
    if (myLayerIndex == 0){
        sum = sum / 200;
    }
    assert(std::isfinite(sum));
    output = doActivation(sum);
    assert(std::isfinite(output));
}

//*************************************************************************************
//forward propagation of error:
//*************************************************************************************

void Neuron::setErrorAtInput(double _value) {
    for(int i=0; i<nInputs; i++){
        inputErrors[i] = _value;
    }
}

void Neuron::propErrorForward(int _index,  double _value){
    assert((_index>=0)&&(_index<nInputs));
    inputErrors[_index] = _value;
}

void Neuron::calcErrorOutput(){
    double errorSum =0;
    for (int i=0; i<nInputs; i++) {
        errorSum += inputErrors[i] * weights[i];
    }
    assert(std::isfinite(errorSum));
    forwardError = errorSum * doActivationPrime(sum); //could do normalisation here
    assert(std::isfinite(output));
}

double Neuron::getForwardError(){
    return (forwardError);
}

//*************************************************************************************
//back propagation of error
//*************************************************************************************

void Neuron::setError(double _leadError){
    backwardError = _leadError * doActivationPrime(sum);
    //cout << backwardError << endl;
    assert(std::isfinite(backwardError));
    /*might take a different format to propError*/
}

void Neuron::propError(double _nextSum){
    backwardError = _nextSum * doActivationPrime(sum);
    assert(std::isfinite(_nextSum));
}

double Neuron::getBackwardError(){
    return (backwardError);
}

//*************************************************************************************
//learning
//*************************************************************************************

void Neuron::updateWeights(){
    weightSum = 0;
    maxWeight = 0;
    minWeight = 0;
    double force = 1;
    if (myLayerIndex == 0){
        force  = 1; //forces a bigger change on the first layer for visualisation in greyscale
    }
    for (int i=0; i<nInputs; i++){
        weights[i] += learningRate * (backwardError * forwardError) * inputs[i];
        weightSum += fabs(weights[i]);
        maxWeight = max (maxWeight,weights[i]);
        minWeight = min (maxWeight,weights[i]);
    }
    if (myLayerIndex == 0 && myNeuronIndex == 0){
        cout << backwardError << " * " << forwardError << endl;
    }
}

double Neuron::doActivation(double _sum){
    double thisoutput = 0;
    switch(actMet){
        case 0:
            thisoutput = (1/(1+(exp(-_sum)))) - 0.5;
            break;
        case 1:
            thisoutput = tanh(_sum);
            break;
        case 2:
            thisoutput = _sum;
            break;
    }
    return (thisoutput);
}

double Neuron::doActivationPrime(double _input){
    double result = 0;
    switch(actMet){
        case 0:
            result = 1 * (0.5 + doActivation(_input)) * (0.5 - doActivation(_input)); //exp(-_input) / pow((exp(-_input) + 1),2);
            break;
        case 1:
            result = 1 - pow (tanh(_input), 2);
            break;
        case 2:
            result = 1;
            break;
    }
    return (result);
}

//*************************************************************************************
//global settings
//*************************************************************************************

void Neuron::setGlobalError(double _globalError){
  globalError = _globalError;
}

//*************************************************************************************
// getters:
//*************************************************************************************

double Neuron::getOutput(){
    return (output);
}

double Neuron::getSumOutput(){
    return (sum);
}

double Neuron::getMaxWeight(){
  return maxWeight;
}

double Neuron::getMinWeight(){
  return minWeight;
}

double Neuron::getSumWeight(){
  return weightSum;
}

double Neuron::getWeightChange(){
    weightsDifference = 0;
    weightChange = 0;
    for (int i=0; i<nInputs; i++){
        weightsDifference = weights[i] - initialWeights[i];
        weightChange += pow(weightsDifference,2);
    }
    return (weightChange);
}

double Neuron::getWeightDistance(){
    return sqrt(weightChange);
}

double Neuron::getGlobalError(){
    return (globalError);
}

int Neuron::getnInputs(){
    return (nInputs);
}

double Neuron::getWeights(int _inputIndex){
    assert((_inputIndex>=0)&&(_inputIndex<nInputs));
    return (weights[_inputIndex]);
}

double Neuron::getInitWeights(int _inputIndex){
    assert((_inputIndex>=0)&&(_inputIndex<nInputs));
    return (initialWeights[_inputIndex]);
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

void Neuron::saveWeights(){
  char l = '0';
  char n = '0';
  l += myLayerIndex + 1;
  n += myNeuronIndex + 1;
  string name = "w";
  name += 'L';
  name += l;
  name += 'N';
  name += n;
  name += ".csv";
  std::ofstream Icofile;
  Icofile.open(name, fstream::app);
  for (int i=0; i<nInputs; i++){
    Icofile << weights[i] << " " ;
  }
  Icofile << "\n";
  Icofile.close();
}

void Neuron::printNeuron(){
    cout<< "\t \t This neuron has " << nInputs << " inputs:";
    for (int i=0; i<nInputs; i++){
        cout<< " " << inputs[i];
    }
    cout<<endl;
    cout<< "\t \t The weights for those inputs are:";
    for (int i=0; i<nInputs; i++){
        cout<< " " << weights[i];
    }
    cout<<endl;
    cout<< "\t \t The bias of the neuron is: " << bias << endl;
    cout<< "\t \t The sum and output of this neuron are: " << sum << ", " << output << endl;
}
