#include "../include/cldl/Net.h"

#include <fstream>

int main()
{
    int iterations = 1;
    Net *net;
    constexpr int nLayers = 10;
    int nNeurons[nLayers] = {10,9,8,7,6,5,4,3,2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 20;
    double inputs[nInputs] = {1};
    double* inputsp = inputs;
    double leadError = 1;
    double learningRate = 1;

    net= new Net(nLayers, nNeuronsP, nInputs, 5);
    net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);
    net->setLearningRate(learningRate);
    net->setErrorCoeff(0,1,0,0,0,0);
    const int injectError_1[1] = {10}; // There won't be any propagation starting at layer index 0
    const int* injectError_1Pointer = &injectError_1[0];

    const int injectError_2[9] = {9,8,7,6,5,4,3,2,1}; // There won't be any propagation starting at layer index 0
    const int* injectError_2Pointer = &injectError_2[0];

    for (int i = 0; i < iterations; i++){
        net->setInputs(inputsp);
        net->propInputs();
        //cout << " INSPECT FORWARD PROPAGATION: ****************************************************" << endl;
        //net->printNetwork();
        net->customBackProp(injectError_1Pointer, 0); // this is doing normal backpropagation
        net->customBackProp(injectError_2Pointer, 1); // this is doing local propagation
        //net->customForwardProp(startIndexPointer, 1);
        //cout << " INSPECT BACK PROPAGATION: ****************************************************" << endl;
        //net->printNetwork();
        net->updateWeights();
        //cout << " INSPECT LEARNING: ****************************************************" << endl;
        //net->printNetwork();
        //net->saveWeights();
    }
    //net->snapWeights();

    delete net;
    return 0;

}
