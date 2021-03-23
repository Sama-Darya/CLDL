#include "../include/cldl/Net.h"

#include <fstream>

int main()
{
    int iterations = 3;
    Net *net;
    constexpr int nLayers = 10;
    int nNeurons[nLayers] = {10,9,8,7,6,5,4,3,2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 20;
    double inputs[nInputs] = {1};
    double* inputsp = inputs;
    double leadError = 1;
    double learningRate = 1;

    net= new Net(nLayers, nNeuronsP, nInputs);
    net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);
    net->setLearningRate(learningRate);
    net->setErrorCoeff(0,1,0,0,0,0);
    int startIndex[10] = {9,8,7,6,5,4,3,2,1,0}; // There won't be any propagation starting at layer index 0

    for (int i = 0; i < iterations; i++){
        net->setInputs(inputsp);
        net->propInputs();
        //cout << " INSPECT FORWARD PROPAGATION: ****************************************************" << endl;
        //net->printNetwork();
        net->setBackwardError(leadError);
        net->allInOneBackProp(startIndex);
        //cout << " INSPECT BACK PROPAGATION: ****************************************************" << endl;
        //net->printNetwork();
        net->updateWeights();
        //cout << " INSPECT LEARNING: ****************************************************" << endl;
        //net->printNetwork();
        //net->saveWeights();
    }
    net->snapWeights();

    delete net;
    return 0;

}
