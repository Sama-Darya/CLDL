#include "../include/cldl/Net.h"
#include <algorithm>

using namespace std;

int main(){
    const int iterations = 2;
    Net *net;
    constexpr int nLayers = 3;
    int nNeurons[nLayers] = {2,2,2};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 20;
    double inputs[nInputs] = {0};
    for (int i = 0 ; i <nInputs; i++){
        inputs[i] = i;
    }
    double* inputsP = &inputs[0];
    double leadError[iterations] = {0};
    for (int i = 0 ; i <iterations; i++){
        leadError[i] = (i+1) * 1000;
    }

    double learningRate = 0.1;

    net= new Net(nLayers, nNeuronsP, nInputs, 1);
    net->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE,
                     Neuron::Act_Sigmoid);
    net->setLearningRate(learningRate);

    std::vector<int> injectionLayers;
    injectionLayers.reserve(3);
    injectionLayers = {2};

    for (int i = 0; i < iterations; i++){
        net->setInputs(inputsP);
        net->propInputs();
        // do action
        // get the error
        cout << "main: " << leadError[i] << endl;
        net->masterPropagate(injectionLayers, 0,
                             Net::BACKWARD, leadError[i],
                             Neuron::Value); // this is doing normal backpropagation
        net->updateWeights();
        net->getWeightDistance();
        cout << net->getOutput(0) << " HI " << net->getOutput(1) << endl;
    }
    delete net;
    return 0;

}
