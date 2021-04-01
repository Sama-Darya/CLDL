#include "../include/cldl/Net.h"
#include <algorithm>

using namespace std;

int main(){
    const int iterations = 10;
    Net *net;
    constexpr int nLayers = 10;
    int nNeurons[nLayers] = {10,9,8,7,6,5,4,3,2,2};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 20;
    double inputs[nInputs] = {0};
    for (int i = 0 ; i <nInputs; i++){
        inputs[i] = i;
    }
    double* inputsP = &inputs[0];
    double leadError[iterations] = {0};
    for (int i = 0 ; i <iterations; i++){
        leadError[i] = i * 500;
    }

    double learningRate = 0.1;

    net= new Net(nLayers, nNeuronsP, nInputs, 2);
    net->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE,
                     Neuron::Act_Sigmoid);
    net->setLearningRate(learningRate);

    std::vector<int> injectionLayers;
    injectionLayers.reserve(3);
    injectionLayers = {2,9,1,0};

    for (int i = 0; i < iterations; i++){
        net->setInputs(inputsP);
        net->propInputs();
        net->masterPropagate(injectionLayers, 0,
                             Net::BACKWARD, leadError[iterations],
                             Neuron::Sign); // this is doing normal backpropagation
        net->masterPropagate(injectionLayers, 1,
                             Net::BACKWARD, leadError[iterations],
                             Neuron::Absolute); // this is doing local propagation
        net->updateWeights();
        cout << net->getOutput(0) << "  " << net->getOutput(1) << endl;
    }
    delete net;
    return 0;

}
