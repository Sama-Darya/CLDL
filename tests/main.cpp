#include "../include/cldl/Net.h"

#include <fstream>
#include <algorithm>
#include <array>

using namespace std;

int main(){
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

    std::vector<int> injectionLayers;
    injectionLayers.reserve(7);
    injectionLayers = {8,9,2,6,3,5,0};

    for (int i = 0; i < iterations; i++){
        net->setInputs(inputsp);
        net->propInputs();
        net->masterPropagate(injectionLayers, 0, Net::BACKWARD, leadError); // this is doing normal backpropagation
        net->masterPropagate(injectionLayers, 1, Net::FORWARD, leadError); // this is doing local propagation
        net->updateWeights();
    }


    delete net;
    return 0;

}
