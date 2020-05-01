#include "../include/cldl/Net.h"

#include <fstream>

int main()
{
    int iterations = 2;
    Net *net;
    constexpr int nLayers = 2;
    int nNeurons[nLayers] = {2,1};
    int* nNeuronsP = nNeurons;
    constexpr int nInputs = 3;
    double inputs[nInputs] = {1,1,1};
    double* inputsp = inputs;
    double leadError = 1;
    double learningRate = 1;
    cout << "no way" << endl;

    net= new Net(nLayers, nNeuronsP, nInputs);
    net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);
    net->setLearningRate(learningRate);
    net->setErrorCoeff(0,1,0,0,0,0);

    for (int i = 0; i < iterations; i++){
        net->setInputs(inputsp);
        net->propInputs();
        cout << " INSPECT FORWARD PROPAGATION: ****************************************************" << endl;
        net->printNetwork();
        net->setBackwardError(leadError);
        net->propErrorBackward();
        cout << " INSPECT BACK PROPAGATION: ****************************************************" << endl;
        net->printNetwork();
        net->updateWeights();
        cout << " INSPECT LEARNING: ****************************************************" << endl;
        net->printNetwork();
        net->saveWeights();
    }
    net->snapWeights();

    delete net;
    return 0;

}
