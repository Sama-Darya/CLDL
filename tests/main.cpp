#include "clbp/Net.h"

#include <fstream>

int main()
{
    int line = 0;
    Net *net;
    constexpr int nLayers=2;
    int nNeurons[nLayers]={2,1};
    int* nNeuronsP=nNeurons;
    constexpr int nInputs=3;
    double inputs[nInputs]={0,1,0};
    double* inputsp=inputs;
    double leadError=-10;
    double learningRate=0.01;

    net= new Net(nLayers, nNeuronsP, nInputs);
    net->initWeights(Neuron::W_ONES, Neuron::B_NONE);
    net->setLearningRate(learningRate);

    net->initWeights(Neuron::W_ONES, Neuron::B_NONE);
    net->setInputs(inputsp);
    net->propInputs();
    net->setError(leadError);
    net->propError();
    net->printNetwork();
    net->updateWeights();
    net->saveWeights();
    double weightDistance=net->getWeightDistance();
    line++;
    std::ofstream myfile2;
    myfile2.open ("weights2.txt", fstream::app);
    myfile2 << line << " " << weightDistance << "\n";
    myfile2.close();

    delete net;
    return 0;

}
