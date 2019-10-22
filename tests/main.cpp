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
    double inputs[nInputs]={1,1,1};
    double* inputsp=inputs;
    double leadError=-10;
    double learningRate=1;

    net= new Net(nLayers, nNeuronsP, nInputs);
    net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);
    net->setLearningRate(learningRate);


    net->setInputs(inputsp);
    net->propInputs();
    cout<< " INSPECTION 1: " << endl;
    net->printNetwork();

    net->setError(leadError);
    net->propError();
    net->updateWeights();
    cout<< " INSPECTION 2: " << endl;
    net->printNetwork();

    net->setInputs(inputsp);
    net->propInputs();
    cout<< " INSPECTION 3: " << endl;
    net->printNetwork();

//    net->saveWeights();
//    double weightDistance=net->getWeightDistance();
//    line++;
//    std::ofstream myfile2;
//    myfile2.open ("weights2.txt", fstream::app);
//    myfile2 << line << " " << weightDistance << "\n";
//    myfile2.close();

    delete net;
    return 0;

}
