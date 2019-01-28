#include "Net.h"
#include <fstream>

int main()
{
    int line = 0;
    IcoNet *icoNet;
    constexpr int nLayers=2;
    int nNeurons[nLayers]={2,1};
    int* nNeuronsP=nNeurons;
    constexpr int nInputs=3;
    double inputs[nInputs]={0,1,0};
    double* inputsp=inputs;
    double leadError=-10;
    double learningRate=0.01;

    icoNet= new IcoNet(nLayers, nNeuronsP, nInputs);
    icoNet->initWeights(IcoNeuron::W_ONES, IcoNeuron::B_NONE);
    icoNet->setLearningRate(learningRate);

    icoNet->initWeights(IcoNeuron::W_ONES, IcoNeuron::B_NONE);
    icoNet->setInputs(inputsp);
    icoNet->propInputs();
    icoNet->setError(leadError);
    icoNet->propError();
    icoNet->printNetwork();
    icoNet->updateWeights();
    icoNet->saveWeights();
    double weightDistance=icoNet->getWeightDistance();
    line++;
    std::ofstream myfile2;
    myfile2.open ("weights2.txt", fstream::app);
    myfile2 << line << " " << weightDistance << "\n";
    myfile2.close();

    delete icoNet;
    return 0;

}
