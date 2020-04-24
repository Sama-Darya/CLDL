#include "../include/cldl/Net.h"

#include <fstream>

int main()
{
    int line = 0;
    Net *net;
    constexpr int nLayers=11;
    int nNeurons[nLayers]={11,10,9,8,7,6,5,4,3,2,1};
    int* nNeuronsP=nNeurons;
    constexpr int nInputs=3;
    double inputs[nInputs]={1,1,1};
    double* inputsp=inputs;
    double leadError=0;
    double learningRate=1;

    net= new Net(nLayers, nNeuronsP, nInputs);
    net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);
    net->setLearningRate(learningRate);
    //net->setErrorCoeff(0,1,2,3,4);


    for (int i=0; i<1; i++){
        net->setInputs(inputsp);
        net->propInputs();
//        net->setGlobalError(leadError);
//        net->setBackwardError(leadError);
//        net->propErrorBackward();
//        net->setForwardError(leadError);
//        net->propErrorForward();
//        net->setMidError(1,leadError);
//        net->propMidErrorForward();
//        net->propMidErrorBackward();
        net->setLocalError(leadError);
        net->propGlobalErrorBackwardLocally();

//        net->doEchoError(leadError);
//        net->updateWeights();
        //cout << "THE GRADIENT: " << net->getGradient(Neuron::onBackwardError, Layer::exploding);
    }

//    net->setInputs(inputsp);
//    net->propInputs();
//    cout<< " INSPECTION 1: ****************************************************" << endl;
//    net->printNetwork();
//
//    net->setBackwardError(leadError);
//    net->propErrorBackward();
//    net->updateWeights();
//    cout<< " INSPECTION 2: ****************************************************" << endl;
//    net->printNetwork();
//
//    net->setInputs(inputsp);
//    net->propInputs();
//    cout<< " INSPECTION 3: ****************************************************" << endl;
//    net->printNetwork();

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
