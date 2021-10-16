#include "../include/cldl/Net.h"
#include <algorithm>

using namespace std;
int main(){
//    unitTest* unitest;
//    unitest = new unitTest();
//    unitest->test_net_setInputs();
//    unitest->test_net_porpInputs();
//    unitest->test_net_masterPropagation();
const int NLAYERS = 2;
Net* net;
int nPROPAGATIONS = 1;
const int NetnInputs = 1;
double inputs[NetnInputs] = {1};
int N1 = 1;
int nNeurons[NLAYERS]={2,1};
int* nNeuronsp=nNeurons;
net = new Net(NLAYERS, nNeuronsp, NetnInputs, nPROPAGATIONS);
net->initNetwork(Neuron::W_ONES, Neuron::B_NONE, Neuron::Act_Sigmoid);
double LEARNINGRATE = 1;
net->setLearningRate(LEARNINGRATE);
std::vector<int> injectionLayers_BackProp;
injectionLayers_BackProp.reserve(1);
injectionLayers_BackProp = {1};
double error = 1;
double* pred_pointer = &inputs[0];
net->setInputs(pred_pointer);
net->propInputs();
net->masterPropagate(injectionLayers_BackProp, 0,
                     Net::BACKWARD, error,
                     Neuron::Value);
net->printNetwork();
net->updateWeights();
cout << "*********** updateWeights ***********" << endl;
net->printNetwork();
return 0;
}
