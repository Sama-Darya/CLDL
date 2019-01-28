#include "Layer.h"
#include "Neuron.h"

#include <fstream>

IcoLayer::IcoLayer(int _nNeurons, int _nInputs)
{
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron
    neurons = new IcoNeuron*[nNeurons];
    /* dynamic allocation of memory to n number of
     * neuron-pointers and returning a pointer, "neurons",
     * to the first element */
    for (int i=0;i<nNeurons;i++){
        neurons[i]=new IcoNeuron(nInputs);
    }
    /* each element of "neurons" pointer is itself a pointer
     * to a neuron object with specific no. of inputs*/
}

IcoLayer::~IcoLayer(){
    for(int i=0;i<nNeurons;i++) {
        delete neurons[i];
    }
    delete[] neurons;
    //delete[] inputs;
    /* it is important to delete any dynamic
     * memory allocation created by "new" */
}

void IcoLayer::setInputs(double* _inputs){
    inputs=_inputs;
    for (int j=0; j<nInputs; j++){
        IcoNeuron** neuronsp = neurons;//point to the 1st neuron
        /* sets a temporarily pointer to neuron-pointers
         * within the scope of this function. this is inside
         * the loop, so that it is set to the first neuron
         * everytime a new value is distributed to neurons */
        double input= *inputs; //take this input value
        for (int i=0; i<nNeurons; i++){
            (*neuronsp)->setInput(j,input);
            //set this input value for this neuron
            neuronsp++; //point to the next neuron
        }
        inputs++; //point to the next input value
    }
}

void IcoLayer::propInputs(int _index, double _value){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->propInputs(_index, _value);
    }
}

void IcoLayer::calcOutputs(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->calcOutput();
    }
}

void IcoLayer::setError(double _leadError){
    /* this is only for the final layer */
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setError(_leadError);
    }
}

void IcoLayer::propError(int _neuronIndex, double _nextSum){
    neurons[_neuronIndex]->propError(_nextSum);
}

double IcoLayer::getError(int _neuronIndex){
    return (neurons[_neuronIndex]->getError());
}

double IcoLayer::getSumOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getSumOutput());
}

double IcoLayer::getWeights(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getWeights(_weightIndex));
}

double IcoLayer::getInitWeight(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getInitWeights(_weightIndex));
}

double IcoLayer::getWeightChange(){
    for (int i=0; i<nNeurons; i++){
        weightChange += neurons[i]->getWeightChange();
    }

    //cout<< "IcoLayer: WeightChange is: " << weightChange << endl;
    return (weightChange);
}

double IcoLayer::getWeightDistance(){
    weightDistance=sqrt(weightChange);
    return (weightDistance);
}

double IcoLayer::getOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getOutput());
}


void IcoLayer::initWeights(IcoNeuron::weightInitMethod _wim, IcoNeuron::biasInitMethod _bim){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->initWeights(_wim, _bim);
    }
}

void IcoLayer::setlearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setLearningRate(learningRate);
    }
}

void IcoLayer::updateWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->updateWeights();
    }
}

int IcoLayer::getnNeurons(){
    return (nNeurons);
}

int IcoLayer::saveWeights(int _layerIndex, int _neuronCount){
    char l = '0';
    char n = '0';
    l += _layerIndex + 1;
    for (int i=0; i<nNeurons; i++){
        _neuronCount += 1;
        string name = "neuronWeight";
        n += 1;
        name += l;
        name += n;
        name += ".txt";
        neurons[i]->saveWeights(name);
    }
    return (_neuronCount);
}

void IcoLayer::snapWeights(int _layerIndex){
    std::ofstream wfile;
    char l = '0';
    l += _layerIndex + 1;
    string name = "layerWeight";
    name += l;
    name += ".txt";
    wfile.open(name);
    for (int i=0; i<nNeurons; i++){
        for (int j=0; j<nInputs; j++){
            wfile << neurons[i]->getWeights(j) << " ";
        }
        wfile << "\n";
    }
    wfile.close();
}

IcoNeuron* IcoLayer::getNeuron(int _neuronIndex){
    assert(_neuronIndex < nNeurons);
    return (neurons[_neuronIndex]);
}

void IcoLayer::printLayer(){
    cout<< "\t This layer has " << nNeurons << " Neurons" <<endl;
    cout<< "\t There are " << nInputs << " inputs to this layer" <<endl;
    for (int i=0; i<nNeurons; i++){
        cout<< "\t Neuron number " << i << ":" <<endl;
        neurons[i]->printNeuron();
    }

}

