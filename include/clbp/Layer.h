#pragma once
#include "Neuron.h"


class IcoLayer
{
public:
    IcoLayer(int _nNeurons, int _nInputs);
    ~IcoLayer();

    void setInputs( double* _inputs); //only for the first layer
    void initWeights(IcoNeuron::weightInitMethod _wim, IcoNeuron::biasInitMethod _bim);
    void calcOutputs();
    double getOutput(int _neuronIndex);
    double getSumOutput(int _neuronIndex);
    void propInputs(int _index, double _value);
        /*this is for hidden and output layers (not input)*/
    void printLayer();
    void propError(int _neuronIndex, double _nextSum);
    int getnNeurons();
    void setlearningRate(double _learningRate);
    double getError(int _neuronIndex);
    double getWeights(int _neuronIndex, int _weightIndex);
    double getInitWeight(int _neuronIndex, int _weightIndex);
    double getWeightDistance();
    double getWeightChange();
    void setError(double _leadError);
    void updateWeights();
    int saveWeights(int _layerIndex, int _neuronCount);
    void snapWeights(int _layerIndex); // This one just saves the final weights i.e. overwrites them

    IcoNeuron* getNeuron(int _neuronIndex);

private:
    int nNeurons=0;
    int nInputs=0;
    double* inputs=0;
    IcoNeuron** neurons=0;
    double learningRate=0;
    double weightDistance=0;
    double weightChange=0;
};
