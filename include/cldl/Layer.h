#pragma once
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>

#include "Neuron.h"

class Layer {
public:
    Layer(int _nNeurons, int _nInputs, int _nInternalErrors);
    ~Layer();
    void initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);
    void setLearningRate(double _learningRate);
    void setInputs(const double *_inputs);
    void propInputs(int _index, double _value);
    void calcOutputs();
    void setInternalErrors(int _internalErrorIndex, double _sumValue, int _neuronIndex);
    double getInternalErrors(int _internalErrorIndex, int _neuronIndex);
    void setErrorInputsAndCalculateInternalError(int _index, double _value, int _internalErrorIndex);
    void updateWeights();
    Neuron *getNeuron(int _neuronIndex);
    int getnNeurons();
    double getOutput(int _neuronIndex);
    double getSumOutput(int _neuronIndex);
    double getWeights(int _neuronIndex, int _weightIndex);
    double getWeightChange();
    double getWeightDistance();
    double getInitWeight(int _neuronIndex, int _weightIndex);
    void saveWeights();
    void snapWeights();
    void printLayer();

private:
    // initialisation:
    int nNeurons = 0;
    int nInputs = 0;
    double learningRate = 0;
    int myLayerIndex = 0;
    Neuron **neurons = nullptr;
    int nInternalErrors = 0;
    int layerHasReported = 0;
    const double *inputs = nullptr;
    double weightChange=0;
};