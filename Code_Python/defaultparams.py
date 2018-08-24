'''
 Main entry

 Copyright (c) 2012, "Simon Brodeur"
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:
 
  - Redistributions of source code must retain the above copyright notice, 
    this list of conditions and the following disclaimer.
  - Redistributions in binary form must reproduce the above copyright notice, 
    this list of conditions and the following disclaimer in the documentation 
    and/or other materials provided with the distribution.
  - Neither the name of the NECOTIS research group nor the names of its contributors 
    may be used to endorse or promote products derived from this software 
    without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
 NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
 OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 POSSIBILITY OF SUCH DAMAGE.
'''

from brian import *
import numpy as np
from rankorder import *
import scipy.io


def createDefaultTimitInput(audioDuration, offset, microcircuit, tuner, extractParams, inputModelParams, showPlot=True):
    from learning import InputConnection
    from main_timit import auditoryFeatureExtractionModel

    audioFilename = 'data/timitAudio.wav'
    input, features = auditoryFeatureExtractionModel(audioFilename, audioDuration, offset, extractParams,
                                                     inputModelParams, showPlot)
    inputConnections = InputConnection(input, microcircuit.microcircuit, state='v', microcircuit=microcircuit,
                                       tuner=tuner)
    inputWeightFunc = lambda i, j: numpy.random.uniform(0.5, 2.0)

    # Number of average input synaptic connections is fixed to 10% of reservoir links
    nbInputs = len(input)

    fracInput = 0.1
    nbInputConn = int(fracInput * microcircuit.connections.W.nnz)
    sparseness = float(nbInputConn) / float(nbInputs * len(microcircuit.microcircuit))
    inputConnections.connect_random(input, microcircuit.microcircuit, sparseness=sparseness, weight=inputWeightFunc)
    return (input, inputConnections, features)


def createDefaultPoissonInput(microcircuit, tuner):
    from learning import InputConnection

    nbInputs = 64
    theta0 = 2 * numpy.pi * numpy.random.rand(nbInputs)
    r0 = linspace(25 * Hz, 50 * Hz, nbInputs)
    fAmpMod = 0.5 * Hz
    input = PoissonGroup(nbInputs, rates=lambda t: (1 + cos(2 * numpy.pi * fAmpMod * t + theta0)) * r0)
    inputConnections = InputConnection(input, microcircuit.microcircuit, state='v', microcircuit=microcircuit,
                                       tuner=tuner)
    inputWeightFunc = lambda i, j: numpy.random.uniform(0.5, 2.0)

    # Number of average input synaptic connections is fixed to 10% of reservoir links
    fracInput = 0.1
    nbInputConn = int(fracInput * microcircuit.connections.W.nnz)
    sparseness = float(nbInputConn) / float(nbInputs * len(microcircuit.microcircuit))
    inputConnections.connect_random(input, microcircuit.microcircuit, sparseness=sparseness, weight=inputWeightFunc)
    return (input, inputConnections)


def createDefaultRandomPatternInput(duration, microcircuit, tuner, patterns=None):
    from learning import InputConnection

    # Generate patterns if not provided
    if patterns == None:
        nbNeurons = len(microcircuit.microcircuit)
        nbActiveNeurons = 32
        nbPatterns = 2
        width = 20 * ms
        spiketimes = generateRandomPatterns(nbNeurons, nbActiveNeurons, nbPatterns, width, delayEpoch=1 * ms)
        patterns = Patterns(nbNeurons, nbActiveNeurons, nbPatterns, spiketimes, width)

    # Generate random sequence of patterns
    seqSpiketimes = []
    seq = []
    nbEpochs = int(duration / patterns.width)
    for n in range(nbEpochs):
        p = numpy.random.randint(0, patterns.nbPatterns)
        seq.append(p)
        for i, t in patterns.spiketimes[p]:
            seqSpiketimes.append((i, t + n * patterns.width))

    input = SpikeGeneratorGroup(patterns.nbNeurons, seqSpiketimes)
    inputConnections = InputConnection(input, microcircuit.microcircuit, state='v', microcircuit=microcircuit,
                                       tuner=tuner)

    # FIXME: don't choose inhibitory neurons
    neuronIdx = range(len(microcircuit.microcircuit))
    numpy.random.shuffle(neuronIdx)
    for i in range(len(input)):
        inputConnections.W[i, neuronIdx[i]] = 2.0

    return (input, inputConnections, seq, seqSpiketimes, patterns)


# function create by MarcAntoine
def sequenceInputEeg(microcircuit, tuner, seqIdx=None):
    from learning import InputConnection

    # import spike sequences from matlab
    f = scipy.io.loadmat('dataset_simon.mat')
    finalSequence = f["final_output_spike"]

    # index unique
    names = scipy.io.loadmat('label_dataset_simon.mat')['v'][0]
    classidx = 0
    labelmap = {}
    labels = []

    for name in names:
        name = name[0][0]
        if name not in labelmap:
            labelmap[name] = classidx
            classidx += 1
        labels.append(labelmap[name])

    # Generate random sequence of patterns
    seqSpiketimes = []
    dt = 2.0 * ms
    nbNeurons = finalSequence.shape[0]
    trialDelay = 1000 * ms
    t = 0.0

    if seqIdx is None:
        for s in range(finalSequence.shape[2]):
            for n in range(finalSequence.shape[1]):
                idx = np.where(finalSequence[:, n, s] > 0)[0]
                for i in idx:
                    seqSpiketimes.append((i, t))
                t += dt
            t += trialDelay
    else:
        for n in range(finalSequence.shape[1]):
            idx = np.where(finalSequence[:, n, seqIdx] > 0)[0]
            for i in idx:
                seqSpiketimes.append((i, t))
            t += dt

    input = SpikeGeneratorGroup(nbNeurons, seqSpiketimes)
    inputConnections = InputConnection(input, microcircuit.microcircuit, state='v', microcircuit=microcircuit,
                                       tuner=tuner)

    # FIXME: don't choose inhibitory neurons
    neuronIdx = range(len(microcircuit.microcircuit))
    numpy.random.shuffle(neuronIdx)
    for i in range(len(input)):
        inputConnections.W[i, neuronIdx[i]] = 2.0

    return (input, inputConnections, seqSpiketimes, labels)


def createDefaultPatternInput(duration, microcircuit, tuner, refPatterns=None):
    from learning import InputConnection

    # Generate patterns if not provided
    if refPatterns == None:
        nbInputs = 3
        nbPatterns = 4
        widthEpoch = 20 * ms
        orders, times = generateRankOrderCodedPatterns(nbInputs, nbPatterns, widthEpoch, delayEpoch=1 * ms, refractory=2.0 * ms)

    else:
        nbInputs = refPatterns.orders.shape[1]
        nbPatterns = refPatterns.orders.shape[0]
        widthEpoch = refPatterns.width
        orders = refPatterns.orders
        times = refPatterns.times

    # Generate random sequence of patterns
    spiketimes = []
    seq = []
    orderData = []
    nbEpochs = int(duration / widthEpoch)
    for n in range(nbEpochs):
        p = numpy.random.randint(0, nbPatterns)
        seq.append(p)
        orderData.append(orders[p, :])
        for i in range(nbInputs):
            t = times[p, i] + n * widthEpoch
            spiketimes.append((i, t))

    input = SpikeGeneratorGroup(nbInputs, spiketimes)
    inputConnections = InputConnection(input, microcircuit.microcircuit, state='v', microcircuit=microcircuit,
                                       tuner=tuner)

    neuronIdx = range(len(microcircuit.microcircuit))
    numpy.random.shuffle(neuronIdx)
    for i in range(nbInputs):
        inputConnections.W[i, neuronIdx[i]] = 2.0

        # Number of average input synaptic connections is fixed to 10% of reservoir links      fracInput = 0.01
    #    nbInputConn = int(fracInput * microcircuit.connections.W.nnz)
    #    sparseness = float(nbInputConn) / float(nbInputs * len(microcircuit.microcircuit))
    #    inputWeightFunc = lambda i, j: numpy.random.uniform(0.5, 2.0)
    #    inputConnections.connect_random(input, microcircuit.microcircuit, sparseness=sparseness, weight=inputWeightFunc)

    return (input, inputConnections, seq, orderData, orders, times, widthEpoch)

def createDefaultRankOrderCodedInput(duration, microcircuit, tuner):
    from learning import InputConnection

    nbInputs = 6
    widthEpoch = 10 * ms
    nbEpochs = int(duration / widthEpoch)
    orders, spiketimes = generateRankOrderCodedData(nbInputs, nbEpochs, widthEpoch, delayEpoch=0.5 * ms,
                                                    refractory=5 * ms)
    input = SpikeGeneratorGroup(len(microcircuit.microcircuit), spiketimes)
    inputConnections = InputConnection(input, microcircuit.microcircuit, state='v', microcircuit=microcircuit,
                                       tuner=tuner)
    inputWeightFunc = lambda i, j: numpy.random.uniform(0.5, 2.0)

    # Number of average input synaptic connections is fixed to 10% of reservoir links
    fracInput = 0.1
    nbInputConn = int(fracInput * microcircuit.connections.W.nnz)
    sparseness = float(nbInputConn) / float(nbInputs * len(microcircuit.microcircuit))
    inputConnections.connect_random(input, microcircuit.microcircuit, sparseness=sparseness, weight=inputWeightFunc)
    return (input, inputConnections, orders)


# function created by Francois Favreau
def sequenceInputSynthetic(duration, microcircuit, tuner):
    from learning import InputConnection

    nbInputs = 3    # number of input neurons
    widthEpoch = 1 * ms    # samples will be provided to the network each widthEpoch

    # import spike sequences from matlab
    f = scipy.io.loadmat('dataset_ampl_freq.mat')
    finalSequence = f["final_output_spike"]

    #calculating when the neuron will fire
    spiketimes = []
    nbEpochs = int(duration / len(finalSequence))

    inputMode = 0   # choice of how to send inputs to the network
    if (inputMode == 0):  # each input neuron receives the same signal
        for n in range (len(finalSequence[0,:])):       # len(finalSequence[0,:]) = 1000
            for i in range (nbInputs):
                if finalSequence[0,n]==1:   # if we get a "1", the neuron fire
                    t = n * widthEpoch
                    spiketimes.append((i,t))
    else:   # delayed inputs
        for n in range (len(finalSequence[0,:])):       # len(finalSequence[0,:]) = 1000
            for i in range(nbInputs):
                if finalSequence[0,n]==1:   # if we get a "1", the neuron fire
                    t = (n+i) * widthEpoch   #TODO : change value t depending on the desired delay
                    spiketimes.append((i,t))


    #Create input matrix and input connection matrix
    input = SpikeGeneratorGroup(nbInputs, spiketimes)
    inputConnections = InputConnection(input, microcircuit.microcircuit, state='v', microcircuit=microcircuit, tuner=tuner)

    #Shuffle
    neuronIdx = range(len(microcircuit.microcircuit))
    numpy.random.shuffle(neuronIdx)
    for i in range(nbInputs):
        inputConnections.W[i, neuronIdx[i]] = 2.0

    return (input, inputConnections, widthEpoch)


def getDefaultExtractParams():
    extractParams = {'cfbmin': 100 * Hz, 'cfbmax': 4 * kHz,
                     'mfbmin': 5 * Hz, 'mfbmax': 15 * Hz,
                     'nbModulationFilters': 4}
    inputModelParams = {'threshold': 0.55, 'sigma': 0.02}
    return (extractParams, inputModelParams)


def getDefaultTrainingParams():
    method = 'LTSOCP'
    trainParams = {'alpha': 1.0, 'learningRate': 0.5,
                   'wmin': -1.0, 'wmax': 1.0, 'ponderation': 'f'}
    return (method, trainParams)


def getDefaultMicrocircuitModelParams():
    neuronModel = 'lif_adapt'
    modelParams = {'excitatoryProb': 0.8, 'refractory': 5 * ms, 'vti': 0.10}
    return (neuronModel, modelParams)


def getDefaultConnectivityParams(structure):
    if structure == 'random':
        connectivity = 'random-uniform'
        connectParams = {'m': 17, 'wmin': 0.25, 'wmax': 0.5, 'delay': 0.0 * ms,
                         'macrocolumnShape': [1, 1, 1], 'minicolumnShape': [8, 8, 8]}
    elif structure == 'small-world':
        connectivity = 'small-world'
        connectParams = {'m': 16, 'wmin': 0.25, 'wmax': 0.5, 'delay': 0.0 * ms,
                         'macrocolumnShape': [2, 2, 2], 'minicolumnShape': [4, 4, 4],
                         'intercolumnarSynapsesRatio': 0.1, 'intercolumnarStrengthFactor': 0.85,
                         'intracolumnarSparseness': 8.0, 'intercolumnarSparseness': 8.0, 'delay': 0 * ms}
    N = numpy.prod(connectParams['minicolumnShape']) * numpy.prod(connectParams['macrocolumnShape'])
    return (N, connectivity, connectParams)


def getDefaultSimulationTimeStep():
    return 1.0 * ms
