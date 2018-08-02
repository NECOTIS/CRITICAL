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

import brian_no_units
from brian import *
import scipy
import numpy
import pylab
import time
import pickle
import threading
import time
import bisect
import cPickle as pickle

import defaultparams
from learning import *
from microcircuit import *
from monitors import *
from testing import *
from rankorder import *

from pyfann import libfann

class DelayedRateMonitor(SpikeMonitor):

    def __init__(self, source, epochWidth=50 * ms, delay=0 * ms):
        SpikeMonitor.__init__(self, source, record=False, delay=delay)
        if bin:
            self._bin = int(epochWidth / source.clock.dt)
        else:
            self._bin = 1 # bin size in number
        self._acc = numpy.zeros(len(self.source))
        self.epochWidth = epochWidth
        self.delay = delay
        self.rate = numpy.zeros(len(self.source))
        self._curstep = 0
        self._clock = source.clock

    def reinit(self):
        SpikeMonitor.reinit(self)
        self._acc = numpy.zeros(len(self.source))
        self.rate = numpy.zeros(len(self.source))
        self._curstep = 0

    def propagate(self, spikes):
        if self._curstep == 0:
            self.rate[:] = self._acc / float(self._bin * self._clock.dt)
            self._acc = numpy.zeros(len(self.source))
            self._curstep = self._bin

        for i in spikes:
            self._acc[i] += 1.0
        self._curstep -= 1

class MfrReservoirStateReadout(NetworkOperation):

    def __init__(self, N, microcircuit, epochWidth, delay=0 * ms, preallocate=10000):

        self.N = N
        self.stateData = numpy.zeros((preallocate, N))
        self.epochCounter = 0
        self.stepCounter = 0
        self.delay = delay
        self.epochWidth = epochWidth
        self.epochSteps = int(epochWidth / defaultclock.dt)

        # Choose a subset of the population for the readout
        neuronIdx = range(0, len(microcircuit.microcircuit))
        numpy.random.shuffle(neuronIdx)
        self.readoutNeurons = neuronIdx[0:N]
        self.rateMonitor = DelayedRateMonitor(microcircuit.microcircuit, epochWidth, delay)
        NetworkOperation.__init__(self, lambda:None, clock=EventClock(dt=defaultclock.dt))
        self.contained_objects += [self.rateMonitor]

    def __call__(self):

        if self.clock.t - (self.delay + self.epochWidth) < 0.0:
            return

        if self.stepCounter == 0:
            # Get the list of spikes for the current epoch
            tmin = self.clock.t - self.epochWidth
            tmax = self.clock.t

            state = self.rateMonitor.rate[self.readoutNeurons]

            self.epochCounter += 1
            self.stepCounter = self.epochSteps

            print 'Reservoir state at t = %4.1fms, epoch = [%4.1fms, %4.1fms]: orders = ' % (self.clock.t / ms, tmin / ms, tmax / ms), state

        self.stepCounter -= 1

    def getStateData(self):
        return list(self.stateData[0:self.epochCounter, :])

class RocReservoirStateReadout(NetworkOperation):

    def __init__(self, N, microcircuit, epochWidth, delay=0 * ms, preallocate=10000):

        self.N = N
        self.orderData = numpy.zeros((preallocate, N))
        self.epochCounter = 0
        self.stepCounter = 0
        self.delay = delay
        self.epochWidth = epochWidth
        self.epochSteps = int(epochWidth / defaultclock.dt)

        # FIXME: don't choose inhibitory neurons

        # Choose a subset of the population for the readout
        neuronIdx = range(0, len(microcircuit.microcircuit))
        numpy.random.shuffle(neuronIdx)
        self.readoutNeurons = neuronIdx[0:N]
        self.spikeMonitor = LastSpikeMonitor(microcircuit.microcircuit)
        NetworkOperation.__init__(self, lambda:None, clock=EventClock(dt=defaultclock.dt))
        self.contained_objects += [self.spikeMonitor]

    def __call__(self):

        if self.clock.t - (self.delay + self.epochWidth) < 0.0:
            return

        if self.stepCounter == 0:
            # Get the list of spikes for the current epoch
            tmin = self.clock.t - self.epochWidth
            tmax = self.clock.t
            orderedSpikeTimes = numpy.zeros((len(self.readoutNeurons), 2))
            for i in range(len(self.readoutNeurons)):
                readoutNeuron = self.readoutNeurons[i]
                orderedSpikeTimes[i, 1] = readoutNeuron
                lastSpiketime = self.spikeMonitor.lastSpikeTimes[readoutNeuron]
                # Verify is the last spike time exists and is within the current epoch
                if lastSpiketime == 0.0 or lastSpiketime < tmin:
                    # Large number only to make them at the end of the list
                    lastSpiketime = float('inf')

                orderedSpikeTimes[i, 0] = lastSpiketime

            # Calculate rank order for each spike
            orders = numpy.lexsort((orderedSpikeTimes[:, 1], orderedSpikeTimes[:, 0]))

            # Generate order vector with neurons at fixed indices
            orderFixed = numpy.zeros(len(self.readoutNeurons))
            for i in range(orders.shape[0]):
                order = float(i) / float(orders.shape[0] - 1)
                if orderedSpikeTimes[orders[i], 0] == float('inf'):
                    order = -1
                orderFixed[orders[i]] = order

            # DFT transform for translation invariance
#            for i in range(orders.shape[0]):
#                if orderedSpikeTimes[orders[i], 0] == float('inf'):
#                    orders[i] = 0
#                else:
#                    orders[i] += 1
#
#            Y = numpy.fft.fft(orders) / len(orders)
#            Y = Y[1:(len(orders) / 2)]
#            ordersMag = abs(Y) ** 2
#
#            orderFixed = numpy.zeros(len(orders))
#            orderFixed[0:len(ordersMag)] = ordersMag

            if self.epochCounter >= self.orderData.shape[0]:
                # Warning: dynamically resizing array
                self.orderData.resize((2 * self.orderData.shape[0], self.orderData.shape[1]), refcheck=False)

            self.orderData[self.epochCounter, :] = orderFixed
            self.epochCounter += 1
            self.stepCounter = self.epochSteps

            #print 'Reservoir state at t = %4.1fms, epoch = [%4.1fms, %4.1fms]: orders = ' % (self.clock.t / ms, tmin / ms, tmax / ms), orders

        self.stepCounter -= 1

    def getStateData(self):
        return list(self.orderData[0:self.epochCounter, :])

def classifyMultilayerANN(stateData, targets):

    numInput = len(stateData[0])
    numHidden = 5 * numInput
    numOutput = numpy.max(targets) + 1

    desiredError = 0.001
    maxIterations = 1000
    iterationsBetweenReports = 100

    ann = libfann.neural_net()
    ann.create_standard_array((numInput, numHidden, numOutput))
    ann.set_learning_rate(0.1)
    ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)
    ann.set_training_algorithm(libfann.TRAIN_QUICKPROP)
    ann.set_train_error_function(libfann.ERRORFUNC_LINEAR)

    #Create the train and test datasets
    inputs = []
    outputs = []
    for i in range(len(stateData)):
        inputs.append(stateData[i])
        outBin = [0] * numOutput
        outBin[targets[i]] = 1.0
        outputs.append(outBin)

    allData = libfann.training_data()
    allData.set_train_data(inputs, outputs)
    allData.scale_input_train_data(-1.0, 1.0)
    allData.scale_output_train_data(-1.0, 1.0)
    allData.shuffle_train_data()

    #Split to train and test
    trainRatio = 0.75
    nbData = allData.length_train_data()
    nbTrainData = int(trainRatio * nbData)

    trainData = libfann.training_data(allData)
    trainData.subset_train_data(0, nbTrainData)
    testData = libfann.training_data(allData)
    testData.subset_train_data(nbTrainData, nbData - nbTrainData)

    #Training the network
    ann.train_on_data(trainData, maxIterations, iterationsBetweenReports, desiredError)

    #Accuracy
    testInputs = testData.get_input()
    refOutputs = testData.get_output()
    nbErrors = 0
    nbTotal = len(testInputs)
    for i in range(len(testInputs)):
        output = ann.run(testInputs[i])
        maxOutputIdx = numpy.array(output).argmax()
        maxRefOutputIdx = numpy.array(refOutputs[i]).argmax()
        if maxOutputIdx != maxRefOutputIdx:
            nbErrors += 1
    accuracy = float(nbTotal - nbErrors) / float(nbTotal)

    return accuracy


def classifyKNearestNeighbor(stateData, targets):

    #Create the train and test datasets
    minLen = numpy.min((len(stateData), len(targets)))
    data = numpy.array(stateData[0:minLen])
    labels = numpy.array(targets[0:minLen])

    # Generate train and test data (with shuffled data)
    trainRatio = 0.75
    nbTrainData = int(trainRatio * len(data))
    dataIdx = range(0, len(data))
    numpy.random.shuffle(dataIdx)
    trainData = list(data[dataIdx[0:nbTrainData]])
    trainLabels = list(labels[dataIdx[0:nbTrainData]])
    testData = list(data[dataIdx[nbTrainData:]])
    testLabels = list(labels[dataIdx[nbTrainData:]])

    # Create the classifier
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(trainData, trainLabels)

    # Validate classification on the test data
    outputLabels = neigh.predict(testData)
    nbErrors = 0
    nbTotal = len(outputLabels)
    for i in range(len(outputLabels)):
        if outputLabels[i] != testLabels[i]:
            nbErrors += 1
    accuracy = float(nbTotal - nbErrors) / float(nbTotal)

    return accuracy

def testReservoirReadout(structure):

    extractParams, inputModelParams = defaultparams.getDefaultExtractParams()

    N, connectivity, connectParams = defaultparams.getDefaultConnectivityParams(structure)
    neuronModel, modelParams = defaultparams.getDefaultMicrocircuitModelParams()
    microcircuit = Microcircuit(N, neuronModel, modelParams, connectivity, connectParams)
    network = Network(microcircuit.microcircuit, microcircuit.connections)

    # Input
    duration = 1000 * ms
    input, inputConnections, seq, orderData, patterns = defaultparams.createDefaultPatternInput(duration, microcircuit, None)
    network.add(input)
    network.add(inputConnections)

    # Readout
    N = 8
    delay = 10 * ms
    steps = int(duration / patterns.width)
    readout = MfrReservoirStateReadout(N, microcircuit, patterns.width, delay, preallocate=steps)
    network.add(readout)

    # Monitor
    spikeMonitor = RealtimeSpikeMonitor(microcircuit.microcircuit, refresh=25 * ms, showlast=200 * ms, unit=ms)
    network.add(spikeMonitor)

    spikeMonitor = RealtimeSpikeMonitor(input, refresh=25 * ms, showlast=200 * ms, unit=ms)
    network.add(spikeMonitor)

    # Simulation
    network.reinit(states=False)
    network.run(duration)

def testReservoirComputing(structure, cbf, classification, task='class'):

    nbTrials = 1

    allAccuracies = None
    allDelays = None
    for n in range(nbTrials):

        N, connectivity, connectParams = defaultparams.getDefaultConnectivityParams(structure)
        neuronModel, modelParams = defaultparams.getDefaultMicrocircuitModelParams()
        microcircuit = Microcircuit(N, neuronModel, modelParams, connectivity, connectParams)
        network = Network(microcircuit.microcircuit, microcircuit.connections)

        defaultclock.dt = defaultparams.getDefaultSimulationTimeStep()

        # Training
        print "Training microcircuit for cbf %f..." % (cbf)
        method, trainParams = defaultparams.getDefaultTrainingParams()
        trainParams['alpha'] = cbf
        trainParams['refractory'] = modelParams['refractory']
        tuner = LTSOCP(microcircuit, trainParams)
        network.add(tuner)

        # Input
        duration = 1000 * ms
        input, inputConnections, seq, seqSpiketimes, patterns = defaultparams.createDefaultRandomPatternInput(duration, microcircuit, tuner)
        network.add(input)
        network.add(inputConnections)

        plotPatterns(patterns)

        # Monitor
        steps = int(duration / defaultclock.dt)
        logMonitor = LTSOCP_Monitor(tuner, refresh=25 * ms, preallocate=steps, clock=EventClock(dt=25 * ms))
        network.add(logMonitor)

        resSpikeMonitor = RealtimeSpikeMonitor(microcircuit.microcircuit, refresh=25 * ms, showlast=200 * ms, unit=ms)
        network.add(resSpikeMonitor)

        inputSpikeMonitor = RealtimeSpikeMonitor(input, refresh=25 * ms, showlast=200 * ms, unit=ms)
        network.add(inputSpikeMonitor)

        # Simulation
        network.reinit(states=False)
        network.run(duration, report='text')
        pylab.close(logMonitor.fig)
        pylab.close(resSpikeMonitor.fig)
        pylab.close(inputSpikeMonitor.fig)

        print "Generating data for readout training..."

        nbTrainEpochs = 5000
        duration = nbTrainEpochs * patterns.width
        inputTrain, inputConnectionsTrain, seq, seqSpiketimes, patterns = defaultparams.createDefaultRandomPatternInput(duration, microcircuit, None, patterns)
        for i in range(inputConnections.W.shape[0]):
            row = inputConnections.W[i, :]
            for j, w in izip(row.ind, row):
                inputConnectionsTrain.W[i, j] = w
        network = Network(microcircuit.microcircuit, microcircuit.connections, inputTrain, inputConnectionsTrain)

        # Readout
        N = 32
        delays = numpy.concatenate((numpy.arange(0, patterns.width, 2 * ms), numpy.arange(patterns.width, 2 * patterns.width, 5 * ms), numpy.arange(2 * patterns.width, 3 * patterns.width, 5 * ms)))
        readouts = []
        for delay in delays:
            readout = RocReservoirStateReadout(N, microcircuit, patterns.width, delay)
            readouts.append(readout)
            network.add(readout)

        # Simulation
        network.reinit(states=False)
        network.run(duration, report='text')

        if task == 'class':
            # Classification
            target = seq
        elif task == 'xor':
            # XOR computation
            target = [seq[0]]
            for i in range(1, len(seq)):
                xorBoolean = numpy.logical_xor(seq[i - 1], seq[i])
                if xorBoolean:
                    target.append(1)
                else:
                    target.append(0)
        elif task == 'parity':
            # Parity computation
            parityOrder = 3
            target = seq[0:parityOrder - 1]
            for i in range(parityOrder, len(seq)):
                idx = numpy.where(seq[(i - parityOrder):i] == 1)
                if len(idx) == 0:
                    nbTrue = 0
                else:
                    nbTrue = len(idx[0])

                if nbTrue % 2 == 0:
                    target.append(1)
                else:
                    target.append(0)
        else:
            raise Exception('Unknown task type: %s', task)

        # Training
        accuracies = numpy.zeros(len(delays))
        for i in range(len(delays)):
            delay = delays[i]
            if classification == 'ann':
                accuracy = classifyMultilayerANN(readouts[i].getStateData(), target)
            elif classification == 'knn':
                accuracy = classifyKNearestNeighbor(readouts[i].getStateData(), target)
            accuracies[i] = accuracy
            print 'Performances after training for delay = %4.1fms: accuracy = %f' % (delay / ms, accuracy)

        if allAccuracies == None:
            allAccuracies = numpy.zeros((nbTrials, len(delays)))
        allAccuracies[n, :] = accuracies

    # Calculate mean and variance of accuracy
    meanAccuracy = numpy.mean(allAccuracies, 0)
    stdAccuracy = numpy.std(allAccuracies, 0)

    # Plot memory capacity curve
    fig = pylab.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(delays / ms, meanAccuracy, color='black')
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([numpy.min(delays) / ms, numpy.max(delays) / ms])
    ax.set_title('Memory capacity')
    ax.set_xlabel('Readout delay [ms]')
    ax.set_ylabel('Classification accuracy')
    ax.grid(True)
    ax.hold(True)

    # Chance-level line
    chanceLevel = 1.0 / float(patterns.nbPatterns)
    l = Line2D([0, numpy.max(delays) / ms], [chanceLevel, chanceLevel], linewidth=2, color='black', linestyle='--')
    ax.add_line(l)
    ax.text(0.05 * numpy.max(delays) / ms, 1.1 * chanceLevel, 'Chance level', fontsize=12, color='black')

    # Add the error bar
    if nbTrials > 1:
        ax.errorbar(delays / ms, meanAccuracy, stdAccuracy, fmt='o', color='black', ecolor='black')

    fig.canvas.draw()


def testGenerateSpikePatterns():

    defaultclock.dt = defaultparams.getDefaultSimulationTimeStep()

    nbNeurons = 512
    nbActiveNeurons = nbNeurons / 10
    nbPatterns = 4
    width = 20 * ms
    spiketimes = generateRandomPatterns(nbNeurons, nbActiveNeurons, nbPatterns, width, delayEpoch=1 * ms)
    patterns = Patterns(nbNeurons, nbActiveNeurons, nbPatterns, spiketimes, width)

    plotPatterns(patterns)

if __name__ == '__main__':

    numpy.random.seed(0)
    numpy.seterr(all='raise')
    pylab.ion()

    #testGenerateSpikePatterns()
    #testReservoirReadout('small-world')
    testReservoirComputing('small-world', 1.0, 'knn')

    print 'All done.'
    pylab.ioff()
    show()
    pylab.close()
