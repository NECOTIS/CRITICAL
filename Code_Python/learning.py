'''
Created on 21.02.2012

@author: "Simon Brodeur"
'''

import brian_no_units
from brian import *
import sys
import scipy
import numpy
import pylab
import time
import threading
import bisect
from itertools import izip
import itertools
import matplotlib
from matplotlib import cm

from monitors import *

def clip(x, a, b):
    min = numpy.min((a, b))
    max = numpy.max((a, b))
    return numpy.clip(x, min, max)

class InputConnection(Connection):
    def __init__(self, source, target, state=0, delay=0 * msecond, modulation=None,
                 structure='sparse', weight=None, sparseness=None, max_delay=0 * ms, microcircuit=None, tuner=None, **kwds):
        self.microcircuit = microcircuit
        self.tuner = tuner
        Connection.__init__(self, source, target, state, delay, modulation,
                 structure, weight, sparseness, max_delay, **kwds)

    def propagate(self, spikes):
        if len(spikes) > 0 and self.tuner != None:
            currentSpikeTime = self.source.clock.t
            for i in spikes:
                # Calculate statistics about local input contributions to postsynaptic neurons
                row = self.W[i, :]
                for j, w in izip(row.ind, row):
                    # Only consider the contribution if not in the refractory period
                    if self.tuner.lastSpikeTimes[j] == 0.0 or (currentSpikeTime - self.tuner.lastSpikeTimes[j] + defaultclock.dt) > self.tuner.refractory:
                        self.tuner.localInContribs[j] += w

        Connection.propagate(self, spikes)

    def do_propagate(self):
        self.propagate(self.source.get_spikes(self.delay))

class LTSOCP_Monitor(NetworkOperation):
    '''
    Realtime plotting of the mean and variance of neuron contribution
    '''

    def __init__(self, tuner, refresh=100 * ms, preallocate=10000, clock=None):
        self.tuner = tuner
        self.refresh = refresh

        self.fig = pylab.figure(facecolor='white')
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.fig.subplots_adjust(hspace=0.5)

        self.data = numpy.zeros((preallocate, 4))
        self.dataCounter = 0
        self.refreshCounter = 0.0

        NetworkOperation.__init__(self, lambda:None, clock=clock)

    def __call__(self):

        # Calculate average contributions for each neurons
        sumContrib = numpy.sum(self.tuner.neuronLogData, axis=1)
        avgContrib = numpy.zeros(self.tuner.neuronLogData.shape[0])
        globalAvgContrib = 0.0
        nbNonNulContrib = 0.0
        for i in range(self.tuner.neuronLogData.shape[0]):
            if self.tuner.neuronLogDataLen[i] > 0:
                avgContrib[i] = sumContrib[i] / float(self.tuner.neuronLogDataLen[i])
                globalAvgContrib += avgContrib[i]
                nbNonNulContrib += 1

        if nbNonNulContrib > 0:
            globalAvgContrib /= float(nbNonNulContrib)

        # Calculate the average variance across all neurons
        globalVarContrib = 0.0
        nbNonNulContrib = 0.0
        varDataFlat = []
        for i in range(self.tuner.neuronLogData.shape[0]):
            if self.tuner.neuronLogDataLen[i] > 0:
                curVarData = self.tuner.neuronLogData[i, 0:self.tuner.neuronLogDataLen[i]]
                varContrib = numpy.var(curVarData)
                varDataFlat.append(curVarData)
                globalVarContrib += varContrib
                nbNonNulContrib += 1
        if len(varDataFlat) == 0:
            localVarContrib = 0
        else:
            localVarContrib = numpy.var(numpy.concatenate(varDataFlat))
        if nbNonNulContrib > 0:
            globalVarContrib /= float(nbNonNulContrib)

        if self.dataCounter >= self.data.shape[0]:
            # Warning: dynamically resizing array
            self.data.resize((2 * self.data.shape[0], self.data.shape[1]), refcheck=False)
        self.data[self.dataCounter, :] = [globalAvgContrib, globalVarContrib, localVarContrib, self.clock.t]
        self.dataCounter += 1

        self.refreshCounter += self.clock.dt
        if self.refreshCounter >= self.refresh and self.tuner.globalLogDataLen > 0:
            self.ax1.cla()
            t = self.tuner.logTimes[0:self.tuner.globalLogDataLen]
            tt = self.data[0:self.dataCounter, 3]

            self.ax1.plot(t / ms, self.tuner.globalLogData[0:self.tuner.globalLogDataLen, 0], color='blue')
            self.ax1.plot(tt / ms, self.data[0:self.dataCounter, 0], color='red', linestyle='dashed')
            self.ax1.set_title('Global average neural contribution')
            self.ax1.set_xlabel('Time [ms]')
            self.ax1.set_ylabel('Amplitude')
            self.ax1.grid(True)

            self.ax2.cla()
            self.ax2.plot(tt / ms, self.data[0:self.dataCounter, 2], color='blue')
            self.ax2.plot(tt / ms, self.data[0:self.dataCounter, 1], color='red', linestyle='dashed')
            self.ax2.set_title('Global variance of neural contribution')
            self.ax2.set_xlabel('Time [ms]')
            self.ax2.set_ylabel('Amplitude')
            self.ax2.grid(True)

            self.fig.canvas.draw()
            self.refreshCounter = 0.0

class LTSOCP(Connection, Monitor):

    def __init__(self, microcircuit, trainParams, maxNeuronLogEvents=8, preallocate=10000):
        self.microcircuit = microcircuit

        # Training parameters
        self.learningRate = trainParams['learningRate']
        self.alpha = trainParams['alpha']
        self.wmin = trainParams['wmin']
        self.wmax = trainParams['wmax']
        self.ponderation = trainParams['ponderation']
        self.refractory = trainParams['refractory']

        # Connection class attributes
        self.source = microcircuit.microcircuit
        self.target = None
        self.W = None
        self.delay = 0.0

        self.maxNeuronLogEvents = maxNeuronLogEvents
        self.neuronLogData = numpy.zeros((len(self.microcircuit.microcircuit), self.maxNeuronLogEvents))
        self.nextNeuronLogEntries = numpy.zeros(len(self.microcircuit.microcircuit))
        self.neuronLogDataLen = numpy.zeros(len(self.microcircuit.microcircuit))

        self.logTimes = numpy.zeros((preallocate, 1))
        self.globalLogData = numpy.zeros((preallocate, 1))
        self.globalLogDataLen = 0

        self.reinit()

    def reinit(self):
        self.localCbfEsts = numpy.zeros(len(self.source))
        self.localInContribs = numpy.zeros(len(self.source))
        self.lastSpikeTimes = numpy.zeros(len(self.source))
        self.iterations = 0

        #Reset log data
        self.neuronLogData[:, :] = 0
        self.nextNeuronLogEntries[:] = 0
        self.globalLogData[:] = 0
        self.globalLogDataLen = 0

    def propagate(self, spikes):
        if len(spikes) > 0:

            alpha_g = 0.0
            avgDeltaW = 0.0
            nbDeltaW = 0
            nbDeltaW_pos = 0
            nbDeltaW_neg = 0
            nbDeltaW_nul = 0

            currentSpikeTime = self.source.clock.t
            nbExcitatorySpikes = 0
            for i in spikes:
                # Skip inhibitory neurons
                if not numpy.sign(self.microcircuit.neuronTypes[i]) > 0.0:
                    continue

                # Calculate statistics about local input contributions to postsynaptic neurons
                row = self.microcircuit.connections.W[i, :]
                for j, w in izip(row.ind, row):
                    # Skip postsynaptic inhibitory neurons connections
                    if not numpy.sign(self.microcircuit.neuronTypes[j]) > 0.0:
                        continue

                    # TODO: should consider the timing - decaying!
                    inc = self.microcircuit.connections.W[i, j]

                    # Only consider the contribution if not in the refractory period
                    if self.lastSpikeTimes[j] == 0.0 or (currentSpikeTime - self.lastSpikeTimes[j] + defaultclock.dt) > self.refractory:
                        self.localInContribs[j] += inc

                # Update local estimation of CBF of the presynaptic neurons
                col = self.microcircuit.connections.W[:, i]
                for j, w in izip(col.ind, col):
                    # Skip presynaptic inhibitory neurons connections
                    if not numpy.sign(self.microcircuit.neuronTypes[j]) > 0.0:
                        continue

                    # TODO: should consider the threshold value

                    # Increment depends on the time since the last presynaptic spike from the neuron
                    if self.lastSpikeTimes[j] > 0.0:
                        tau = self.microcircuit.microcircuit.tau[i]
                        inc = self.microcircuit.connections.W[j, i] * numpy.exp((self.lastSpikeTimes[j] - currentSpikeTime) / tau)
                        inc /= self.localInContribs[i]

                        # FIXME: why is this still happening?
                        if self.localInContribs[i] == 0.0:
                            print >> sys.stderr, 'Warning, neuron %d seems to have spontaneously fired! Accumulated input contributions is: %f' % (i, self.localInContribs[i])
                            break

                        self.localCbfEsts[j] += inc

                # Calculation of the error on the target CBF
                e_cbf = self.alpha - self.localCbfEsts[i]

                # Update postsynaptic weights based on the error
                row = self.microcircuit.connections.W[i, :]
                nbPostSynapses = len(row)
                for j, w in izip(row.ind, row):
                    # Modulated weight changes
                    if self.ponderation == 'u':
                        # Uniform
                        s = 1.0
                    elif self.ponderation == 'l':
                        # Affect the less recently active postsynaptic neurons
                        tau = self.microcircuit.microcircuit.tau[i]
                        s = 1.0 - numpy.exp((self.lastSpikeTimes[j] - currentSpikeTime) / tau)
                    elif self.ponderation == 'f':
                        # Affect the most recently active postsynaptic neurons
                        tau = self.microcircuit.microcircuit.tau[i]
                        s = numpy.exp((self.lastSpikeTimes[j] - currentSpikeTime) / tau)
                    else:
                        raise Exception('Unknown ponderation type: %s' % (self.ponderation))

                    deltaW = self.learningRate * (e_cbf / nbPostSynapses) * s

                    w = clip(w + deltaW, 1e-04, float(self.wmax))
                    self.microcircuit.connections.W[i, j] = w

                    avgDeltaW += deltaW
                    nbDeltaW += 1
                    if deltaW > 0.0:
                        nbDeltaW_pos += 1
                    elif deltaW < 0.0:
                        nbDeltaW_neg += 1
                    else:
                        nbDeltaW_nul += 1

                alpha_g += self.localCbfEsts[i]
                nbExcitatorySpikes += 1

                # Neuron Data logging
                self.neuronLogData[i, self.nextNeuronLogEntries[i]] = self.localCbfEsts[i]
                self.nextNeuronLogEntries[i] += 1
                if self.nextNeuronLogEntries[i] >= self.neuronLogData.shape[1]:
                    self.nextNeuronLogEntries[i] = 0
                if self.neuronLogDataLen[i] < self.neuronLogData.shape[1]:
                    self.neuronLogDataLen[i] += 1

                # Reset local estimation of CBF
                self.localCbfEsts[i] = 0.0
                self.lastSpikeTimes[i] = currentSpikeTime
                self.localInContribs[i] = 0.0

            if nbExcitatorySpikes > 0:
                alpha_g = float(alpha_g) / float(nbExcitatorySpikes)

            if self.globalLogDataLen >= self.globalLogData.shape[0]:
                # Warning: dynamically resizing array
                self.globalLogData.resize((2 * self.globalLogData.shape[0], self.globalLogData.shape[1]), refcheck=False)
                self.logTimes.resize((2 * self.logTimes.shape[0], self.logTimes.shape[1]), refcheck=False)
            self.globalLogData[self.globalLogDataLen, :] = [alpha_g]
            self.logTimes[self.globalLogDataLen, :] = [currentSpikeTime]
            self.globalLogDataLen += 1

            if nbDeltaW > 0:
                avgDeltaW /= nbDeltaW
            else:
                avgDeltaW = 0.0

            #print "[LTSOCP] Iteration no.%d, alpha_g = %f (avg deltaW = %e [pos=%d, nul=%d, neg=%d] )" % (self.iterations, alpha_g, avgDeltaW, nbDeltaW_pos, nbDeltaW_nul, nbDeltaW_neg)
            self.iterations += 1

    def origin(self, P, Q):
        '''
        Returns the starting coordinate of the given groups in
        the connection matrix W.
        '''
        return (P.origin - self.microcircuit.microcircuit.origin, 0)

    def compress(self):
        pass
