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
import numpy
import pylab
import scipy
import bisect

class Patterns(object):
    def __init__(self, nbNeurons, nbActiveNeurons, nbPatterns, spiketimes, width):
        self.nbNeurons = nbNeurons
        self.nbActiveNeurons = nbActiveNeurons
        self.nbPatterns = nbPatterns
        self.spiketimes = spiketimes
        self.width = width

class RocPatterns(Patterns):
    def __init__(self, orders, times, width):
        self.orders = orders
        Patterns.__init__(times, width)

def plotPatterns(patterns, unit=ms):

    fig = pylab.figure(facecolor='white')
    line, = pylab.plot([], [], '.', color='gray')
    ax = fig.add_subplot(1, 1, 1)

    min_y = -0.5
    ax.set_ylim((min_y, patterns.nbNeurons))
    pylab.ylabel('Neuron number')
    if unit == ms:
        pylab.xlabel('Time [ms]')
    elif unit == second:
        pylab.xlabel('Time [sec]')
    else:
        raise Exception('Unsupported unit provided')
    pylab.title('Rank-order coded patterns')

    # Draw spikes
    spikes = []
    for n in range(patterns.nbPatterns):
        for i, t in patterns.spiketimes[n]:
            spikes.append((i, t + n * patterns.width))

    allst = []
    if len(spikes):
        sn, st = numpy.array(spikes).T
    else:
        sn, st = numpy.array([]), numpy.array([])
    st /= unit
    allsn = [sn]
    allst.append(st)
    sn = hstack(allsn)
    st = hstack(allst)

    line.set_xdata(numpy.array(st))
    ax.set_xlim((0.0, numpy.max(st)))
    line.set_ydata(sn)

    # Draw lines between each pattern
    for n in range(patterns.nbPatterns):
        t = n * (patterns.width / unit)
        l = Line2D([t, t], ax.get_ylim(), color='grey', linestyle='--', linewidth=1.0)
        ax.add_line(l)

    fig.canvas.draw()

def generateRandomPatterns(nbNeurons, nbActiveNeurons, nbPatterns, widthEpoch=10 * ms, delayEpoch=1 * ms):

    allSpiketimes = []

    # Loop for each class to generate
    minT = delayEpoch
    maxT = widthEpoch - delayEpoch
    times = numpy.arange(minT, maxT, defaultclock.dt)
    for n in range(nbPatterns):
        spiketimes = []

        # Choose random neurons to fire
        neuronIdx = range(nbNeurons)
        numpy.random.shuffle(neuronIdx)
        neuronIdx = neuronIdx[0:nbActiveNeurons]

        # Choose random time to fire
        timesIdx = numpy.random.randint(0, len(times), nbActiveNeurons)
        for i in range(nbActiveNeurons):
            j = neuronIdx[i]
            t = times[timesIdx[i]]
            spiketimes.append((j, t))

        allSpiketimes.append(spiketimes)

    return allSpiketimes

def generateRankOrderCodedPatterns(nbNeurons, nbPatterns, widthEpoch=10 * ms, delayEpoch=1 * ms, refractory=0.0 * ms):

    spiketimes = numpy.zeros((nbPatterns, nbNeurons))
    orders = numpy.zeros((nbPatterns, nbNeurons))
    nbAvailableTimeSteps = int(widthEpoch / defaultclock.dt)
    if nbAvailableTimeSteps < nbNeurons:
        raise Exception('Temporal resolution is too low for the given number of neurons!')

    # Loop for each class to generate
    minT = delayEpoch
    maxT = widthEpoch - delayEpoch
    times = numpy.linspace(minT, maxT, nbNeurons)
    for n in range(nbPatterns):

        conflictFound = True
        nbRetry = 0
        maxRetry = 100000
        while conflictFound and nbRetry < maxRetry:

            genOrders = range(nbNeurons)
            numpy.random.shuffle(genOrders)

            # Ensure that the pattern doesn't already exist
            conflictFound = False
            for m in range(n):
                if (genOrders == orders[m, :]).all():
                    conflictFound = True;
                    nbRetry += 1
                    break;

            if not conflictFound and refractory > 0.0:
                # Ensure each neuron is not in refractory period if concatenated with every other class
                for target in range(nbNeurons):
                    for m in range(n):
                        if times[genOrders[target]] + widthEpoch - spiketimes[m, target] < refractory:
                            conflictFound = True;
                            nbRetry += 1
                            break;
                    if conflictFound:
                        break;

        if conflictFound:
            raise Exception('Unable to generate all patterns: %d generated' % (n))

        spiketimes[n, :] = times[genOrders]
        orders[n, :] = genOrders

    return orders, spiketimes

def generateRankOrderCodedData(nbNeurons, nbEpochs, widthEpoch=10 * ms, delayEpoch=0.5 * ms, refractory=5.0 * ms):

    spiketimes = []
    orders = numpy.zeros((nbEpochs, nbNeurons))
    nextAllowedSpikeTimes = numpy.zeros(nbNeurons)
    nbAvailableTimeSteps = int(widthEpoch / defaultclock.dt)
    if nbAvailableTimeSteps < nbNeurons:
        raise Exception('Temporal resolution is too low for the given number of neurons!')

    # Loop for each epoch
    for n in range(nbEpochs):
        minT = n * widthEpoch + delayEpoch
        maxT = (n + 1) * widthEpoch - delayEpoch

        # Loop for each time bin
        neuronOpenList = range(nbNeurons)
        currentOrder = 0
        for t in numpy.linspace(minT, maxT, nbNeurons):

            # Choose a neuron at random
            retry = 0
            availableTargetIdx = None
            while availableTargetIdx == None and retry < 10 * len(neuronOpenList):
                targetIdx = numpy.random.randint(0, len(neuronOpenList))
                target = neuronOpenList[targetIdx]

                # Ensure the neuron is not in refractory period
                if t >= nextAllowedSpikeTimes[target]:
                    availableTargetIdx = targetIdx
                else:
                    retry += 1

            if availableTargetIdx == None:
                raise Exception('Refractory constraint not adapted to epoch width!')

            target = neuronOpenList.pop(availableTargetIdx)
            nextAllowedSpikeTimes[target] = t + float(refractory)
            spiketimes.append((target, t))
            orders[n, target] = currentOrder
            currentOrder += 1

    return orders, spiketimes
