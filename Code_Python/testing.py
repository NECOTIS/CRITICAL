'''
Created on 21.02.2012

@author: "Simon Brodeur"
'''

import brian_no_units
from brian import *
from brian.hears import *
from brian.hears import filtering
import scipy
import numpy
import pylab
import time
import bisect
from itertools import izip
import itertools
import matplotlib
from matplotlib import cm
import pickle
from brian.library.IF import Izhikevich, AdaptiveReset
from brian.plotting import raster_plot
import scipy.maxentropy.maxentutils

from monitors import *

def generateAvalancheLengthHist(microcircuit, inputConnections, NbAval=1000, maxAvalTime=150 * ms, fracFiring=0.50):

    pylab.ioff()

    avalTimes = numpy.zeros(NbAval)
    spikeCounts = numpy.zeros(NbAval)
    for n in range(NbAval):

        spiketimes = []
        offsets = [0 * ms]
        maxGenT = 0.0
        maxDelay = 10 * ms
        for offset in offsets:
            # Get the list of input neurons
            targets = range(0, inputConnections.W.shape[0])
            nbInputFiring = int(fracFiring * len(targets))
            numpy.random.shuffle(targets)
            for target in targets[0:nbInputFiring]:
                t = numpy.random.randint(offset / ms, offset / ms + maxDelay / ms + 1) * ms
                spiketimes.append((target, t))
                if t > maxGenT:
                    maxGenT = t

        input = SpikeGeneratorGroup(inputConnections.W.shape[0], spiketimes)
        inputConnectionsTest = Connection(input, microcircuit.microcircuit, state='v')
        for i in range(inputConnections.W.shape[0]):
            row = inputConnections.W[i, :]
            for j, w in izip(row.ind, row):
                inputConnectionsTest.W[i, j] = w
        inputConnections = inputConnectionsTest

        network = Network(microcircuit.microcircuit, microcircuit.connections, input, inputConnections)

        avalDetect = AvalancheEndDetector(network, microcircuit.microcircuit, microcircuit.connections, maxGenT)
        network.add(avalDetect)

        microcircuit.microcircuit.v[:] = microcircuit.microcircuit.v0[:]
        microcircuit.microcircuit.vt[:] = microcircuit.microcircuit.vt0[:]

        network.reinit(states=False)
        network.run(maxAvalTime)

        if avalDetect.getElapsedTime() - maxGenT < 0.0:
            raise Exception('Invalid avalanche simulation!')

        avalTimes[n] = avalDetect.getElapsedTime() - maxGenT
        spikeCounts[n] = avalDetect.getSpikeCount() - len(spiketimes)

    # Show histogram of avalanche sizes
    tmin = 0.0
    tmax = maxAvalTime #numpy.max(avalTimes)
    nbHistBins = numpy.round(tmax / defaultclock.dt)
    hist, bins = numpy.histogram(avalTimes, bins=nbHistBins, range=(tmin, tmax), normed=False)
    hist = numpy.asfarray(hist) / NbAval

    return (hist, bins)


def estimatePowerLawScaling(microcircuit, inputConnections, NbAval=1000, maxAvalTime=150 * ms):

    pylab.ioff()

    avalTimes = numpy.zeros(NbAval)
    spikeCounts = numpy.zeros(NbAval)
    for n in range(NbAval):

        spiketimes = []
        maxGenT = 0.0
        target = numpy.random.randint(0, inputConnections.W.shape[0])
        spiketimes.append((target, 0 * ms))

        input = SpikeGeneratorGroup(inputConnections.W.shape[0], spiketimes)
        inputConnectionsTest = Connection(input, microcircuit.microcircuit, state='v')
        for i in range(inputConnections.W.shape[0]):
            row = inputConnections.W[i, :]
            for j, w in izip(row.ind, row):
                inputConnectionsTest.W[i, j] = w
        inputConnections = inputConnectionsTest

        network = Network(microcircuit.microcircuit, microcircuit.connections, input, inputConnections)

        avalDetect = AvalancheEndDetector(network, microcircuit.microcircuit, microcircuit.connections, maxGenT)
        network.add(avalDetect)

        microcircuit.microcircuit.v[:] = microcircuit.microcircuit.v0[:]
        microcircuit.microcircuit.vt[:] = microcircuit.microcircuit.vt0[:]

        network.reinit(states=False)
        network.run(maxAvalTime)

        avalTimes[n] = avalDetect.elapsedTime
        spikeCounts[n] = avalDetect.spikeCounts

    # Show histogram of avalanche sizes
    tmin = 0.0
    tmax = numpy.max(avalTimes)
    nbHistBins = numpy.round(tmax / defaultclock.dt)
    hist, bins = numpy.histogram(avalTimes, bins=nbHistBins, range=(tmin, tmax), normed=False)
    hist = numpy.asfarray(hist) / len(avalTimes)

    fig = pylab.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Avalanche length [sec]')
    ax.set_ylabel('Probability')
    ax.plot(bins[0:len(hist)], hist)

    # Show histogram of avalanche lengths
    fig = pylab.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Avalanche length [nb propagated spikes]')
    ax.set_ylabel('Probability')

    nmin = 1
    nmax = numpy.max(spikeCounts)
    nbHistBins = numpy.min(((nmax - nmin + 1), 24))
    hist, bins = numpy.histogram(spikeCounts, bins=nbHistBins, range=(nmin, nmax), normed=False)
    hist = numpy.asfarray(hist) / len(spikeCounts)

    idx = numpy.where(hist > 0.0)[0]
    hist = hist[idx]
    bins = bins[idx]

    ax.scatter(bins, hist)

    # Analytic curve
    npl = numpy.arange(nmin + 0.01, nmax, 0.01)
    pl = npl ** (-3.0 / 2.0)
    ax.plot(npl, pl)

    # TODO: find a way to plot correctly in log-log scale
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlim((nmin, nmax))
    ax.set_ylim((numpy.min(hist), numpy.max(hist)))

    # Fitting the power-law line in log-log domain
    logX = numpy.log(bins)
    logY = numpy.log(hist)
    p = numpy.polyfit(logX, logY, 1)

    print 'Best exponential fitting coefficients: ', p
    print 'Estimated exponent using curve fitting: %f' % (p[0])
