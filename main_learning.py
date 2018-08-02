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

import defaultparams
from learning import *
from microcircuit import *
from monitors import *
from testing import *

def comparePonderationScheme(structure, repeatGenerate, NbAval, maxAvalTime, trainingTime):

    defaultclock.dt = defaultparams.getDefaultSimulationTimeStep()
    nbBins = int(maxAvalTime / defaultclock.dt)
    histAcc = numpy.zeros(nbBins)

    # Visualize histogram
    fig = pylab.figure(facecolor='white', figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Avalanche length [sec]')
    ax.set_ylabel('Probability')

    ponderations = ['l', 'u', 'f']
    lineStyles = ['-', '--', '-.']
    labels = ['last-privileged', 'uniform', 'first-privileged']
    for n in range(len(ponderations)):
        i = 0
        successGen = 0
        while successGen < repeatGenerate:
            print 'Generating microcircuit no.%d ...' % (i)
            N, connectivity, connectParams = defaultparams.getDefaultConnectivityParams(structure)
            neuronModel, modelParams = defaultparams.getDefaultMicrocircuitModelParams()

            microcircuit = Microcircuit(N, neuronModel, modelParams, connectivity, connectParams)
            network = Network(microcircuit.microcircuit, microcircuit.connections)

            print 'Training microcircuit ...'
            method, trainParams = defaultparams.getDefaultTrainingParams()
            trainParams['refractory'] = modelParams['refractory']
            trainParams['ponderation'] = ponderations[n]

            # Learning rule
            tuner = LTSOCP(microcircuit, trainParams)
            network.add(tuner)

            # Input
            input, inputConnections = defaultparams.createDefaultPoissonInput(microcircuit, tuner)
            network.add(input)
            network.add(inputConnections)

            network.reinit(states=False)
            network.run(trainingTime)

            print 'Testing microcircuit ...'
            hist, bins = generateAvalancheLengthHist(microcircuit, inputConnections, NbAval=NbAval, maxAvalTime=maxAvalTime)

            # Accumulate histogram values
            histAcc += hist
            successGen += 1
            i += 1

        hist = histAcc / successGen
        ax.plot(bins[0:len(hist)], hist, color='black', linestyle=lineStyles[n], label=labels[n])
        fig.canvas.draw()

    ax.legend(loc='upper right')
    fig.canvas.draw()
    fig.savefig('results/latex/compare_ponderation_scheme_%s.eps' % (structure))

def generateHistogramLearningRule(structure, ponderation, repeatGenerate, NbAval, maxAvalTime):

    defaultclock.dt = defaultparams.getDefaultSimulationTimeStep()
    nbBins = int(maxAvalTime / defaultclock.dt)
    histAcc = numpy.zeros(nbBins)

    # Visualize histogram
    fig = pylab.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)

    fig2 = pylab.figure(facecolor='white')
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_xlabel('Avalanche length [sec]')
    ax2.set_ylabel('Probability')

    i = 0
    successGen = 0
    while successGen < repeatGenerate:
        print 'Generating microcircuit no.%d ...' % (i)
        N, connectivity, connectParams = defaultparams.getDefaultConnectivityParams(structure)
        neuronModel, modelParams = defaultparams.getDefaultMicrocircuitModelParams()

        microcircuit = Microcircuit(N, neuronModel, modelParams, connectivity, connectParams)
        network = Network(microcircuit.microcircuit, microcircuit.connections)

        print 'Training microcircuit ...'
        duration = 2000 * ms
        method, trainParams = defaultparams.getDefaultTrainingParams()
        trainParams['refractory'] = modelParams['refractory']
        trainParams['ponderation'] = ponderation

        # Learning rule
        tuner = LTSOCP(microcircuit, trainParams)
        network.add(tuner)

        # Input
        input, inputConnections = defaultparams.createDefaultPoissonInput(microcircuit, tuner)
        network.add(input)
        network.add(inputConnections)

        network.reinit(states=False)
        network.run(duration)

        print 'Testing microcircuit ...'
        hist, bins = generateAvalancheLengthHist(microcircuit, NbAval=NbAval, maxAvalTime=maxAvalTime)

        # Accumulate histogram values
        histAcc += hist
        successGen += 1

        ax.clear()
        ax.plot(bins[0:len(histAcc)], histAcc / successGen, color='black')
        ax.set_xlabel('Avalanche length [sec]')
        ax.set_ylabel('Probability')
        fig.canvas.draw()

        ax2.plot(bins[0:len(hist)], hist, color='black')
        fig2.canvas.draw()
        i += 1

    ax2.plot(bins[0:len(histAcc)], histAcc / successGen, color='red', linestyle='--')

    fig.savefig('results/latex/aval_distribution_learningRule_%s_%s.eps' % (structure, ponderation))
    fig2.savefig('results/latex/aval_distribution_learningRule_var_%s_%s.eps' % (structure, ponderation))

    pylab.close(fig)
    pylab.close(fig2)

if __name__ == '__main__':

    numpy.random.seed(0)
    numpy.seterr(all='raise')
    pylab.ion()

    repeatGenerate = 20
    NbAval = 1000
    maxAvalTime = 150 * ms
    trainingTime = 1000 * ms
    structure = 'small-world'
    comparePonderationScheme(structure, repeatGenerate, NbAval, maxAvalTime, trainingTime)

#    # Simulation parameters
#    repeatGenerate = 5
#    NbAval = 1000
#    maxAvalTime = 150 * ms
#    structures = ['random', 'small-world']
#    for structure in structures:
#        ponderations = ['l', 'f', 'u']
#        for ponderation in ponderations:
#            generateHistogramLearningRule(structure, ponderation, repeatGenerate, NbAval, maxAvalTime)

    print 'All done.'
    pylab.ioff()
    pylab.show()
    pylab.close()
