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

def calculatePowerLawExponent(structure, NbAval, maxAvalTime, trainingTime):

    print 'Generating microcircuit...'
    N, connectivity, connectParams = defaultparams.getDefaultConnectivityParams(structure)
    neuronModel, modelParams = defaultparams.getDefaultMicrocircuitModelParams()

    microcircuit = Microcircuit(N, neuronModel, modelParams, connectivity, connectParams)
    network = Network(microcircuit.microcircuit, microcircuit.connections)

    print 'Training microcircuit...'
    method, trainParams = defaultparams.getDefaultTrainingParams()
    trainParams['refractory'] = modelParams['refractory']

    # Learning rule
    tuner = LTSOCP(microcircuit, trainParams)
    network.add(tuner)

    # Input
    input, inputConnections = defaultparams.createDefaultPoissonInput(microcircuit, tuner)
    network.add(input)
    network.add(inputConnections)

    network.reinit(states=False)
    network.run(trainingTime)

    print 'Testing microcircuit...'
    estimatePowerLawScaling(microcircuit, inputConnections, NbAval, maxAvalTime)


if __name__ == '__main__':

    numpy.random.seed(0)
    numpy.seterr(all='raise')
    pylab.ion()

    # Simulation parameters
    NbAval = 5000
    maxAvalTime = 150 * ms
    trainingTime = 1000 * ms
    structure = 'small-world'
    calculatePowerLawExponent(structure, NbAval, maxAvalTime, trainingTime)

    print 'All done.'
    pylab.ioff()
    show()
    pylab.close()
