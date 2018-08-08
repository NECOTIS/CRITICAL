'''
Created on 23.01.2012

@author: "Simon Brodeur"
'''

from brian import *
import math
import numpy
import types
from numpy import linalg
import pylab, matplotlib
import scipy
import scipy.sparse.linalg
from scipy import sparse
from scipy import signal
from scipy import *
import scipy.interpolate
from itertools import izip
import itertools
import defaultparams

def _convertConnectionMatrix(C):
    # Compressed format
    W = scipy.sparse.lil_matrix(C.W.shape)
    for i in range(C.W.shape[0]):
        row = C.W[i, :]
        for j, w in izip(row.ind, row):
            W[i, j] = w
    return W

def _quadraticBezierPath(posA, posB, nstep=8, controlDist=0.1, controlDim=[0, 1, 2]):
    segments = []

    vecP1 = posB - posA
    if numpy.linalg.norm(vecP1) == 0.0:
        raise Exception('Same start and end vertices!')

    vecP2 = numpy.array([0, 0, 0])
    vecP2[controlDim[2]] = 1.0
    vecC = numpy.cross(vecP1, vecP2)
    if numpy.linalg.norm(vecC) == 0.0:
        # The vectors are colinear so change the reference plane
        vecP2 = numpy.array([0, 0, 0])
        vecP2[controlDim[1]] = 1.0
    vecC = numpy.cross(vecP1, vecP2)
    vecC = vecC / numpy.linalg.norm(vecC) * controlDist

    posD = posA + (posB - posA) / 2.0
    posC = posD + vecC

    for t in numpy.linspace(0, 1.0, nstep):
        pos = posA * (1 - t) ** 2 + posB * t ** 2 + posC * 2 * t * (1 - t)
        segments.append(pos)
    return segments

def _maxEuclideanDistance(macrocolumnShape, minicolumnShape):
    totalLength = minicolumnShape[0] * macrocolumnShape[0]
    totalWidth = minicolumnShape[1] * macrocolumnShape[1]
    totalHeight = minicolumnShape[2] * macrocolumnShape[2]

    # Calculate the euclidean distance between the neurons
    posMin = numpy.array([0, 0, 0])
    posMax = numpy.array([totalLength, totalWidth, totalHeight])
    D = numpy.sqrt(numpy.sum((posMin - posMax) ** 2))
    return D

def getParameter(params, name, defaultValue=None):
    value = defaultValue
    if name in params:
        value = params[name]
    return value

class Microcircuit():
    '''
    Microcircuit
    '''

    def __init__(self, N, neuronModel='lif', modelParams={}, connectivity='random-uniform', connectParams={}):

        # Create the microcircuit
        print 'Creating the microcircuit...'
        self.macrocolumnShape = getParameter(connectParams, 'macrocolumnShape', [1, 1, 1])
        self.minicolumnShape = getParameter(connectParams, 'minicolumnShape', [1, 1, N])
        self.nbMinicolumns = numpy.prod(self.macrocolumnShape)
        self.nbNeuronsMinicolumn = numpy.prod(self.minicolumnShape)
        self.nbNeuronsTotal = self.nbNeuronsMinicolumn * self.nbMinicolumns

        self.modelParams = modelParams
        self.connectParams = connectParams

        defaultclock.dt = defaultparams.getDefaultSimulationTimeStep()

        if neuronModel == 'lif_adapt':
            # Leaky integrate-and-fire model with noise and adaptive threshold
            vti = getParameter(self.modelParams, 'vti', 0.10)
            tau_vt = getParameter(self.modelParams, 'tau_vt', 50 * ms)
            sigma_n = getParameter(self.modelParams, 'sigma_n', 0.0)
            tau_n = getParameter(self.modelParams, 'tau_n', 25 * ms)

            refractory = getParameter(self.modelParams, 'refractory', 5 * ms)
            eqs = '''
            dv/dt = -(v-v0)/tau + xi*sigma_n/tau_n**0.5: volt
            dvt/dt = -(vt-vt0)/tau_vt : volt
            tau : second
            v0 : volt
            vt0 : volt
            '''

            def applyThresh(P, spikes):
                P.v[spikes] = P.v0[spikes]
                P.vt[spikes] += vti

            resetFunc = SimpleCustomRefractoriness(applyThresh, period=refractory, state='v')
            microcircuit = NeuronGroup(self.nbNeuronsTotal, model=eqs, reset=resetFunc, threshold='v>vt', refractory=refractory, compile=True, freeze=True)

            # Configure neuron state variables initial values
            print 'Configuring neuron state variables initial values...'
            microcircuit.v = 0.0
            microcircuit.vt0 = numpy.random.uniform(1.0, 2.0, self.nbNeuronsTotal)
            microcircuit.vt[:] = microcircuit.vt0[:]
            microcircuit.v0 = numpy.random.uniform(0.5, 1.0, self.nbNeuronsTotal)
            microcircuit.tau = numpy.random.uniform(15 * ms, 25 * ms, self.nbNeuronsTotal)

        else:
            raise Exception('Unsupported neuron model: %s' % (neuronModel))

        self.microcircuit = microcircuit

        # Configure excitatory/inhibitory type of neurons
        print 'Configuring excitatory/inhibitory type of neurons...'
        excitatoryProb = getParameter(self.modelParams, 'excitatoryProb', 0.8)
        self.neuronTypes = numpy.random.random(self.nbNeuronsTotal)
        idxExcitatory = numpy.where(self.neuronTypes < excitatoryProb)
        idxInhibitory = numpy.where(self.neuronTypes >= excitatoryProb)
        self.neuronTypes[idxExcitatory] = 1.0
        self.neuronTypes[idxInhibitory] = -1.0

        delay = getParameter(self.connectParams, 'delay', 0.0 * ms)
        self.connections = Connection(microcircuit, microcircuit, state='v', delay=delay, structure='sparse')

        # Calculate and cache 3D positions of neurons (absolute and relative)
        self.neuronsAbsolutePos = numpy.zeros((self.nbNeuronsTotal, 3))
        self.neuronsRelativePos = numpy.zeros((self.nbNeuronsTotal, 3))
        for i in range(self.nbNeuronsTotal):
            posAbs, posRel = self._calculate3DNeuronCoordinates(i)
            self.neuronsAbsolutePos[i] = posAbs
            self.neuronsRelativePos[i] = posRel

        if isinstance(connectivity, str):
            print 'Creating network topology ''%s'' ...' % (connectivity)
            if connectivity == 'small-world':
                self._applySmallWorldConnectivity()
            elif connectivity == 'random-uniform':
                self._applyRandomUniformConnectivity()
            elif connectivity == 'double-power-law':
                self._applyDoublePowerLawConnectivity()
            elif connectivity == 'hub':
                self._applyHubConnectivity()
            else:
                raise Exception('Unsupported connectivity: %s' % (connectivity))
        else:
            # The connectivity matrix is directly provided
            print 'Loading network topology...'
            cx = scipy.sparse.coo_matrix(connectivity)
            for i, j, v in zip(cx.row, cx.col, cx.data):
                self.connections.W[i, j] = v

        self.connections.compress()

    def getSparseConnectivityMatrix(self):
       return _convertConnectionMatrix(self.connections)

    def _applySmallWorldConnectivity(self):

        nbSynapsesPerNeuron = getParameter(self.connectParams, 'm', 8)
        wmin = getParameter(self.connectParams, 'wmin', 0.0)
        wmax = getParameter(self.connectParams, 'wmax', 1.0)
        intercolumnarSynapsesRatio = getParameter(self.connectParams, 'intercolumnarSynapsesRatio', 0.25)
        intercolumnarStrengthFactor = getParameter(self.connectParams, 'intercolumnarStrengthFactor', 0.85)
        intracolumnarSparseness = getParameter(self.connectParams, 'intracolumnarSparseness', 8.0)
        intercolumnarSparseness = getParameter(self.connectParams, 'intercolumnarSparseness', 8.0)

        # Configure each minicolumn within the microcircuit
        print 'Configuring each minicolumn within the microcircuit...'
        for n in range(self.nbMinicolumns):
            # The boundary indexes of the current minicolumn
            neuronIdxStart = n * self.nbNeuronsMinicolumn
            neuronIdxEnd = neuronIdxStart + self.nbNeuronsMinicolumn

            # Configure intracolumnar connectivity:
            print 'Configuring intracolumnar connectivity for minicolumn %d...' % (n)
            if isinstance(nbSynapsesPerNeuron, types.ListType):
                nbSynapses = numpy.random.randint(nbSynapsesPerNeuron[0], nbSynapsesPerNeuron[1], self.nbNeuronsMinicolumn)
            else:
                nbSynapses = nbSynapsesPerNeuron * numpy.ones(self.nbNeuronsMinicolumn)

            for i in range(neuronIdxStart, neuronIdxEnd):
                targetsPos = self._createIntracolumnarConnections(i, nbSynapses[i - neuronIdxStart], intracolumnarSparseness)
                for targetPos in targetsPos:
                    j = self._getAbsoluteNeuronIndex(targetPos)
                    self.connections[i, j] = self.neuronTypes[i] * numpy.random.uniform(wmin, wmax)
                    #connections.delay[i, j] = delay

        if self.nbMinicolumns > 1:
            # Configure intercolumnar connectivity
            print 'Configuring intercolumnar connectivity...'
            nbIntracolumnarConnections = self.connections.W.nnz
            for n in range(self.nbMinicolumns):
                print 'Configuring intercolumnar connectivity for minicolumn %d...' % (n)
                # The boundary indexes of the source minicolumn
                neuronIdxStart = n * self.nbNeuronsMinicolumn
                neuronIdxEnd = neuronIdxStart + self.nbNeuronsMinicolumn

                # Select a random subset of neurons within the minicolumn
                nbIntercolumnarConnections = numpy.round(nbIntracolumnarConnections * intercolumnarSynapsesRatio / self.nbMinicolumns)
                idx = numpy.random.randint(neuronIdxStart, neuronIdxEnd, nbIntercolumnarConnections)
                for i in idx:
                    w = 0
                    while w < 1:
                        targetPos = self._createIntercolumnarConnections(i, 1, intercolumnarSparseness)[0]
                        j = self._getAbsoluteNeuronIndex(targetPos)
                        if self.connections[i, j] == 0.0:
                            self.connections[i, j] = self.neuronTypes[i] * intercolumnarStrengthFactor * numpy.random.uniform(wmin, wmax)
                            w = w + 1

    def _applyRandomUniformConnectivity(self):

        # Fixed number of output synapses
        nbSynapsesPerNeuron = getParameter(self.connectParams, 'm', 8)
        wmin = getParameter(self.connectParams, 'wmin', 0.0)
        wmax = getParameter(self.connectParams, 'wmax', 1.0)

        for i in range(self.nbNeuronsTotal):
            idx = range(0, self.nbNeuronsTotal)
            numpy.random.shuffle(idx)
            idx = idx[0:nbSynapsesPerNeuron + 1]

            c = 0
            for j in idx:
                # Do not allow self-recurrent connection
                if i != j:
                    self.connections[i, j] = numpy.sign(self.neuronTypes[i]) * numpy.random.uniform(wmin, wmax)
                    c += 1
                    if c == nbSynapsesPerNeuron:
                        break

    def _applyDoublePowerLawConnectivity(self):
        raise NotImplementedError()

    def _applyHubConnectivity(self):
        raise NotImplementedError()

    def _connectivityParameters(self, a, b):

        # Calculate the euclidean distance between the neurons
        posA = self._get3DAbsoluteCoordinates(a)
        posB = self._get3DAbsoluteCoordinates(b)
        dist = numpy.sqrt(numpy.sum((posA - posB) ** 2))
        distProb = numpy.exp(-dist ** 2 / self.intracolumnarSparseness)

        # Probability and delay based on the pre- and post-synaptic neurons types
        if self.neuronTypes[a] > 0 and self.neuronTypes[b] > 0:
            # EE connection
            typeProb = 0.3
            delayFactor = 1.5 * ms
        elif self.neuronTypes[a] > 0 and self.neuronTypes[b] < 0:
            # EI connection
            typeProb = 0.2
            delayFactor = 0.8 * ms
        elif self.neuronTypes[a] < 0 and self.neuronTypes[b] > 0:
            # IE connection
            typeProb = 0.4
            delayFactor = 0.8 * ms
        else:
            # II connection
            typeProb = 0.1
            delayFactor = 0.8 * ms

        # Total probability of connection
        prob = typeProb * distProb
        delayVariation = 0.15
        randomDelayFactor = (1.0 - delayVariation) * numpy.random.random() + delayVariation
        delay = delayFactor * randomDelayFactor * dist
        return (prob, delay)

    def _get3DAbsoluteCoordinates(self, a):
        return self.neuronsAbsolutePos[a]

    def _getRelativeNeuronIndex(self, pos):
        iAbs, jAbs, kAbs = pos

        iM = numpy.floor(iAbs / float(self.minicolumnShape[0]))
        jM = numpy.floor(jAbs / float(self.minicolumnShape[1]))
        kM = numpy.floor(kAbs / float(self.minicolumnShape[2]))

        iRel = iAbs - iM * self.minicolumnShape[0]
        jRel = jAbs - jM * self.minicolumnShape[1]
        kRel = kAbs - kM * self.minicolumnShape[2]

        # Get the subspace index within the 3D minicolumn
        c = iRel * self.minicolumnShape[1] + jRel + kRel * (self.minicolumnShape[0] * self.minicolumnShape[1])
        return c

    def _getAbsoluteNeuronIndex(self, pos):
        iAbs, jAbs, kAbs = pos

        iM = numpy.floor(iAbs / float(self.minicolumnShape[0]))
        jM = numpy.floor(jAbs / float(self.minicolumnShape[1]))
        kM = numpy.floor(kAbs / float(self.minicolumnShape[2]))

        iRel = iAbs - iM * self.minicolumnShape[0]
        jRel = jAbs - jM * self.minicolumnShape[1]
        kRel = kAbs - kM * self.minicolumnShape[2]

        # Get the subspace index within the 3D minicolumn
        c = iRel * self.minicolumnShape[1] + jRel + kRel * (self.minicolumnShape[0] * self.minicolumnShape[1])

        # Get the minicolumn index
        n = iM * self.macrocolumnShape[1] + jM + kM * (self.macrocolumnShape[0] * self.macrocolumnShape[1])

        # Get the neuron index
        a = c + n * self.nbNeuronsMinicolumn
        return a

    def _get3DRelativeCoordinates(self, a):
        return self.neuronsRelativePos[a]

    def _get3DRelativeNeuronIndex(self, a):
        # Get the subspace index within the 3D minicolumn
        b = numpy.mod(a, self.nbNeuronsMinicolumn)
        return b

    def _createIntercolumnarConnections(self, a, nbConnections, intercolumnarSparseness):
        posA = self._get3DAbsoluteCoordinates(a)
        constraints = self._get3DIntercolumnarConstraints(a)

        samples = numpy.zeros((nbConnections, 3))
        n = 0
        while n < nbConnections:

            # Try to generate a valable sample
            posR = numpy.zeros(3)
            # Loop for each dimension
            for dim in range(3):
                dminL, dmaxL, dminR, dmaxR = constraints[dim, :]

                r = 2.0 * numpy.random.rand() - 1.0
                if r > 1.0 / 3.0:
                    # Left side
                    dSign = 1.0
                    dmin = dminL
                    dmax = dmaxL
                    if dminL == 0.0:
                        # Nothing on left side, so forced choose the right side
                        dSign *= -1
                        dmin = dminR
                        dmax = dmaxR
                elif r < -1.0 / 3.0:
                    # Right side
                    dSign = -1.0
                    dmin = dminR
                    dmax = dmaxR
                    if dminR == 0.0:
                        # Nothing on right side, so forced to choose the left side
                        dSign *= -1
                        dmin = dminL
                        dmax = dmaxL
                else:
                    dmin = 0.0
                    dmax = 0.0

                if dmin > 0.0 and dmax > 0.0:
                    if dmin == dmax:
                        dmin = 0.0

                    # Perform uniform Monte-Carlo sampling in the pdf domain following the constraints
                    pmax = numpy.exp(-dmin ** 2 / intercolumnarSparseness)
                    pmin = numpy.exp(-dmax ** 2 / intercolumnarSparseness)
                    pSample = (pmax - pmin) * numpy.random.rand() + pmin

                    # Calculate the distance from the pdf sample
                    dSample = numpy.sqrt(-intercolumnarSparseness * numpy.log(pSample))
                    dSample = numpy.round(dSample)
                    posR[dim] = -dSign * dSample
                else:
                    posR[dim] = 0.0

            # Check validity of the sample
            if numpy.linalg.norm(posR) == 0:
                # Not connected
                continue

            posT = posA + posR
            if filter(lambda row: numpy.array_equal(row, posT), samples):
                # Existing connection
                continue

            samples[n] = posT
            n = n + 1

        return samples

    def _createIntracolumnarConnections(self, a, nbConnections, intracolumnarSparseness, strict=True):
        posA = self._get3DAbsoluteCoordinates(a)
        constraints = self._get3DIntracolumnarConstraints(a)

        samples = numpy.zeros((nbConnections, 3))

        idxA = self._get3DRelativeNeuronIndex(a)
        occupacyTable = numpy.zeros(self.nbNeuronsMinicolumn, dtype='bool')
        occupacyTable[idxA] = True

        n = 0
        while n < nbConnections:
            # Try to generate a valid sample
            posR = numpy.zeros(3)

            # Loop for each dimension
            for dim in range(3):
                dminL, dmaxL, dminR, dmaxR = constraints[dim, :]

                r = 2.0 * numpy.random.rand() - 1.0
                if r > 1.0 / 3.0:
                    # Left side
                    dSign = 1.0
                    dmin = dminL
                    dmax = dmaxL
                    if dminL == 0.0:
                        # Nothing on left side, so forced choose the right side
                        dSign *= -1
                        dmin = dminR
                        dmax = dmaxR
                elif r < -1.0 / 3.0:
                    # Right side
                    dSign = -1.0
                    dmin = dminR
                    dmax = dmaxR
                    if dminR == 0.0:
                        # Nothing on right side, so forced to choose the left side
                        dSign *= -1
                        dmin = dminL
                        dmax = dmaxL
                else:
                    dmin = 0.0
                    dmax = 0.0

                if dmin > 0.0 and dmax > 0.0:
                    if dmin == dmax:
                        dmin = 0.0

                    # Perform uniform Monte-Carlo sampling in the pdf domain following the constraints
                    pmax = numpy.exp(-dmin ** 2 / intracolumnarSparseness)
                    pmin = numpy.exp(-dmax ** 2 / intracolumnarSparseness)
                    pSample = (pmax - pmin) * numpy.random.rand() + pmin

                    # Calculate the distance from the pdf sample
                    dSample = numpy.sqrt(-intracolumnarSparseness * numpy.log(pSample))
                    dSample = numpy.round(dSample)
                    posR[dim] = -dSign * dSample
                else:
                    posR[dim] = 0.0

            # Check validity of the sample
            if numpy.linalg.norm(posR) == 0:
                # Not connected
                continue

            posT = posA + posR
            idx = self._getRelativeNeuronIndex(posT)
            if occupacyTable[idx] == True:
                if not strict:
                    n = n + 1
                continue
            else:
                occupacyTable[idx] = True

            samples[n, :] = posT
            n = n + 1

        return samples

    def _get3DIntracolumnarConstraints(self, a):
        constraints = numpy.zeros((3, 4))
        posA = self._get3DRelativeCoordinates(a)

        for i in range(len(posA)):
            #Boundaries on the left
            dminL = 1.0
            dmaxL = posA[i]
            if posA[i] == 0:
                dminL = 0.0
                dmaxL = 0.0

            #Boundaries on the right
            dminR = 1.0
            dmaxR = self.minicolumnShape[i] - posA[i] - 1
            if posA[i] == self.minicolumnShape[i] - 1:
                dminR = 0.0
                dmaxR = 0.0

            constraints[i] = numpy.array([dminL, dmaxL, dminR, dmaxR])

        return constraints

    def _get3DIntercolumnarConstraints(self, a):
            constraints = numpy.zeros((3, 4))
            posA_rel = self._get3DRelativeCoordinates(a)

            # Get the minicolumn index
            n = self._getMinicolumnIndex(a)
            posM = self._calculate3DMicrocolumnCoordinates(n)

            for i in range(3):
                #Boundaries on the left
                if posM[i] == 0:
                    dminL = 0.0
                    dmaxL = 0.0
                else:
                    dminL = posA_rel[i] + 1
                    dmaxL = dminL + posM[i] * self.minicolumnShape[i] - 1

                #Boundaries on the right
                if posM[i] == self.macrocolumnShape[i] - 1:
                    dminR = 0.0
                    dmaxR = 0.0
                else:
                    dminR = self.minicolumnShape[i] - posA_rel[i]
                    dmaxR = dminR + (self.macrocolumnShape[i] - (posM[i] + 1)) * self.minicolumnShape[i] - 1

                constraints[i] = numpy.array([dminL, dmaxL, dminR, dmaxR])

            return constraints

    def _getMinicolumnIndex(self, a):
        return numpy.floor(a / float(self.nbNeuronsMinicolumn))

    def _calculate3DMicrocolumnCoordinates(self, n):
        kAbs = numpy.floor(n / float(self.macrocolumnShape[0] * self.macrocolumnShape[1]))
        b = numpy.mod(n, self.macrocolumnShape[0] * self.macrocolumnShape[1])
        iAbs = numpy.floor(b / float(self.macrocolumnShape[1]))
        jAbs = numpy.mod(b, self.macrocolumnShape[1])
        return numpy.array([iAbs, jAbs, kAbs])

    def _calculate3DNeuronCoordinates(self, a):

        # Get the minicolumn index
        n = self._getMinicolumnIndex(a)
        if n >= self.nbMinicolumns:
            raise Exception('Neuron is outside the defined column shapes: %d' % (a))

        # Get the subspace index within the 3D minicolumn
        b = numpy.mod(a, self.nbNeuronsMinicolumn)

        # Get the neuron height position (relative to minicolumn)
        kRel = numpy.floor(b / float(self.minicolumnShape[0] * self.minicolumnShape[1]))

        # Get the subspace index within the 2D layer
        c = numpy.mod(b, self.minicolumnShape[0] * self.minicolumnShape[1])

        # Get the neuron length and width positions (relative to minicolumn)
        iRel = numpy.floor(c / float(self.minicolumnShape[1]))
        jRel = numpy.mod(c, self.minicolumnShape[1])

        # Get the neuron length and width positions (relative to macrocolumn)
        iM, jM, kM = self._calculate3DMicrocolumnCoordinates(n)
        iAbs = iRel + iM * self.minicolumnShape[0]
        jAbs = jRel + jM * self.minicolumnShape[1]
        kAbs = kRel + kM * self.minicolumnShape[2]

        return (numpy.array([iAbs, jAbs, kAbs]), numpy.array([iRel, jRel, kRel]))

    def printConnectivityStats(self, showDetails=False, showHistogram=False):

        print 'Calculating microcircuit connectivity statistics...'

        if showDetails:
            # Number of isolated neurons and weakly-connected neurons (doesn't include isolated neurons)
            nbIsolatedNeurons = 0
            weaklyConnectedThreshold = 2
            nbWeaklyConnectedNeurons = 0

            for i in range(self.nbNeuronsTotal):
                outputs = self.connections.W[i, :]
                nbOutputConnections = len(outputs)
                inputs = self.connections.W[i, :]
                nbInputConnections = len(inputs)

                if nbOutputConnections == 0 or nbInputConnections == 0:
                    nbIsolatedNeurons = nbIsolatedNeurons + 1
                elif nbInputConnections > 0 and nbInputConnections < weaklyConnectedThreshold:
                    nbWeaklyConnectedNeurons = nbWeaklyConnectedNeurons + 1

            spectralRadius = self.calculateSpectralRadius()

        # Average number of synapses/neuron
        avgSynapsesPerNeuron = float(self.connections.W.nnz) / self.nbNeuronsTotal

        # Total number of synapses
        nbSynapsesTotal = self.connections.W.nnz

        # Neuron types
        ratioExcitatory = float(len(numpy.where(self.neuronTypes > 0)[0])) / self.nbNeuronsTotal
        ratioInhibitory = float(len(numpy.where(self.neuronTypes < 0)[0])) / self.nbNeuronsTotal

        if showHistogram:

            # Flatten sparse matrix and extract synaptic weights and distances
            W = numpy.zeros(self.connections.W.nnz)
            D = numpy.zeros(self.connections.W.nnz)
            idx = 0
            for i in range(self.connections.W.shape[0]):
                row = self.connections.W[i, :]
                for j, w in izip(row.ind, row):
                    posA = self._get3DAbsoluteCoordinates(i)
                    posB = self._get3DAbsoluteCoordinates(j)
                    d = numpy.sqrt(numpy.sum((posB - posA) ** 2))

                    W[idx] = w
                    D[idx] = d
                    idx += 1

            # Connection strength histogram
            nbHistBins = 256
            hist, bins = numpy.histogram(W, bins=nbHistBins, range=(amin(W), amax(W)), normed=True)
            hist = numpy.asfarray(hist) / len(W)

            fig = pylab.figure(facecolor='white')
            pylab.plot(bins[0:len(hist)], hist)
            pylab.title('Connectivity strength histogram')
            pylab.xlabel('Connectivity strength')
            pylab.ylabel('Probability')

            # Connection length histogram
            nbHistBins = numpy.ceil(_maxEuclideanDistance(self.macrocolumnShape, self.minicolumnShape))
            hist, bins = numpy.histogram(D, bins=nbHistBins, range=(amin(D), amax(D)), normed=True)
            hist = numpy.asfarray(hist) / len(D)

            fig = pylab.figure(facecolor='white')
            pylab.plot(bins[0:len(hist)], hist)
            pylab.title('Connectivity length histogram')
            pylab.xlabel('Connection length')
            pylab.ylabel('Probability')

        print '############## MICROCIRCUIT CONNECTIVITY STATISTICS ###############'
        print 'Macrocolumn shape: (%d x %d x %d)' % (self.macrocolumnShape[0], self.macrocolumnShape[1], self.macrocolumnShape[2])
        print 'Number of minicolumns: %d' % (self.nbMinicolumns)
        print 'Minicolumn shape: (%d x %d x %d)' % (self.minicolumnShape[0], self.minicolumnShape[1], self.minicolumnShape[2])
        print 'Number of neurons per minicolumn: %d' % (self.nbNeuronsMinicolumn)
        print '-------------------------------------------------------------------'
        print 'Total number of neurons: %d' % (self.nbNeuronsTotal)
        print 'Neuron type ratio: excitatory (%3.2f%%), inhibitory(%3.2f%%)' % (ratioExcitatory * 100, ratioInhibitory * 100)

        if showDetails:
            print 'Total number of isolated neurons: %d (%3.2f%%)' % (nbIsolatedNeurons, float(nbIsolatedNeurons) / self.nbNeuronsTotal * 100.0)
            print 'Total number of weakly-connected neurons (< %d input synapses): %d (%3.2f%%)' % (weaklyConnectedThreshold, nbWeaklyConnectedNeurons, float(nbWeaklyConnectedNeurons) / self.nbNeuronsTotal * 100.0)
            print 'Spectral radius of the connectivity matrix: %f' % (spectralRadius)

        print 'Average number of synapses/neuron: %1.2f' % (avgSynapsesPerNeuron)
        print 'Total number of synapses: %d' % (nbSynapsesTotal)
        print '###################################################################'

    def draw3D(self, minicolumnSpacing=1, showAxisLabels=True):
        from mpl_toolkits.mplot3d import axes3d, art3d

        print 'Drawing 3D microcircuit overview...'

        fig = pylab.figure(facecolor='white', figsize=(6, 5))
        ax = axes3d.Axes3D(fig)

        # Draw each minicolumn
        for n in range(self.nbMinicolumns):
            posM = self._calculate3DMicrocolumnCoordinates(n)

            # The boundary indexes of the current minicolumn
            neuronIdxStart = n * self.nbNeuronsMinicolumn
            neuronIdxEnd = neuronIdxStart + self.nbNeuronsMinicolumn

            coordonates = numpy.zeros((self.nbNeuronsMinicolumn, 3))
            idx = 0
            for i in range(neuronIdxStart, neuronIdxEnd):
                # Offset the coordinates to have a viewable spacing between columns
                coordonates[idx] = self._get3DAbsoluteCoordinates(i) + minicolumnSpacing * posM
                idx = idx + 1

            ax.plot(coordonates[:, 0], coordonates[:, 1], coordonates[:, 2], 'o')

        # Draw each synaptic connections in the microcircuit
        for i in range(self.connections.W.shape[0]):
            row = self.connections.W[i, :]
            for j, w in izip(row.ind, row):
                # Get the source and target minicolumn coordinates
                posM_a = self._calculate3DMicrocolumnCoordinates(self._getMinicolumnIndex(i))
                posM_b = self._calculate3DMicrocolumnCoordinates(self._getMinicolumnIndex(j))

                # Get the source and target neuron coordinates, introducing a spacing between minicolumns
                posA = self._get3DAbsoluteCoordinates(i) + minicolumnSpacing * posM_a
                posB = self._get3DAbsoluteCoordinates(j) + minicolumnSpacing * posM_b

                # Generate a synaptic link between the neurons with a Bezier curve
                curvePoints = _quadraticBezierPath(posA, posB, nstep=8, controlDist=0.5)
                segments = []
                for s in range(len(curvePoints) - 1):
                    segments.append((curvePoints[s], curvePoints[s + 1]))

                # Use different colors for intracolumnar, intercolumnar and inhibitory connections
                if numpy.array_equal(posM_a, posM_b):
                    #Same minicolumn
                    if self.neuronTypes[i] > 0:
                        #Excitatory
                        color = 'gray'
                    else:
                        #Inhibitory
                        color = 'dodgerblue'
                else:
                    #Different minicolumn
                    if self.neuronTypes[i] > 0:
                        #Excitatory
                        color = 'firebrick'
                    else:
                        #Inhibitory
                        color = 'yellowgreen'

                lines = art3d.Line3DCollection(segments, color=color)
                ax.add_collection3d(lines)

        # Disable any grid/pane/axis
        ax.grid(False)
        for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
            for t in a.get_ticklines() + a.get_ticklabels():
                t.set_visible(False)
            a.line.set_visible(False)
            a.pane.set_visible(False)

        if showAxisLabels:
            ax.set_xlabel('Length')
            ax.set_ylabel('Width')
            ax.set_zlabel('Height')

        return fig

    def calculateSpectralRadius(self):
        W = self.getSparseConnectivityMatrix()
        vals = scipy.sparse.linalg.eigs(W, k=6, which='LM', return_eigenvectors=False)
        spectralRadius = numpy.max(numpy.abs(vals))
        return spectralRadius

    def normalizeBySpectralRadius(self, scaleFactor=0.9):
        spectralRadius = self.calculateSpectralRadius()
        if spectralRadius != 0.0:
            for i in range(self.connections.W.shape[0]):
                row = self.connections.W[i, :]
                for j, w in izip(row.ind, row):
                    self.connections.W[i, j] = (w / spectralRadius) * scaleFactor
