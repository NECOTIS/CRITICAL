'''
Created on 23.01.2012

@author: "Simon Brodeur"
'''

import logging
import numpy as np
import scipy.sparse

from brian2.units import ms, um
from brian2.groups import NeuronGroup
from brian2.synapses.synapses import Synapses
from brian2.units.allunits import meter

logger = logging.getLogger(__name__)


def createNeuronGroup(N, refractory=5 * ms, tau=50 * ms, vti=0.1):

    # Integrate-and-fire neuron with adaptive threshold
    eqs = '''
        v : 1
        dvt/dt = -(vt-vt0)/tau : 1
        tau : second
        vt0 : 1
        vti : 1

        u : 1
        ulast : 1
        ns : 1
        ctot : 1
        m : integer

        x : meter
        y : meter
        z : meter

        midx : 1
        mmidx : 1
        ntype : 1
    '''

    reset = '''
    v = 0.0
    vt += vti
    '''

    G = NeuronGroup(N, model=eqs, reset=reset, threshold='v>vt', refractory=refractory, method='exact')

    eqs = '''
    eps = 1e-3    # to avoid division by zero
    u = (ctot / (ns + eps))
    ulast = ulast * int(ns == 0) + u * int(ns > 0)
    ns = 0
    ctot = 0
    '''
    G.run_regularly(eqs, dt=100 * ms)

    # Configure neuron state variables initial values
    G.v = 0.0
    G.vt = 1.0
    G.tau = tau
    G.vt0 = 1.0
    G.vti = vti
    G.ctot = 0.0
    G.ns = 0.0
    G.u = 1.0

    return G


# FIXME: should update weights only when all post-synaptic neurons to a synapse have spiked
def createCriticalSynapses(G, forceSpiking=True):

    # Configure synaptic connections
    eqs = '''
    c : 1    # accumulation of contribution to post-synaptic spikes
    w : 1    # synaptic contribution per spike
    cbf : 1  # target critical branching factor
    lr : 1   # learning rate
    '''

    # NOTE: only consider the contribution if not in the refractory period
    on_pre = '''
    v_post += w  * int(not_refractory_post)   # add contribution to the dynamic of the post-synaptic neuron
    c += w * int(not_refractory_post)         # add contribution of presynaptic spike to contribution dynamic
    '''

    on_post = '''
    eps = 1e-3    # to avoid division by zero
    ctot_pre += (c / v_post)                              # add true contributions of past presynaptic spikes normalized by the threshold level
    ns_pre += 1
    _u = ((ctot_pre + (c / v_post)) / (ns_pre + eps))
    w = clip(w + lr * (cbf - _u), 0.0, 10.0)              # modulate weight based on the error on the critical branching factor
    c = 0.0                                               # reset the contribution dynamic for this synapse
    '''
    # s = exp((lastspike - t) / 0.1)

    S = Synapses(G, G, model=eqs, on_pre=on_pre, on_post=on_post)

    if forceSpiking:
        # Increase the weight if no postsynaptic firing occured in a given interval
        eqs = '''
        eps = 1e-3    # to avoid division by zero
        _u = (ctot_pre / (ns_pre + eps))
        _v = xor(_u, eps)
        w = clip(w + lr * (cbf - _u) * int(ns_pre == 0), 0.0, 10.0)
        '''
        S.run_regularly(eqs, dt=450 * ms)

    return S


class Microcircuit(object):
    '''
    Microcircuit
    '''

    def __init__(self, connectivity='small-world', macrocolumnShape=[2, 2, 2], minicolumnShape=[4, 4, 4], minicolumnSpacing=100 * um, neuronSpacing=10 * um):

        self.__dict__.update(macrocolumnShape=macrocolumnShape, minicolumnShape=minicolumnShape,
                             minicolumnSpacing=minicolumnSpacing, neuronSpacing=neuronSpacing)

        # Create the microcircuit
        logger.debug('Creating the microcircuit...')
        self.nbMinicolumns = np.prod(self.macrocolumnShape)
        self.nbNeuronsMinicolumn = np.prod(self.minicolumnShape)
        self.nbNeuronsTotal = self.nbNeuronsMinicolumn * self.nbMinicolumns

        self.G = createNeuronGroup(self.nbNeuronsTotal)

        # Set the 3D coordinates of the neurons
        # Loop over macrocolums
        positions = []
        macrocolumIndices = []
        minicolumIndices = []
        for im in range(self.macrocolumnShape[0]):
            for jm in range(self.macrocolumnShape[1]):
                for km in range(self.macrocolumnShape[2]):
                    # Loop over minicolums
                    for i in range(self.minicolumnShape[0]):
                        for j in range(self.minicolumnShape[1]):
                            for k in range(self.minicolumnShape[2]):
                                xn = float(im * (self.minicolumnShape[0]) + i) * self.neuronSpacing
                                yn = float(jm * (self.minicolumnShape[1]) + j) * self.neuronSpacing
                                zn = float(km * (self.minicolumnShape[2]) + k) * self.neuronSpacing
                                positions.append([xn, yn, zn])
                                macrocolumIndices.append(im * self.macrocolumnShape[0] + jm * self.macrocolumnShape[1] + km)
                                minicolumIndices.append(i * self.minicolumnShape[0] + j * self.minicolumnShape[1] + k)
        positions = np.array(positions, dtype=np.float32)
        macrocolumIndices = np.array(macrocolumIndices, dtype=np.int)
        minicolumIndices = np.array(minicolumIndices, dtype=np.int)

        # Configure neuron state variables for storing 3D absolute positions
        self.G.x = positions[:, 0] * meter
        self.G.y = positions[:, 1] * meter
        self.G.z = positions[:, 2] * meter
        self.G.midx = macrocolumIndices
        self.G.mmidx = minicolumIndices

        # Configure neuron state variables initial values
        logger.debug('Configuring neuron state variables initial values...')
        self.G.v = 0.0

        # Configure excitatory/inhibitory type of neurons
        logger.debug('Configuring excitatory/inhibitory type of neurons...')
        excitatoryProb = 0.8
        neuronTypes = np.random.random(self.nbNeuronsTotal)
        ntypes = np.zeros((self.nbNeuronsTotal,), dtype=np.float32)
        ntypes[np.where(neuronTypes < excitatoryProb)] = 1.0
        ntypes[np.where(neuronTypes >= excitatoryProb)] = -1.0
        self.G.ntype = ntypes

        self.S = createCriticalSynapses(self.G)

        if isinstance(connectivity, str):
            logger.debug('Creating network topology ''%s'' ...' % (connectivity))
            if connectivity == 'small-world':
                # Connections inside minicolums
                self.S.connect(condition='i != j and mmidx_pre == mmidx_post',
                               p='p_max*exp(-(x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2) / (3*(125*umeter)**2)',
                               namespace={'p_max': 0.01})

                # Connections across minicolums
                self.S.connect(condition='i != j and mmidx_pre != mmidx_post',
                               p='p_max*exp(-(x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2) / (3*(125*umeter)**2)',
                               namespace={'p_max': 0.01})

                wmin = 0.0
                wmax = 1.0
                self.S.w = np.random.uniform(wmin, wmax, size=len(self.S.w))

            elif connectivity == 'random':
                # Connections inside minicolums
                self.S.connect(condition='i != j',
                               p='p_max*exp(-(x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2) / (3*(125*umeter)**2)',
                               namespace={'p_max': 0.01})

            else:
                raise Exception('Unsupported connectivity: %s' % (connectivity))
        else:
            # The connectivity matrix is directly provided
            logger.debug('Loading network topology...')
            cx = scipy.sparse.coo_matrix(connectivity)
            self.S.connect(i=cx.row, j=cx.col)
            for i, j, v in zip(cx.row, cx.col, cx.data):
                self.S.w[i, j] = v

    def printConnectivityStats(self):

        logger.debug('Calculating microcircuit connectivity statistics...')

        # Number of isolated neurons and weakly-connected neurons (doesn't include isolated neurons)
        nbIsolatedNeurons = 0
        weaklyConnectedThreshold = 2
        nbWeaklyConnectedNeurons = 0

        for i in range(self.nbNeuronsTotal):
            nbOutputConnections = np.count_nonzero(self.S.w[i, :])
            nbInputConnections = np.count_nonzero(self.S.w[:, i])

            if nbOutputConnections == 0 or nbInputConnections == 0:
                nbIsolatedNeurons = nbIsolatedNeurons + 1
            elif nbInputConnections > 0 and nbInputConnections < weaklyConnectedThreshold:
                nbWeaklyConnectedNeurons = nbWeaklyConnectedNeurons + 1

        # Average number of synapses/neuron
        avgSynapsesPerNeuron = float(len(self.S)) / self.nbNeuronsTotal

        # Total number of synapses
        nbSynapsesTotal = len(self.S)

        # Neuron types
        ratioExcitatory = float(len(np.where(self.G.ntype > 0)[0])) / self.nbNeuronsTotal
        ratioInhibitory = float(len(np.where(self.G.ntype < 0)[0])) / self.nbNeuronsTotal

        print('############## MICROCIRCUIT CONNECTIVITY STATISTICS ###############')
        print('Macrocolumn shape: (%d x %d x %d)' % (self.macrocolumnShape[0], self.macrocolumnShape[1], self.macrocolumnShape[2]))
        print('Number of minicolumns: %d' % (self.nbMinicolumns))
        print('Minicolumn shape: (%d x %d x %d)' % (self.minicolumnShape[0], self.minicolumnShape[1], self.minicolumnShape[2]))
        print('Number of neurons per minicolumn: %d' % (self.nbNeuronsMinicolumn))
        print('-------------------------------------------------------------------')
        print('Total number of neurons: %d' % (self.nbNeuronsTotal))
        print('Neuron type ratio: excitatory (%3.2f%%), inhibitory(%3.2f%%)' % (ratioExcitatory * 100, ratioInhibitory * 100))
        print('Total number of isolated neurons: %d (%3.2f%%)' % (nbIsolatedNeurons, float(nbIsolatedNeurons) / self.nbNeuronsTotal * 100.0))
        print('Total number of weakly-connected neurons (< %d input synapses): %d (%3.2f%%)' % (weaklyConnectedThreshold, nbWeaklyConnectedNeurons, float(nbWeaklyConnectedNeurons) / self.nbNeuronsTotal * 100.0))

        print('Average number of synapses/neuron: %1.2f' % (avgSynapsesPerNeuron))
        print('Total number of synapses: %d' % (nbSynapsesTotal))
        print('###################################################################')
