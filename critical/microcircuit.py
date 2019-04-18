# Copyright (c) 2018, NECOTIS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Authors: Simon Brodeur, Jean Rouat (advisor)
# Date: November 13th, 2018
# Organization: Groupe de recherche en Neurosciences Computationnelles et Traitement Intelligent des Signaux (NECOTIS),
# Université de Sherbrooke, Canada

import logging
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, art3d
from brian2.units import ms, um, meter, Hz
from brian2.groups import NeuronGroup
from brian2.synapses.synapses import Synapses

logger = logging.getLogger(__name__)


def quadraticBezierPath(posA, posB, nstep=8, controlDist=0.1, controlDim=[0, 1, 2]):
    segments = []

    vecP1 = posB - posA
    if np.linalg.norm(vecP1) == 0.0:
        raise Exception('Same start and end vertices!')

    vecP2 = np.array([0, 0, 0])
    vecP2[controlDim[2]] = 1.0
    vecC = np.cross(vecP1, vecP2)
    if np.linalg.norm(vecC) == 0.0:
        # The vectors are colinear so change the reference plane
        vecP2 = np.array([0, 0, 0])
        vecP2[controlDim[1]] = 1.0
    vecC = np.cross(vecP1, vecP2)
    vecC = vecC / np.linalg.norm(vecC) * controlDist

    posD = posA + (posB - posA) / 2.0
    posC = posD + vecC

    for t in np.linspace(0, 1.0, nstep):
        pos = posA * (1 - t) ** 2 + posB * t ** 2 + posC * 2 * t * (1 - t)
        segments.append(pos)
    return segments


def createNeuronGroup(N, refractory=5 * ms, tau=50 * ms, vti=0.1, srate=0.0 * Hz):

    # Integrate-and-fire neuron with adaptive threshold
    eqs = '''
        v : 1
        dvt/dt = -(vt-vt0)/tau : 1
        tau : second
        vt0 : 1
        vti : 1
        srate : Hz

        ct : 1
        cbf : 1

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

    if srate > 0.0:
        threshold = 'v >= vt or rand() < srate * dt'
    else:
        threshold = 'v >= vt'

    G = NeuronGroup(N, model=eqs, reset=reset, threshold=threshold, refractory=refractory, method='euler')

    # Configure neuron state variables initial values
    G.v = 0.0
    G.vt = 1.0
    G.tau = tau
    G.vt0 = 1.0
    G.vti = vti
    G.srate = srate
    G.ct = 0.0
    G.cbf = 0.0

    return G


def createCriticalSynapses(G, delay=3.0 * ms):

    # Configure synaptic connections
    eqs = '''
    c : 1    # accumulator for the contribution to post-synaptic spikes
    w : 1    # synaptic contribution per spike
    cbft : 1 # target critical branching factor
    lr : 1   # learning rate
    '''

    # NOTE: only consider the contribution if not in the refractory period
    on_pre = '''
    v_post += w  * ntype_pre * int(not_refractory_post)                  # add contribution of the presynaptic spike to the dynamic of the post-synaptic neuron
    c += w * int(ntype_pre > 0) * int(not_refractory_post)               # accumulate the contributions of the presynaptic spikes going through this synapse
    w = clip(w + lr * (cbft - cbf_pre) * int(ntype_pre > 0), 0.0, 1.0)   # modulate weight based on the error on the target critical branching factor
    ct_pre = 0.0                                                         # reset the total accumulated contributions from the synapse
    '''

    on_post = '''
    ct_pre += (c / (vt_post))                    # add the true contributions of past presynaptic spikes
    cbf_pre = ct_pre                             # store the last critical branching factor
    c = 0.0                                      # reset accumulator in synapse
    '''

    S = Synapses(G, G, model=eqs, on_pre=on_pre, on_post=on_post, delay=delay)

#     if forceSpiking:
#         # Increase the weight if no postsynaptic firing occured in a given interval
#         eqs = '''
#         w = clip(w + lr * int(cbf_pre == 0.0), 0.0, 1.0)
#         '''
#         S.run_regularly(eqs, dt=500 * ms)

    return S


class Microcircuit(object):
    '''
    Microcircuit
    '''

    def __init__(self, connectivity='small-world', macrocolumnShape=[2, 2, 2], minicolumnShape=[4, 4, 4], minicolumnSpacing=100 * um, neuronSpacing=10 * um, p_max=0.1):

        self.__dict__.update(macrocolumnShape=macrocolumnShape, minicolumnShape=minicolumnShape,
                             minicolumnSpacing=minicolumnSpacing, neuronSpacing=neuronSpacing)

        # Create the microcircuit
        logger.debug('Creating the microcircuit...')
        self.nbMinicolumns = np.prod(self.macrocolumnShape)
        self.nbNeuronsMinicolumn = np.prod(self.minicolumnShape)
        self.nbNeuronsTotal = self.nbNeuronsMinicolumn * self.nbMinicolumns

        self.G = createNeuronGroup(self.nbNeuronsTotal, vti=0.1, tau=450 * ms, srate=0.0 * Hz)

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

        logger.debug('Creating network topology ''%s'' ...' % (connectivity))
        if connectivity == 'small-world':
            # Connections inside minicolums
            self.S.connect(condition='i != j and mmidx_pre == mmidx_post',
                           p='p_max*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2) / (3*(125*umeter)**2))',
                           namespace={'p_max': p_max})

            # Connections across minicolums
            self.S.connect(condition='i != j and mmidx_pre != mmidx_post',
                           p='p_max*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2) / (3*(125*umeter)**2))',
                           namespace={'p_max': p_max})

            wmin = 0.0
            wmax = 0.25
            self.S.w = np.random.uniform(wmin, wmax, size=len(self.S.w))

        elif connectivity == 'random':
            # Connections inside minicolums
            self.S.connect(condition='i != j',
                           p='p_max*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2) / (3*(125*umeter)**2))',
                           namespace={'p_max': p_max})

            wmin = 0.0
            wmax = 0.25
            self.S.w = np.random.uniform(wmin, wmax, size=len(self.S.w))

        else:
            raise Exception('Unsupported connectivity: %s' % (connectivity))

        self.S.c = 0.0
        self.S.cbft = 1.0
        self.S.lr = 0.0

    def printConnectivityStats(self):

        logger.debug('Calculating microcircuit connectivity statistics...')

        # Number of isolated neurons and weakly-connected neurons (doesn't include isolated neurons)
        weaklyConnectedThreshold = 2
        nbOutputConnections = np.zeros((self.nbNeuronsTotal,), dtype=np.int)
        nbInputConnections = np.zeros((self.nbNeuronsTotal,), dtype=np.int)
        for i, j in zip(self.S.i, self.S.j):
            nbOutputConnections[i] += 1
            nbInputConnections[j] += 1
        nbIsolatedNeurons = np.count_nonzero(np.logical_or(nbOutputConnections == 0, nbInputConnections == 0))
        nbWeaklyConnectedNeurons = np.count_nonzero(np.logical_and(nbInputConnections > 0, nbInputConnections < weaklyConnectedThreshold))

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

    def draw3D(self, showAxisLabels=True):
        logger.debug('Drawing 3D microcircuit overview...')

        fig = plt.figure(facecolor='white', figsize=(6, 5))
        ax = axes3d.Axes3D(fig)

        # Calculate mean minimum distance between neurons
        positions = np.stack([self.G[:].x, self.G[:].y, self.G[:].z], axis=-1)
        D = np.linalg.norm(positions - positions[:, np.newaxis], ord=2, axis=-1)
        D[D == 0.0] = np.inf
        neuronSpacing = np.mean(np.min(D, axis=-1))
        logger.debug('Estimated mean neural spacing is %f m' % (neuronSpacing))

        # Draw each neuron
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'o')

        # Draw each synaptic connections
        for i, j in zip(self.S.i, self.S.j):
            posA = np.array([self.G.x[i], self.G.y[i], self.G.z[i]])
            posB = np.array([self.G.x[j], self.G.y[j], self.G.z[j]])

            # Generate a synaptic link between the neurons with a Bezier curve
            curvePoints = quadraticBezierPath(posA, posB, nstep=8, controlDist=float(0.5 * neuronSpacing))
            segments = [(curvePoints[s], curvePoints[s + 1]) for s in range(len(curvePoints) - 1)]

            # Use different colors for intracolumnar, intercolumnar and inhibitory connections
            if np.array_equal(self.G.mmidx[i], self.G.mmidx[j]):
                # Same minicolumn
                if self.G.ntype[i] > 0:
                    # Excitatory
                    color = 'gray'
                else:
                    # Inhibitory
                    color = 'dodgerblue'
            else:
                # Different minicolumn
                if self.G.ntype[i] > 0:
                    # Excitatory
                    color = 'firebrick'
                else:
                    # Inhibitory
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
