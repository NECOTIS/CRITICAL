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
# Date: April 18th, 2019
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


def createNeuronGroup(N, refractory=5 * ms, tau_vt=50 * ms, vti=0.1, srate=0.0 * Hz):

    # Leaky integrate-and-fire neuron with adaptive threshold
    eqs = '''
        dv/dt = -(v-v0)/tau: 1  (unless refractory) # membrane potential
        dvt/dt = -(vt-vt0)/tau_vt : 1               # adaptive threshold
        tau : second                          # time constant for membrane potential
        tau_vt : second                       # time constant for adaptive threshold
        v0 : 1                                # membrane potential reset value
        vt0 : 1                               # adaptive threshold reset value
        vti : 1                               # adaptive threshold increment
        srate : Hz                            # spontaneous rate

        cbf : 1               # last estimation of the target branching factor
        c_in_tot : 1          # estimation of total input contributions
        c_out_tot : 1         # estimation of total output contributions
        c_out_ref : 1         # target output contribution

        x : meter             # position of the neuron
        y : meter
        z : meter

        midx : 1        # macrocolumn index inside the microcircuit
        mmidx : 1       # minicolumn index inside the microcircuit

        ntype : 1       # type of the neuron (excitatory/inhibitory)
    '''

    reset = '''
    v = v0        # reset membrane potential
    vt += vti     # increment adaptive threshold
    '''

    # Spike detection
    if srate > 0.0:
        threshold = 'v > vt or rand() < srate * dt'
    else:
        threshold = 'v > vt'

    G = NeuronGroup(N, model=eqs, reset=reset, threshold=threshold,
                    refractory=refractory, method='exact')

    # Configure neuron state variables initial values
    G.v = 0.0
    G.tau = '15*ms + 10*ms * rand()'
    G.tau_vt = tau_vt
    G.vt0 = '1.0 + 1.0 * rand()'
    G.v0 = '0.5 + 0.5 * rand()'
    G.vt[:] = G.vt0[:]
    G.vti = vti
    G.srate = srate

    # Configure initial state for learning rule
    G.cbf = 0.0
    G.c_in_tot = 0.0
    G.c_out_tot = 0.0
    G.c_out_ref = 1.0

    G.ntype = 1.0  # Excitatory neuron

    return G


def createCriticalSynapses(G, delay=0.0 * ms):

    # Configure synaptic connections
    eqs = '''
    w : 1                                # synaptic contribution per spike
    alpha : 1                            # learning rate
    '''

    # From: Brodeur, S., & Rouat, J. (2012). Regulation toward Self-organized Criticality in a Recurrent Spiking Neural Reservoir.
    #       In A. E. Villa, W. Duch, P. Érdi, F. Masulli, & G. Palm (Eds.), Artificial Neural Networks and Machine Learning -- ICANN 2012 (Vol. 7552, pp. 547–554). Springer.
    # see Algorithm 1: Local unsupervised learning rule for self-organized
    # criticality
    on_pre = {  # Step 1: Update estimations of local input contributions
                'pre_transmission': '''
                v_post += w  * ntype_pre * int(not_refractory_post)                                                # Add contribution of the presynaptic spike to the dynamic of the post-synaptic neuron
                c_in_tot_post += w * ntype_pre * int(ntype_pre > 0) * int(not_refractory_post)                     # Update estimations of local input contributions to postsynaptic neurons
                ''',
                # Step 3 and 4: Calculate the error on the target contribution, and update postsynaptic weights to reduce the error
                'pre_plasticity': '''
                cbf_pre = c_out_tot_pre                                                                                        # Store current estimate of the target branching factor
                e = (c_out_ref_pre - c_out_tot_pre)                                                                            # Calculate the error on the target contribution
                w = clip(w + alpha * (e / N_outgoing) * exp(-(t - lastspike_post)/tau_pre) * int(ntype_pre > 0), 1e-4, 1.0)    # Update postsynaptic weights to reduce the error, with ponderation scheme to favor recently active postsynaptic neurons
                ''',
                # Step 5: Reset state variables to accumulate contributions for another interspike interval
                'pre_reset': '''
                c_in_tot_pre = 0.0
                c_out_tot_pre = 0.0
              '''
    }

    on_post = {
        # Step 2: Update estimations of local output contributions
        # NOTE: The case c_in_tot_post = 0 can happen if spontaneous activity elicited a postsynaptic spike.
        # In that case, c_out_tot_pre will be zero (no contribution).
        'post_feedback': '''
                eps = 1e-10
                c_out_tot_pre += (w * int(ntype_pre > 0) * int(c_in_tot_post > 0) / (c_in_tot_post + eps)) * int(lastspike_pre > 0.0 * ms) * exp(-(t - lastspike_pre)/ tau_post)               # Update estimations of local output contributions in presynaptic neurons, with ponderation scheme to favor recently active presynaptic neurons.
                ''',
        # Step 5: Reset state variables to accumulate contributions for another interspike interval
        # NOTE: we need this in case the postsynaptic neuron has no outgoing connections
        'post_reset': '''
                c_in_tot_post = 0.0
               '''
    }

    S = Synapses(G, G, model=eqs, on_pre=on_pre, on_post=on_post,
                 delay={'pre_transmission': delay})

    # NOTE: the execution order is important
    S.pre_transmission.order = 1
    S.post_feedback.order = 2
    S.pre_plasticity.order = 3
    S.pre_reset.order = 4
    S.post_reset.order = 5

    return S


class Microcircuit(object):
    '''
    Microcircuit
    '''

    def __init__(self, connectivity='small-world',
                 macrocolumnShape=[2, 2, 2], minicolumnShape=[4, 4, 4], minicolumnSpacing=100 * um,
                 neuronSpacing=10 * um, p_max=0.1, srate=0.0 * Hz, excitatoryProb=0.8, delay=0.0 * ms):

        self.__dict__.update(macrocolumnShape=macrocolumnShape, minicolumnShape=minicolumnShape,
                             minicolumnSpacing=minicolumnSpacing, neuronSpacing=neuronSpacing)

        # Create the microcircuit
        logger.debug('Creating the microcircuit...')
        self.nbMinicolumns = np.prod(self.macrocolumnShape)
        self.nbNeuronsMinicolumn = np.prod(self.minicolumnShape)
        self.nbNeuronsTotal = self.nbNeuronsMinicolumn * self.nbMinicolumns

        self.G = createNeuronGroup(self.nbNeuronsTotal, vti=0.1, srate=srate)

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
                                xn = float(
                                    im * (self.minicolumnShape[0]) + i) * self.neuronSpacing
                                yn = float(
                                    jm * (self.minicolumnShape[1]) + j) * self.neuronSpacing
                                zn = float(
                                    km * (self.minicolumnShape[2]) + k) * self.neuronSpacing
                                if connectivity == 'small-world':
                                    xn += im * minicolumnSpacing
                                    yn += jm * minicolumnSpacing
                                    zn += km * minicolumnSpacing
                                positions.append([xn, yn, zn])
                                macrocolumIndices.append(
                                    im * self.macrocolumnShape[0] + jm * self.macrocolumnShape[1] + km)
                                minicolumIndices.append(
                                    i * self.minicolumnShape[0] + j * self.minicolumnShape[1] + k)
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
        neuronTypes = np.random.random(self.nbNeuronsTotal)
        ntypes = np.zeros((self.nbNeuronsTotal,), dtype=np.float32)
        ntypes[np.where(neuronTypes < excitatoryProb)] = 1.0
        ntypes[np.where(neuronTypes >= excitatoryProb)] = -1.0
        self.G.ntype = ntypes

        self.S = createCriticalSynapses(self.G, delay)

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

        elif connectivity == 'random':
            # Connections inside minicolums
            self.S.connect(condition='i != j',
                           p='p_max*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2) / (3*(125*umeter)**2))',
                           namespace={'p_max': p_max})
        else:
            raise Exception('Unsupported connectivity: %s' % (connectivity))

        # Configure initial weights [0.25, 0.50]
        self.S.w = '0.25 + 0.25 * rand()'

        # Configure learning rule parameters
        self.S.alpha = 0.1

    def printConnectivityStats(self):

        logger.debug('Calculating microcircuit connectivity statistics...')

        # Number of isolated neurons and weakly-connected neurons (doesn't
        # include isolated neurons)
        weaklyConnectedThreshold = 2
        nbOutputConnections = np.zeros((self.nbNeuronsTotal,), dtype=np.int)
        nbInputConnections = np.zeros((self.nbNeuronsTotal,), dtype=np.int)
        for i, j in zip(self.S.i, self.S.j):
            nbOutputConnections[i] += 1
            nbInputConnections[j] += 1
        nbIsolatedNeurons = np.count_nonzero(np.logical_or(
            nbOutputConnections == 0, nbInputConnections == 0))
        nbWeaklyConnectedNeurons = np.count_nonzero(np.logical_and(
            nbInputConnections > 0, nbInputConnections < weaklyConnectedThreshold))

        # Average number of synapses/neuron
        avgSynapsesPerNeuron = float(len(self.S)) / self.nbNeuronsTotal

        # Total number of synapses
        nbSynapsesTotal = len(self.S)

        # Neuron types
        ratioExcitatory = float(
            len(np.where(self.G.ntype > 0)[0])) / self.nbNeuronsTotal
        ratioInhibitory = float(
            len(np.where(self.G.ntype < 0)[0])) / self.nbNeuronsTotal

        logger.info(
            '############## MICROCIRCUIT CONNECTIVITY STATISTICS ###############')
        logger.info('Macrocolumn shape: (%d x %d x %d)' % (
            self.macrocolumnShape[0], self.macrocolumnShape[1], self.macrocolumnShape[2]))
        logger.info('Number of minicolumns: %d' % (self.nbMinicolumns))
        logger.info('Minicolumn shape: (%d x %d x %d)' % (
            self.minicolumnShape[0], self.minicolumnShape[1], self.minicolumnShape[2]))
        logger.info('Number of neurons per minicolumn: %d' %
                    (self.nbNeuronsMinicolumn))
        logger.info(
            '-------------------------------------------------------------------')
        logger.info('Total number of neurons: %d' % (self.nbNeuronsTotal))
        logger.info('Neuron type ratio: excitatory (%3.2f%%), inhibitory(%3.2f%%)' % (
            ratioExcitatory * 100, ratioInhibitory * 100))
        logger.info('Total number of isolated neurons: %d (%3.2f%%)' % (
            nbIsolatedNeurons, float(nbIsolatedNeurons) / self.nbNeuronsTotal * 100.0))
        logger.info('Total number of weakly-connected neurons (< %d input synapses): %d (%3.2f%%)' %
                    (weaklyConnectedThreshold, nbWeaklyConnectedNeurons, float(nbWeaklyConnectedNeurons) / self.nbNeuronsTotal * 100.0))

        logger.info('Average number of synapses/neuron: %1.2f' %
                    (avgSynapsesPerNeuron))
        logger.info('Total number of synapses: %d' % (nbSynapsesTotal))
        logger.info(
            '###################################################################')

    def draw3D(self, showAxisLabels=True):
        logger.debug('Drawing 3D microcircuit overview...')

        fig = plt.figure(facecolor='white', figsize=(6, 5))
        ax = axes3d.Axes3D(fig)

        # Calculate mean minimum distance between neurons
        positions = np.stack([self.G[:].x, self.G[:].y, self.G[:].z], axis=-1)
        D = np.linalg.norm(
            positions - positions[:, np.newaxis], ord=2, axis=-1)
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
            curvePoints = quadraticBezierPath(
                posA, posB, nstep=8, controlDist=float(0.5 * neuronSpacing))
            segments = [(curvePoints[s], curvePoints[s + 1])
                        for s in range(len(curvePoints) - 1)]

            # Use different colors for intracolumnar, intercolumnar and
            # inhibitory connections
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
