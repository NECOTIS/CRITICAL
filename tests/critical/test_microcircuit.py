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
import random
import unittest
import numpy as np
import matplotlib.pyplot as plt

from brian2.units.stdunits import ms, Hz
from brian2.units.allunits import second
from brian2.input.poissongroup import PoissonGroup
from brian2.synapses.synapses import Synapses
from brian2.monitors.statemonitor import StateMonitor
from brian2.core.clocks import defaultclock
from brian2.core.network import Network, scheduling_summary
from brian2.monitors.spikemonitor import SpikeMonitor

from critical.microcircuit import Microcircuit, createNeuronGroup, createCriticalSynapses

from brian2.core.preferences import prefs
prefs.codegen.target = 'numpy'  # use the Python fallback

logger = logging.getLogger(__name__)


class TestMicrocircuit(unittest.TestCase):

    def test_init(self):

        for connectivity in ['small-world', 'random']:
            microcircuit = Microcircuit(
                connectivity, minicolumnShape=[2, 2, 2])
            microcircuit.printConnectivityStats()
            fig = microcircuit.draw3D(showAxisLabels=True)

            plt.ion()
            plt.show()
            plt.draw()
            plt.pause(1.0)
            plt.close(fig)

    def test_neural_dynamic(self):
        G = createNeuronGroup(N=1, refractory=15 * ms, tau=50 * ms, vti=0.1)

        # Input to the network
        P = PoissonGroup(1, 40 * Hz)
        S = Synapses(P, G, on_pre='v += 0.2')
        S.connect()

        M = StateMonitor(G, variables=True, record=True)
        defaultclock.dt = 0.5 * ms
        net = Network(G, S, P, M)
        net.run(1400 * ms)

        fig = plt.figure()
        plt.subplot(311)
        plt.plot(M.t / ms, M.v.T)
        plt.ylabel('V')
        plt.xlabel('Time [ms]')

        plt.subplot(312)
        plt.plot(M.t / ms, M.vt.T)
        plt.ylabel('Threshold')
        plt.xlabel('Time [ms]')

        plt.subplot(313)
        plt.plot(M.t / ms, 1.0 - M.not_refractory.T)
        plt.ylabel('Refractoriness')
        plt.yticks([0.0, 1.0])
        plt.xlabel('Time [ms]')

        plt.tight_layout()

        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(100.0)
        plt.close(fig)

    def test_synapse_dynamic_single(self):
        G = createNeuronGroup(N=2)
        G.c_out_ref = 1.0

        S = createCriticalSynapses(G)
        S.connect(i=0, j=1)
        S.w = 0.5
        S.alpha = 0.1

        # Input to the network
        P = PoissonGroup(1, 40 * Hz)
        Si = Synapses(P, G, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
                                                     c_in_tot_post += w * int(not_refractory_post)''')
        Si.connect(i=0, j=0)
        Si.w = 1.0

        M = SpikeMonitor(G)
        Mg = StateMonitor(G, variables=True, record=True,
                          when='synapses', order=4)
        Ms = StateMonitor(S, variables=True, record=True,
                          when='synapses', order=4)

        defaultclock.dt = 1 * ms
        net = Network(G, S, P, Si, M, Ms, Mg)
        logger.info(scheduling_summary(net))

        duration = 10 * second
        net.run(duration)

        plt.figure()
        plt.subplot(221)
        plt.plot(M.t / ms, M.i, '.')
        plt.ylabel('Neurons')
        plt.yticks([0, 1], ['pre', 'post'])
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.ylim([-0.1, 1.1])

        plt.subplot(222)
        plt.plot(Mg.t / ms, Mg[0].v.T, label='pre')
        plt.plot(Mg.t / ms, Mg[1].v.T, label='post')
        plt.ylabel('v')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.subplot(223)
        plt.plot(Ms.t / ms, Ms.w.T)
        plt.ylabel('w')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])

        plt.subplot(224)
        plt.plot(Mg.t / ms, Mg[0].cbf.T, label='pre')
        plt.ylabel('cbf')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.tight_layout()

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(Mg.t / ms, Mg[1].c_in_tot.T, label='post')
        plt.ylabel('c_in_tot')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.subplot(212)
        plt.plot(Mg.t / ms, Mg[0].c_out_tot.T, label='pre')
        plt.ylabel('c_out_tot')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.tight_layout()

        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(100.0)
        plt.close(fig)

    def test_synapse_dynamic_dual(self):

        G = createNeuronGroup(N=3)
        G.c_out_ref = 1.0

        S = createCriticalSynapses(G)
        S.connect(i=[0, 1], j=[2, 2])
        S.w = 0.5
        S.alpha = 0.1

        # Input to the network
        P = PoissonGroup(2, [20 * Hz, 40 * Hz])
        Si = Synapses(P, G, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
                                                     c_in_tot_post += w * int(not_refractory_post)''')
        Si.connect(i=[0, 1], j=[0, 1])
        Si.w = 0.5

        M = SpikeMonitor(G)
        Mg = StateMonitor(G, variables=True, record=True,
                          when='synapses', order=4)
        Ms = StateMonitor(S, variables=True, record=True,
                          when='synapses', order=4)

        defaultclock.dt = 1 * ms
        net = Network(G, S, P, Si, M, Ms, Mg)
        logger.info(scheduling_summary(net))

        duration = 30 * second
        net.run(duration)

        plt.figure()
        plt.subplot(221)
        plt.plot(M.t / ms, M.i, '.')
        plt.ylabel('Neurons')
        plt.yticks([0, 1, 2], ['pre1', 'pre2', 'post'])
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.ylim([-0.1, 2.1])

        plt.subplot(222)
        plt.plot(Mg.t / ms, Mg[0].v.T, label='pre1')
        plt.plot(Mg.t / ms, Mg[1].v.T, label='pre2')
        plt.plot(Mg.t / ms, Mg[2].v.T, label='post')
        plt.ylabel('v')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.subplot(223)
        plt.plot(Ms.t / ms, Ms.w.T)
        plt.ylabel('w')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])

        plt.subplot(224)
        plt.plot(Mg.t / ms, Mg[0].cbf.T, label='pre1')
        plt.plot(Mg.t / ms, Mg[1].cbf.T, label='pre2')
        plt.ylabel('cbf')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.tight_layout()

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(Mg.t / ms, Mg[2].c_in_tot.T, label='post')
        plt.ylabel('c_in_tot')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.subplot(212)
        plt.plot(Mg.t / ms, Mg[0].c_out_tot.T, label='pre1')
        plt.plot(Mg.t / ms, Mg[1].c_out_tot.T, label='pre2')
        plt.ylabel('c_out_tot')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.tight_layout()

        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(100.0)
        plt.close(fig)

    def test_synapse_dynamic_multi(self):

        G = createNeuronGroup(N=4)
        G.c_out_ref = 1.0

        S = createCriticalSynapses(G)
        S.connect(i=[0, 1, 0, 1], j=[2, 2, 3, 3])
        S.w = 0.5
        S.alpha = 0.1

        # Input to the network
        P = PoissonGroup(2, [40 * Hz, 40 * Hz])
        Si = Synapses(P, G, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
                                                     c_in_tot_post += w * int(not_refractory_post)''')
        Si.connect(i=[0, 1], j=[0, 1])
        Si.w = 0.5

        M = SpikeMonitor(G)
        Mg = StateMonitor(G, variables=True, record=True,
                          when='synapses', order=4)
        Ms = StateMonitor(S, variables=True, record=True,
                          when='synapses', order=4)

        defaultclock.dt = 1 * ms
        net = Network(G, S, P, Si, M, Ms, Mg)
        logger.info(scheduling_summary(net))

        duration = 30 * second
        net.run(duration)

        plt.figure()
        plt.subplot(221)
        plt.plot(M.t / ms, M.i, '.')
        plt.ylabel('Neurons')
        plt.yticks([0, 1, 2, 3], ['pre1', 'pre2', 'post1', 'post2'])
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.ylim([-0.1, 3.1])

        plt.subplot(222)
        plt.plot(Mg.t / ms, Mg[0].v.T, label='pre1')
        plt.plot(Mg.t / ms, Mg[1].v.T, label='pre2')
        plt.plot(Mg.t / ms, Mg[2].v.T, label='post1')
        plt.plot(Mg.t / ms, Mg[3].v.T, label='post2')
        plt.ylabel('v')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.subplot(223)
        plt.plot(Ms.t / ms, Ms.w.T)
        plt.ylabel('w')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])

        plt.subplot(224)
        plt.plot(Mg.t / ms, Mg[0].cbf.T, label='pre1')
        plt.plot(Mg.t / ms, Mg[1].cbf.T, label='pre2')
        plt.ylabel('cbf')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.tight_layout()

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(Mg.t / ms, Mg[2].c_in_tot.T, label='post1')
        plt.plot(Mg.t / ms, Mg[3].c_in_tot.T, label='post2')
        plt.ylabel('c_in_tot')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.subplot(212)
        plt.plot(Mg.t / ms, Mg[0].c_out_tot.T, label='pre1')
        plt.plot(Mg.t / ms, Mg[1].c_out_tot.T, label='pre2')
        plt.ylabel('c_out_tot')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.tight_layout()

        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(100.0)
        plt.close(fig)

    def test_synapse_dynamic_complex(self):
        # NOTE: spontaneous activity is necessary if the weights are small at the beginning of the simulation.
        # Otherwise, the learning rule has no opportunity to update the
        # weights.
        G = createNeuronGroup(N=64, srate=1 * Hz)
        G.c_out_ref = 0.5

        S = createCriticalSynapses(G)
        S.connect(condition='i != j', p=0.1)
        S.w = 0.1 + 0.5 * np.random.uniform(size=len(S))
        S.alpha = 0.1

        # Input to the network
        nbInputs = 8
        P = PoissonGroup(nbInputs, np.random.uniform(
            low=20.0, high=50.0, size=(nbInputs,)) * Hz)
        Si = Synapses(P, G, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
                                                     c_in_tot_post += w * int(not_refractory_post)''')
        Si.connect(i=np.arange(nbInputs), j=np.random.permutation(
            np.arange(len(G)))[:nbInputs])
        Si.w = 0.5

        M = SpikeMonitor(G)
        Mg = StateMonitor(G, variables=True, record=True,
                          when='synapses', order=4)
        Ms = StateMonitor(S, variables=True, record=True,
                          when='synapses', order=4)

        defaultclock.dt = 1 * ms
        net = Network(G, S, P, Si, M, Ms, Mg)
        logger.info(scheduling_summary(net))

        duration = 120 * second
        net.run(duration)

        plt.figure()
        plt.subplot(211)
        plt.plot(M.t / ms, M.i, '.')
        plt.ylabel('Neurons')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])

        plt.subplot(212)
        plt.plot(Ms.t / ms, Ms.w.T)
        plt.ylabel('Weight')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.tight_layout()

        fig = plt.figure()
        meanCbf = np.mean(Mg.cbf.T, axis=-1)
        stdCbf = np.std(Mg.cbf.T, axis=-1)
        plt.plot(Mg.t / ms, meanCbf, color='#1B2ACC')
        plt.fill_between(Mg.t / ms, meanCbf - stdCbf, meanCbf + stdCbf,
                         alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)
        plt.ylabel('cbf')
        plt.xlabel('Time [ms]')

        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(100.0)
        plt.close(fig)

    def test_microcircuit(self):

        # Create the microcircuit
        # NOTE: p_max is chosen so to have an out-degree of N=16
        m = Microcircuit(connectivity='small-world', macrocolumnShape=[
                         2, 2, 2], minicolumnShape=[4, 4, 4], p_max=0.056, srate=1 * Hz)
        m.printConnectivityStats()
        m.draw3D()

        # Configure CRITICAL learning rule
        m.S.c_out_ref = 0.5      # target critical branching factor
        m.S.alpha = 0.1           # learning rate

        # Define the inputs to the microcircuit
        # NOTE: Number of average input synaptic connections is fixed to 10% of reservoir links
        # FIXME: need to implement variable phase and base rate for each poisson process
        # see:
        # https://brian2.readthedocs.io/en/stable/user/input.html?highlight=PoissonGroup#more-on-poisson-inputs
        nbInputs = 64
        P = PoissonGroup(
            nbInputs, rates=np.linspace(25 * Hz, 50 * Hz, nbInputs))
        Si = Synapses(P, m.G, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
                                                       c_in_tot_post += w * int(not_refractory_post)''')
        Si.connect(p=0.1 * len(m.S) / (nbInputs * len(m.G)))
        Si.w = '0.5 + 1.5 * rand()'

        # Configure the monitors and simulation
        M = SpikeMonitor(m.G, record=True)
        Mi = SpikeMonitor(P, record=True)
        Mg = StateMonitor(m.G, variables=['cbf'], record=True)
        Ms = StateMonitor(m.S, variables=['w'], record=True, dt=100 * ms)
        defaultclock.dt = 1 * ms
        net = Network(m.G, m.S, P, Si, M, Mi, Mg, Ms)
        logger.info(scheduling_summary(net))

        # Run the simulation with input stimuli enabled
        durationOn = 120 * second
        net.run(durationOn, report='text')

        # Continue the simulation with input stimuli disable
        durationOff = 10 * second
        P.active = False
        Si.active = False
        net.run(durationOff, report='text')

        # Compute total duration of the simulation
        duration = durationOn + durationOff

        # Visualization of the simulation
        plt.figure()
        plt.subplot(311)
        plt.title('Spiking activity (input)')
        plt.plot(Mi.t / ms, Mi.i, '.', color='b')
        plt.ylabel('Neurons')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])

        plt.subplot(312)
        plt.title('Spiking activity (output)')
        plt.plot(M.t / ms, M.i, '.', color='b')
        plt.ylabel('Neurons')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])

        plt.subplot(313)
        plt.title('Critical branching factor')
        averagingWindowLen = 200 * ms
        windowLen = int(averagingWindowLen / defaultclock.dt)
        cbf = np.stack([np.convolve(cbf, 1 / windowLen * np.ones((windowLen,),
                                                                 dtype=np.float), mode='same') for cbf in Mg.cbf])
        meanCbf = np.mean(cbf.T, axis=-1)
        stdCbf = np.std(cbf.T, axis=-1)
        plt.plot(Mg.t / ms, meanCbf, color='#1B2ACC')
        plt.fill_between(Mg.t / ms, meanCbf - stdCbf, meanCbf + stdCbf,
                         alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)
        plt.ylabel('cbf')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.tight_layout()

        plt.figure()
        plt.title('Synaptic weights')
        meanW = np.mean(Ms.w.T, axis=-1)
        stdW = np.std(Ms.w.T, axis=-1)
        plt.plot(Ms.t / ms, meanW, color='#1B2ACC')
        plt.fill_between(Ms.t / ms, meanW - stdW, meanW + stdW,
                         alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)
        plt.ylabel('Strength')
        plt.xlabel('Time [ms]')

        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(100.0)
        plt.close('all')


if __name__ == '__main__':

    # Fix the seed of all random number generator
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

#     suite = unittest.TestSuite()
#     suite.addTest(TestMicrocircuit("test_synapse_dynamic_dual"))
#     runner = unittest.TextTestRunner()
#     runner.run(suite)
