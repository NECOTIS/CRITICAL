# Copyright (c) 2018, Simon Brodeur
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

import logging
import unittest
import matplotlib.pyplot as plt

from brian2.units.stdunits import ms, Hz
from brian2.input.poissongroup import PoissonGroup
from brian2.synapses.synapses import Synapses
from brian2.monitors.statemonitor import StateMonitor
from brian2.core.clocks import defaultclock
from brian2.core.network import Network
from brian2.monitors.spikemonitor import SpikeMonitor

from critical.microcircuit import Microcircuit, createNeuronGroup, createCriticalSynapses
from critical.visualization import draw3D

from brian2.core.preferences import prefs
from brian2.units.allunits import second
prefs.codegen.target = 'numpy'  # use the Python fallback

logger = logging.getLogger(__name__)


class TestMicrocircuit(unittest.TestCase):

    def test_init(self):

        for connectivity in ['small-world', 'random']:
            microcircuit = Microcircuit(connectivity, minicolumnShape=[2, 2, 2])
            microcircuit.printConnectivityStats()
            fig = draw3D(microcircuit.G, microcircuit.S, showAxisLabels=True)

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
        plt.pause(1.0)
        plt.close(fig)

    def test_synapse_dynamic_single(self):
        G = createNeuronGroup(N=2, refractory=15 * ms, tau=50 * ms, vti=0.1)
        S = createCriticalSynapses(G)

        S.connect(i=0, j=1)
        S.w = 0.05
        S.cbf = 1.0
        S.lr = 0.01

        # Input to the network
        P = PoissonGroup(1, 40 * Hz)
        Si = Synapses(P, G, model='w : 1', on_pre='v_post += w')
        Si.connect(i=0, j=0)
        Si.w = 0.25

        M = SpikeMonitor(G)
        Mg = StateMonitor(G, variables=True, record=True)
        Ms = StateMonitor(S, variables=True, record=True)

        defaultclock.dt = 0.5 * ms
        net = Network(G, S, P, Si, M, Ms, Mg)

        duration = 10 * second
        net.run(duration)

        fig = plt.figure()
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
        plt.plot(Ms.t / ms, Mg[0].ulast.T)
        plt.ylabel('cbf')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])

        plt.tight_layout()

        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(1.0)
        plt.close(fig)

    def test_synapse_dynamic_dual(self):
        G = createNeuronGroup(N=3, refractory=15 * ms, tau=50 * ms, vti=0.1)
        S = createCriticalSynapses(G)

        S.connect(i=[0, 1], j=[2, 2])
        S.w = 0.05
        S.cbf = 0.5
        S.lr = 0.01

        # Input to the network
        P = PoissonGroup(2, [20 * Hz, 40 * Hz])
        Si = Synapses(P, G, model='w : 1', on_pre='v_post += w')
        Si.connect(i=[0, 1], j=[0, 1])
        Si.w = 0.25

        M = SpikeMonitor(G)
        Mg = StateMonitor(G, variables=True, record=True)
        Ms = StateMonitor(S, variables=True, record=True)

        defaultclock.dt = 0.5 * ms
        net = Network(G, S, P, Si, M, Ms, Mg)

        duration = 30 * second
        net.run(duration)

        fig = plt.figure()
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
        plt.plot(Ms.t / ms, Mg[0].ulast.T, label='pre1')
        plt.plot(Ms.t / ms, Mg[1].ulast.T, label='pre2')
        plt.ylabel('cbf')
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
        G = createNeuronGroup(N=4, refractory=15 * ms, tau=50 * ms, vti=0.1)
        S = createCriticalSynapses(G)

        S.connect(i=[0, 1, 0, 1], j=[2, 2, 3, 3])
        S.w = 0.05
        S.cbf = 1.0
        S.lr = 0.01

        # Input to the network
        P = PoissonGroup(2, [20 * Hz, 40 * Hz])
        Si = Synapses(P, G, model='w : 1', on_pre='v_post += w')
        Si.connect(i=[0, 1], j=[0, 1])
        Si.w = 0.25

        M = SpikeMonitor(G)
        Mg = StateMonitor(G, variables=True, record=True)
        Ms = StateMonitor(S, variables=True, record=True)

        defaultclock.dt = 0.5 * ms
        net = Network(G, S, P, Si, M, Ms, Mg)

        duration = 30 * second
        net.run(duration)

        fig = plt.figure()
        plt.subplot(221)
        plt.plot(M.t / ms, M.i, '.')
        plt.ylabel('Neurons')
        plt.yticks([0, 1, 2, 3], ['pre1', 'pre2', 'post1', 'post2'])
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.ylim([-0.1, 2.1])

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
        plt.plot(Ms.t / ms, Mg[0].ulast.T, label='pre1')
        plt.plot(Ms.t / ms, Mg[1].ulast.T, label='pre2')
        plt.ylabel('cbf')
        plt.xlabel('Time [ms]')
        plt.xlim([0.0, duration / ms])
        plt.legend()

        plt.tight_layout()

        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(100.0)
        plt.close(fig)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestMicrocircuit("test_synapse_dynamic_dual"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
