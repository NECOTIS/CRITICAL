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

import random
import logging
import numpy as np
import matplotlib.pyplot as plt

from brian2.units import ms, Hz
from brian2.synapses.synapses import Synapses
from brian2.input.poissongroup import PoissonGroup
from brian2.core.clocks import defaultclock
from brian2.monitors.spikemonitor import SpikeMonitor
from brian2.core.network import Network
from brian2.units.allunits import second
from brian2.monitors.statemonitor import StateMonitor

from critical.microcircuit import Microcircuit

logger = logging.getLogger(__name__)


def main():

    # Choose the duration of the training
    duration = 10 * second

    fig = plt.figure(facecolor='white', figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Average output contributions')
    ax.grid(True)

    targetCbfs = [0.6, 0.8, 1.0, 1.2, 1.4]
    linestyleList = ['-', '--', '-.', ':', '-']
    linewidthList = [1, 1, 1, 2, 1]
    colorList = ['grey', 'black', 'black', 'black', 'black']
    for n in range(len(targetCbfs)):
        logger.info('Simulating for target branching factor of %f' % (targetCbfs[n]))

        # Create the microcircuit
        # NOTE: p_max is chosen so to have an out-degree of N=16
        structure = 'random'
        m = Microcircuit(connectivity=structure, macrocolumnShape=[2, 2, 2], minicolumnShape=[4, 4, 4],
                         p_max=0.056, srate=0 * Hz, excitatoryProb=0.8, delay=0.0 * ms)

        # Configure CRITICAL learning rule
        m.S.c_out_ref = targetCbfs[n]      # target critical branching factor
        m.S.alpha = 0.1                    # learning rate

        # Define the inputs to the microcircuit
        # NOTE: Number of average input synaptic connections is fixed to 10% of reservoir links
        nbInputs = 64
        P = PoissonGroup(nbInputs, rates=np.linspace(25 * Hz, 50 * Hz, nbInputs))
        Si = Synapses(P, m.G, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
                                                       c_in_tot_post += w * int(not_refractory_post)''')
        Si.connect(p=0.1 * len(m.S) / (nbInputs * len(m.G)))
        Si.w = '0.5 + 1.5 * rand()'

        logger.info('Number of neurons in the population: %d' % (len(m.G)))
        logger.info('Number of synapses in the population: %d' % (len(m.S)))

        # Configure the monitors and simulation
        # NOTE: setting a high time resolution increase the stability of the learning rule
        M = SpikeMonitor(m.G, record=True)
        Mi = SpikeMonitor(P, record=True)
        Mg = StateMonitor(m.G, variables=['cbf'], record=True)
        defaultclock.dt = 0.1 * ms
        net = Network(m.G, m.S, P, Si, M, Mi, Mg)

        # Run the simulation with input stimuli and plasticity enabled
        m.S.plastic = True
        net.run(duration, report='text')

        # NOTE: compute statistics on excitatory neurons only
        meanCbf = np.mean(Mg.cbf.T[:, m.G.ntype > 0], axis=-1)
        ax.plot(Mg.t, meanCbf,
                color=colorList[n], linestyle=linestyleList[n], linewidth=linewidthList[n], label='target = %1.1f' % (targetCbfs[n]))

        # Compute population average firing rate
        avgInputFiringRate = len(Mi.i) / (nbInputs * duration)
        avgOutputFiringRate = len(M.i) / (len(m.G) * duration)
        logger.info('Average input firing rate: %4.2f Hz' % (avgInputFiringRate))
        logger.info('Average output firing rate: %4.2f Hz' % (avgOutputFiringRate))

#         # Visualization of the simulation
#         fig2 = plt.figure(facecolor='white', figsize=(6, 5))
#         ax2 = fig.add_subplot(1, 1, 1)
#         plt.subplot(211)
#         plt.title('Spiking activity (input)')
#         plt.plot(Mi.t / ms, Mi.i, '.', color='b')
#         plt.ylabel('Neurons')
#         plt.xlabel('Time [ms]')
#         plt.xlim([0.0, duration / ms])
#
#         plt.subplot(212)
#         plt.title('Spiking activity (output)')
#         plt.plot(M.t / ms, M.i, '.', color='b')
#         plt.ylabel('Neurons')
#         plt.xlabel('Time [ms]')
#         plt.xlim([0.0, duration / ms])
#         plt.show()

    ax.set_ylim([0.0, 2.0])

    # Visualization
    ax.legend(loc='lower right', ncol=2)
    fig.canvas.draw()
    fig.savefig('convergence_poisson_%s.eps' % (structure))

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Fix the seed of all random number generator
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    main()
    logger.info('All done.')
