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

from brian2.core.preferences import prefs
prefs.codegen.target = 'numpy'  # use the Python fallback

logger = logging.getLogger(__name__)


def main():

    # Choose the duration of the training
    durationOn = 10 * second
    targetCbf = 1.0

    logger.info('Simulating for target branching factor of %f' % (targetCbf))

    # Create the microcircuit
    # NOTE: p_max is chosen so to have an out-degree of N=16
    structure = 'random'
    m = Microcircuit(connectivity=structure, macrocolumnShape=[2, 2, 2], minicolumnShape=[4, 4, 4],
                     p_max=0.056, srate=1 * Hz, excitatoryProb=1.0, delay=0.0 * ms)

    # Configure CRITICAL learning rule
    m.S.c_out_ref = targetCbf          # target critical branching factor
    m.S.alpha = 0.1                    # learning rate

    # Define the inputs to the microcircuit
    # NOTE: Number of average input synaptic connections is fixed to 10% of reservoir links
    nbInputs = 64
    P = PoissonGroup(nbInputs, rates=25 * Hz)
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
    net.run(durationOn, report='text')

    # Compute population average firing rate
    avgInputFiringRate = len(Mi.i) / (nbInputs * durationOn)
    avgOutputFiringRate = len(M.i) / (len(m.G) * durationOn)
    logger.info('Average input firing rate: %4.2f Hz' % (avgInputFiringRate))
    logger.info('Average output firing rate: %4.2f Hz' % (avgOutputFiringRate))

    # Continue the simulation with input stimuli and plasticity disabled
    durationOff = 20 * second
    P.active = False
    Si.active = False
    m.S.plastic = False
    net.run(durationOff, report='text')

    duration = durationOn + durationOff

    # Visualization of the simulation
    # NOTE: show only the last 10 sec of the simulation
    fig = plt.figure(facecolor='white', figsize=(6, 5))
    plt.subplot(211)
    plt.title('Spiking activity (input)')
    plt.plot(Mi.t / ms, Mi.i, '.', color='b')
    plt.ylabel('Neurons')
    plt.xlabel('Time [ms]')
    plt.xlim([(duration - 10 * second) / ms, duration / ms])

    plt.subplot(212)
    plt.title('Spiking activity (output)')
    plt.plot(M.t / ms, M.i, '.', color='b')
    plt.ylabel('Neurons')
    plt.xlabel('Time [ms]')
    plt.xlim([(duration - 10 * second) / ms, duration / ms])

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Fix the seed of all random number generator
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    main()
    logger.info('All done.')
