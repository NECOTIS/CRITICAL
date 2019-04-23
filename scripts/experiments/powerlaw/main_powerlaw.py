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
# Date: April 22th, 2019
# Organization: Groupe de recherche en Neurosciences Computationnelles et Traitement Intelligent des Signaux (NECOTIS),
# Université de Sherbrooke, Canada

import random
import logging
import numpy as np
import matplotlib.pyplot as plt

from brian2.units import ms, Hz
from brian2.synapses.synapses import Synapses
from brian2.core.clocks import defaultclock
from brian2.monitors.spikemonitor import SpikeMonitor
from brian2.core.network import Network
from brian2.units.allunits import second
from brian2.monitors.statemonitor import StateMonitor
from brian2.input.spikegeneratorgroup import SpikeGeneratorGroup
from brian2.groups.neurongroup import NeuronGroup
from brian2.core.functions import implementation
from brian2.units.fundamentalunits import check_units
from brian2.core.base import BrianObject
from brian2.devices.cpp_standalone.codeobject import CPPStandaloneCodeObject
from brian2.core.preferences import prefs

from critical.microcircuit import Microcircuit

logger = logging.getLogger(__name__)


class AvalancheEndDetector(BrianObject):

    def __init__(self, G, maxQuietTime=15 * ms, network=None, *args, **kwargs):
        super(AvalancheEndDetector, self).__init__(*args, **kwargs)

        prefs.codegen.cpp.headers += ['"run.h"']  # This is necessary to use brian_end()

        namespace = None
        if network is not None:
            # Stop simulation when avalanche end is detected
            @implementation(CPPStandaloneCodeObject, '''
            double stop_on_avalanche_end(double lastspiketime_input, double maxQuietTime) {
                if (lastspiketime_input > maxQuietTime) {
                    brian_end();
                    std::exit(0);
                }
                return 0.0;
            }
            ''')
            @implementation('numpy', discard_units=True)
            @check_units(lastspiketime_input=second, maxQuietTime=second, result=1)
            def stop_on_avalanche_end(lastspiketime_input, maxQuietTime):
                if lastspiketime_input > maxQuietTime:
                    network.stop()
                return 0.0

            namespace = {'stop_on_avalanche_end': stop_on_avalanche_end, 'maxQuietTime': maxQuietTime}

        # Spike monitor used to compute the avalanche length
        M = SpikeMonitor(G, record=True)

        # Dummy neuron and synapses used to monitor the end of the avalanche from the spiking activity
        P = NeuronGroup(1, '''
        lastspiketime_input : second
        ''', namespace=namespace)
        P.lastspiketime_input = 0.0 * second
        Sae = Synapses(G, P, on_pre='''
        lastspiketime_input = 0.0 * second
        ''')
        Sae.connect('True')
        P.run_regularly('lastspiketime_input += dt', when='synapses')
        if network is not None:
            P.run_regularly('dummy = stop_on_avalanche_end(lastspiketime_input, maxQuietTime)', when='after_synapses')

        self.P = P
        self.M = M
        self.contained_objects.extend([P, M, Sae])

    def getAvalancheSize(self):
        return max(int(self.M.num_spikes - 1), 0)

    def getAvalancheLength(self):
        if len(self.M.t) > 1:
            minTime, maxTime = np.min(self.M.t), np.max(self.M.t)
            avalancheLength = (maxTime - minTime)
        else:
            avalancheLength = 0.0 * ms
        return avalancheLength


def estimatePowerLawScaling(net, microcircuit, nbSamples=1000, maxAvalancheTime=150 * ms):

    # Disable plasticity and spontaneous activity
    microcircuit.S.plastic = False
    microcircuit.G.noise.active = False

    # Spike generator used for input stimulation
    # NOTE: use a high weight to force spiking of the postsynaptic neuron
    nbInputs = len(microcircuit.G)
    G = SpikeGeneratorGroup(nbInputs, indices=[0], times=[0 * ms])
    Si = Synapses(G, microcircuit.G, model='w : 1', on_pre='''v_post += w * int(not_refractory_post)
                                                              c_in_tot_post += w * int(not_refractory_post)''')
    Si.connect(j='i')
    Si.w = 10.0

    # Detect the end of avalanche and stop the simulation if no spike occured in the last 5 ms
    D = AvalancheEndDetector(microcircuit.G, maxQuietTime=5 * ms, network=net)
    defaultclock.dt = 0.1 * ms
    net.add([G, Si, D])
    net.store('before_testing')

    # Generate multiple avalanches
    # NOTE: only select excitatory neurons
    targets = np.arange(len(microcircuit.G), dtype=np.int)
    validTargets = targets[microcircuit.G.ntype > 0]
    avalancheSizes = []
    for n in range(nbSamples):
        if (n + 1) % 100 == 0:
            logger.debug('Generating avalanche no.%d (out of %d)' % (n + 1, nbSamples))

        # Reinitialization
        net.restore('before_testing')

        # Chose a target neuron in the population
        target = np.random.choice(validTargets)
        G.set_spikes(indices=[target], times=[defaultclock.t + 1 * ms])
        net.run(maxAvalancheTime)

        # Get the number of elicited spikes
        logger.debug('Avalanche no.%d: size of %d, length of %4.3f ms' % (n + 1, D.getAvalancheSize(), D.getAvalancheLength() / ms))
        avalancheSizes.append(D.getAvalancheSize())

    avalancheSizes = np.array(avalancheSizes)

    # Compute the histogram of avalanche sizes
    sizes = np.arange(0, np.max(avalancheSizes) + 1)
    bins = np.concatenate((sizes, [np.max(avalancheSizes) + 1, ])) - 0.5
    hist, _ = np.histogram(avalancheSizes, bins)
    pdf = hist.astype(np.float) / np.sum(hist)
    assert len(pdf) == len(sizes)

    # Show histogram of avalanche sizes
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.scatter(sizes[pdf > 0], pdf[pdf > 0], color='k', marker='.')
    ax.set_xlabel('Avalanche size')
    ax.set_ylabel('Probability')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Fitting the power-law line in log-log domain
    mask = np.logical_and(sizes > 0, pdf > 0)
    logX = np.log(sizes[mask])
    logY = np.log(pdf[mask])
    p = np.polyfit(logX, logY, 1)
    a, b = p[0], p[1]
    logger.info('Estimated exponent using curve fitting: %f' % (a))

    # Plot the fitted powerlaw curve
    npl = np.arange(0, np.max(sizes) + 1)
    pl = np.exp(b) * (npl ** (a))
    ax.plot(npl, pl, color='k', linestyle='--')

    return fig


def main():

    # Choose the duration of the training
    duration = 60 * second
    targetCbf = 1.0

    logger.info('Simulating for target branching factor of %f' % (targetCbf))

    # Create the microcircuit
    # NOTE: p_max is chosen so to have an out-degree of N=16
    m = Microcircuit(connectivity='small-world', macrocolumnShape=[2, 2, 2], minicolumnShape=[4, 4, 4],
                     p_max=0.056, srate=1 * Hz, excitatoryProb=0.8, delay='1*ms + 2*ms * rand()')

    # Configure CRITICAL learning rule
    m.G.c_out_ref = targetCbf          # target critical branching factor
    m.S.alpha = 0.05                   # learning rate

    logger.info('Number of neurons in the population: %d' % (len(m.G)))

    # Configure the monitors and simulation
    # NOTE: setting a high time resolution increase the stability of the learning rule
    M = SpikeMonitor(m.G, record=True)
    Mg = StateMonitor(m.G, variables=['cbf'], record=True, dt=10 * ms)
    defaultclock.dt = 0.1 * ms
    net = Network(m.getBrianObjects())
    net.store('initialized')

    # Add inputs and monitors
    net.add([M, Mg])

    # Run the simulation with input stimuli and plasticity enabled
    m.S.plastic = True
    net.run(duration, report='text')

    # Compute population average firing rate
    avgOutputFiringRate = len(M.i) / (len(m.G) * duration)
    logger.info('Average output firing rate: %4.2f Hz' % (avgOutputFiringRate))

    # NOTE: compute statistics on excitatory neurons only
    meanCbf = np.mean(Mg.cbf.T[:, m.G.ntype > 0], axis=-1)
    fig = plt.figure(facecolor='white', figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Average output contributions')
    ax.plot(Mg.t, meanCbf, color='k')
    ax.set_ylim((0.0, 2.0))
    fig.tight_layout()
    fig.savefig('convergence.eps')

    # Visualization of the simulation
    # NOTE: show only the last 10 sec of the simulation
    fig = plt.figure(facecolor='white', figsize=(6, 5))
    plt.subplot(111)
    plt.title('Spiking activity (output)')
    plt.plot(M.t / ms, M.i, '.', color='b')
    plt.ylabel('Neurons')
    plt.xlabel('Time [ms]')
    plt.xlim([0.0, duration / ms])
    fig.tight_layout()

    # Disable spontaneous activity and let the population reach a resting state
    m.S.plastic = False
    m.G.noise.active = False
    net.remove([M, Mg])
    net.run(10 * second)
    net.store('after_learning')

    fig = estimatePowerLawScaling(net, m, nbSamples=1000, maxAvalancheTime=250 * ms)
    fig.savefig('avalanches_distribution.eps')

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Fix the seed of all random number generator
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    main()
    logger.info('All done.')
