# Copyright (c) 2012-2018, NECOTIS
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

from brian2.units.stdunits import ms
from brian2.units.allunits import second
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


class RocPattern(object):
    def __init__(self, orders, times, width):
        self.orders = orders
        self.times = times
        self.width = width


def plotPatterns(patterns, unit=ms):

    fig = plt.figure(facecolor='white')
    line, = plt.plot([], [], '.', color='gray')
    ax = fig.add_subplot(1, 1, 1)

    nbNeurons = np.max([len(p.orders) for p in patterns])
    min_y = -0.5
    ax.set_ylim((min_y, nbNeurons))
    plt.ylabel('Neuron number')
    if unit == ms:
        plt.xlabel('Time [ms]')
    elif unit == second:
        plt.xlabel('Time [sec]')
    else:
        raise Exception('Unsupported unit provided')
    plt.title('Rank-order coded patterns')

    # Draw spikes
    spikes = []
    for n, p in enumerate(patterns):
        for i, t in zip(range(nbNeurons), p.times):
            spikes.append((i, t + n * p.width))

    allst = []
    if len(spikes):
        sn, st = np.array(spikes).T
    else:
        sn, st = np.array([]), np.array([])
    st /= unit
    allsn = [sn]
    allst.append(st)
    sn = np.hstack(allsn)
    st = np.hstack(allst)

    line.set_xdata(np.array(st))
    ax.set_xlim((0.0, np.max(st)))
    line.set_ydata(sn)

    # Draw lines between each pattern
    for n in range(len(patterns)):
        t = n * (patterns[n].width / unit)
        line = Line2D([t, t], ax.get_ylim(), color='grey', linestyle='--', linewidth=1.0)
        ax.add_line(line)

    fig.canvas.draw()
    return fig


def generateRankOrderCodedPatterns(nbNeurons, nbPatterns, widthEpoch=10 * ms, padding=1 * ms, refractory=0.0 * ms):

    spiketimes = np.zeros((nbPatterns, nbNeurons)) * ms
    orders = np.zeros((nbPatterns, nbNeurons))

    # Loop for each class to generate
    patterns = []
    minT = padding
    maxT = widthEpoch - padding
    times = np.linspace(minT, maxT, nbNeurons)
    for n in range(nbPatterns):

        logger.debug('Generating pattern no.%d (out of %d)' % (n + 1, nbPatterns))

        conflictFound = True
        nbRetry = 0
        maxRetry = 100000
        while conflictFound and nbRetry < maxRetry:

            if nbRetry > 0 and nbRetry % 1000 == 0:
                logger.debug('Number of retries: %d' % (nbRetry))

            genOrders = list(range(nbNeurons))
            np.random.shuffle(genOrders)

            # Ensure that the pattern doesn't already exist
            conflictFound = False
            for m in range(n):
                if (genOrders == orders[m, :]).all():
                    conflictFound = True
                    nbRetry += 1
                    break

            if not conflictFound and refractory > 0.0:
                # Ensure each neuron is not in refractory period if concatenated with every other class
                for target in range(nbNeurons):
                    for m in range(n):
                        if times[genOrders[target]] + widthEpoch - spiketimes[m, target] < refractory:
                            conflictFound = True
                            nbRetry += 1
                            break
                    if conflictFound:
                        break

        if conflictFound:
            raise Exception('Unable to generate all patterns: %d generated' % (n))

        patterns.append(RocPattern(genOrders, times[genOrders], widthEpoch))

    return patterns


def generateRankOrderCodedData(patterns, duration, delayEpoch):
    t = 0.0 * second
    indices = []
    times = []
    while t < duration:
        p = np.random.choice(patterns)
        if t + p.width + delayEpoch >= duration:
            break
        indices.extend(range(len(p.times)))
        times.extend(t + p.times)
        t += p.width + delayEpoch
    indices = np.array(indices, dtype=np.int)
    times = np.array(times) * second

    # Sort by time
    sortIndices = np.argsort(times)
    times = times[sortIndices]
    indices = indices[sortIndices]

    return indices, times
