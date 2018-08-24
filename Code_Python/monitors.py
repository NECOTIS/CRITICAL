'''
Created on 21.02.2012

@author: "Simon Brodeur"
'''

import brian_no_units
from brian import *
from brian.hears import *
from brian.hears import filtering
import scipy
import numpy
import pylab
import threading
import time
import bisect
from itertools import izip
import itertools
import matplotlib
from matplotlib import cm
import pickle
from brian.library.IF import Izhikevich, AdaptiveReset
from brian.plotting import raster_plot



class NeuronRateMonitor(SpikeMonitor):

    def __init__(self, source, bin=None):
        SpikeMonitor.__init__(self, source, record=False)
        if bin:
            self._bin = int(bin / source.clock.dt)
        else:
            self._bin = 1 # bin size in number
        self._acc = numpy.zeros(len(self.source))
        self.rate = numpy.zeros(len(self.source))
        self._curstep = 0
        self._clock = source.clock

    def reinit(self):
        SpikeMonitor.reinit(self)
        self._acc = numpy.zeros(len(self.source))
        self.rate = numpy.zeros(len(self.source))
        self._curstep = 0

    def propagate(self, spikes):
        if self._curstep == 0:
            self.rate[:] = self._acc / float(self._bin * self._clock.dt)
            self._acc = numpy.zeros(len(self.source))
            self._curstep = self._bin

        for i in spikes:
            self._acc[i] += 1.0
        self._curstep -= 1

class HistRateMonitor(NetworkOperation):

    def __init__(self, G, refresh=500 * ms, nbHistBins=64):
        self.nbHistBins = nbHistBins
        self.fig = pylab.figure(facecolor='white')
        self.ax1 = self.fig.add_subplot(1, 1, 1)

        self.rateMonitor = NeuronRateMonitor(G, bin=refresh)
        NetworkOperation.__init__(self, lambda:None, clock=EventClock(dt=refresh))
        self.contained_objects += [self.rateMonitor]

    def __call__(self):

        rmin = min(self.rateMonitor.rate)
        rmax = max(self.rateMonitor.rate)
        hist, bins = numpy.histogram(self.rateMonitor.rate, bins=self.nbHistBins, range=(rmin, rmax), normed=False)
        hist = numpy.asfarray(hist) / len(self.rateMonitor.rate)

        self.ax1.cla()
        self.ax1.plot(bins[0:len(hist)], hist, color='blue')
        self.ax1.set_title('Firing rate histogram')
        self.ax1.set_xlabel('Firing rate [Hz]')
        self.ax1.set_ylabel('Probability')
        self.ax1.grid(True)

        self.fig.canvas.draw()

class RateMonitor(NetworkOperation):

    def __init__(self, G, refresh=100 * ms, preallocate=10000, clock=None, label=''):
        self.refresh = refresh
        self.label = label
        self.iterations = 0

        self.fig = pylab.figure(facecolor='white')
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.fig.subplots_adjust(hspace=0.5)

        self.data = numpy.zeros((preallocate, 2))
        self.dataCounter = 0
        self.refreshCounter = 0.0

        self.rateMonitor = NeuronRateMonitor(G, bin=clock.dt)
        NetworkOperation.__init__(self, lambda:None, clock=clock)
        self.contained_objects += [self.rateMonitor]

    def __call__(self):

        globalRate = numpy.mean(self.rateMonitor.rate)
        varRate = numpy.var(self.rateMonitor.rate)

        if self.dataCounter >= self.data.shape[0]:
            # Warning: dynamically resizing array
            self.data.resize((2 * self.data.shape[0], self.data.shape[1]))
        self.data[self.dataCounter, :] = [globalRate, varRate]
        self.dataCounter += 1
        t = arange(0, self.dataCounter) * self.clock.dt

        self.refreshCounter += self.clock.dt
        if self.refreshCounter >= self.refresh:
            self.ax1.cla()
            self.ax1.plot(t / ms, self.data[0:self.dataCounter, 0], color='blue')
            self.ax1.set_title('Global average firing rate: %s' % (self.label))
            self.ax1.set_xlabel('Time [ms]')
            self.ax1.set_ylabel('Firing rate [Hz]')
            self.ax1.grid(True)

            self.ax2.cla()
            self.ax2.plot(t / ms, self.data[0:self.dataCounter, 1], color='blue')
            self.ax2.set_title('Global variance of firing rate: %s' % (self.label))
            self.ax2.set_xlabel('Time [ms]')
            self.ax2.set_ylabel('Firing rate [Hz]')
            self.ax2.grid(True)

            self.fig.canvas.draw()
            self.refreshCounter = 0.0

class AvalancheEndDetector(NetworkOperation):

    def __init__(self, network, G, C, minDelay=0.0, clock=None):
        self.C = C

        self.lock = threading.Lock()
        self.counterMonitor = SpikeCounter(G)
        self.lastCounts = numpy.zeros(C.W.shape[0], dtype=int)
        self.elapsedTime = -1 * defaultclock.dt
        self.spikeCounts = 0
        self.minDelay = minDelay

        NetworkOperation.__init__(self, lambda:None, clock=clock)
        self.contained_objects += [self.counterMonitor]

    def __call__(self):
        counts = self.counterMonitor.count - self.lastCounts

        stop = False
        if numpy.sum(counts) == 0 and self.elapsedTime >= 0.0 and self.elapsedTime >= self.minDelay:
            network.stop()
            stop = True

        self.lastCounts[:] = self.counterMonitor.count[:]
        self.setSpikeCount(self.spikeCounts + numpy.sum(counts))

        if not stop:
            self.setElapsedTime(self.elapsedTime + defaultclock.dt)

    def setElapsedTime(self, t):
        try:
            self.lock.acquire()
            self.elapsedTime = t
        finally:
            self.lock.release()

    def getElapsedTime(self):
        t = None
        try:
            self.lock.acquire()
            t = self.elapsedTime
        finally:
            self.lock.release()
        return t

    def setSpikeCount(self, c):
        try:
            self.lock.acquire()
            self.spikeCounts = c
        finally:
            self.lock.release()

    def getSpikeCount(self):
        c = None
        try:
            self.lock.acquire()
            c = self.spikeCounts
        finally:
            self.lock.release()
        return c

class SecondTupleArray(object):
    def __init__(self, obj):
        self.obj = obj
    def __getitem__(self, i):
        return float(self.obj[i][1])
    def __len__(self):
        return len(self.obj)

class SpikeRateMonitor(SpikeMonitor):

    times = property(fget=lambda self:array(self._times))
    times_ = times
    rates = property(fget=lambda self:array(self._rates))
    rates_ = rates

    def __init__(self, source, bin=None):
        SpikeMonitor.__init__(self, source)
        if bin:
            self._bin = int(bin / source.clock.dt)
        else:
            self._bin = 1 # bin size in number
        self._rates = []
        for n in range(len(source)):
            self._rates.append([])
        self._times = []
        self._curstep = 0
        self._clock = source.clock
        self._factor = 1. / float(self._bin * source.clock.dt)

    def reinit(self):
        SpikeMonitor.reinit(self)
        self._rates = []
        for n in range(len(self.source)):
            self._rates.append([])
        self._times = []
        self._curstep = 0

    def propagate(self, spikes):
        if self._curstep == 0:
            for n in range(len(self.source)):
                self._rates[n].append(0.)
            self._times.append(self._clock._t) # +.5*bin?
            self._curstep = self._bin

        for spike in spikes:
            self._rates[spike][-1] += self._factor
        self._curstep -= 1


class RealtimeSpikeMonitor(NetworkOperation):

    def __init__(self, G, refresh=50 * ms, showlast=500 * ms, unit=ms):

        self.refresh = refresh
        self.showlast = showlast

        # Monitors
        self.monitor = SpikeMonitor(G)

        self.clock = EventClock(dt=refresh)
        NetworkOperation.__init__(self, lambda:None, clock=self.clock)
        self.contained_objects.append([self.monitor])

        self.fig = pylab.figure(facecolor='white')
        self.ax = self.fig.add_subplot(1, 1, 1)
        line, = pylab.plot([], [], '.', color='grey')
        self.line = line

        self.ax.set_ylim((0, len(G)))
        pylab.ylabel('Neuron number')
        if unit == ms:
            pylab.xlabel('Time [ms]')
        elif unit == second:
            pylab.xlabel('Time [sec]')
        else:
            raise Exception('Unsupported unit provided')
        self.unit = unit
        pylab.title('')

    def get_plot_coords(self, tmin=None, tmax=None):
        allst = []
        mspikes = self.monitor.spikes
        if tmin is not None and tmax is not None:
            x = SecondTupleArray(mspikes)
            imin = bisect.bisect_left(x, tmin)
            imax = bisect.bisect_right(x, tmax)
            mspikes = mspikes[imin:imax]
        if len(mspikes):
            sn, st = numpy.array(mspikes).T
        else:
            sn, st = numpy.array([]), numpy.array([])
        st /= self.unit
        allsn = [sn]
        allst.append(st)
        sn = hstack(allsn)
        st = hstack(allst)
        nmax = len(self.monitor.source)
        return st, sn, nmax

    def __call__(self):

        if matplotlib.is_interactive():
            st, sn, nmax = self.get_plot_coords(self.clock._t - float(self.showlast), self.clock._t)
            self.ax.set_ylim((0, len(self.monitor.source)))
            self.ax.set_xlim((self.clock.t - self.showlast) / self.unit, self.clock.t / self.unit)
            self.line.set_xdata(numpy.array(st))
            self.line.set_ydata(sn)
            self.fig.canvas.draw()


class RealtimeMatrixMonitor(NetworkOperation):

    def __init__(self, C, wmin=None, wmax=None, refresh=100 * ms):
        self.C = C
        self.wmin = float(wmin)
        self.wmax = float(wmax)
        self.cmap = matplotlib.cm.get_cmap('hot')
        self.fig = pylab.figure(facecolor='white')
        self.ax = self.fig.add_subplot(1, 1, 1)

        clock = EventClock(dt=refresh)
        NetworkOperation.__init__(self, lambda:None, clock=clock)

    def __call__(self):
        W = self.C.W.todense()
        wmin, wmax = self.wmin, self.wmax
        if wmin is None: wmin = amin(W)
        if wmax is None: wmax = amax(W)
        if wmax - wmin < 1e-20: wmax = wmin + 1e-20
        W = self.cmap(numpy.clip((W - wmin) / (wmax - wmin), 0, 1), bytes=True)
        self.ax.imshow(W[:, :, :3], aspect='auto', origin='lower')
        self.fig.canvas.draw()


class RealtimeHistMonitor(NetworkOperation):

    def __init__(self, C, wmin=None, wmax=None, nbHistBins=256, refresh=100 * ms):
        self.C = C
        self.wmin = float(wmin)
        self.wmax = float(wmax)
        self.nbHistBins = nbHistBins

        self.fig = pylab.figure(facecolor='white')
        self.ax = self.fig.add_subplot(1, 1, 1)

        clock = EventClock(dt=refresh)
        NetworkOperation.__init__(self, lambda:None, clock=clock)

    def __call__(self):

        # Flatten sparse matrix
        W = numpy.zeros(self.C.W.nnz)
        idx = 0
        for i in range(self.C.W.shape[0]):
            row = self.C.W[i, :]
            for j, w in izip(row.ind, row):
                W[idx] = w
                idx += 1

        wmin, wmax = self.wmin, self.wmax
        if wmin is None: wmin = amin(W)
        if wmax is None: wmax = amax(W)
        if wmax - wmin < 1e-20: wmax = wmin + 1e-20

        hist, bins = numpy.histogram(W, bins=self.nbHistBins, range=(wmin, wmax), normed=False)

        self.ax.cla()
        self.ax.plot(bins[0:len(hist)], hist)
        #self.ax.set_ylim((0.0, 1.0))
        self.ax.set_title('Synaptic weight strength histogram')
        self.ax.set_xlabel('Synaptic weight strength')
        self.ax.set_ylabel('Probability density')

        self.fig.canvas.draw()

class LastSpikeMonitor(Connection, Monitor):

    def __init__(self, source):
        self.source = source
        self.target = None
        self.lastSpikeTimes = numpy.zeros(len(self.source))
        self.W = None
        self.delay = 0

    def reinit(self):
        self.lastSpikeTimes = numpy.zeros(len(self.source))

    def propagate(self, spikes):
        if len(spikes):
            self.lastSpikeTimes[spikes] = self.source.clock.t

    def origin(self, P, Q):
        '''
        Returns the starting coordinate of the given groups in
        the connection matrix W.
        '''
        return (P.origin - self.source.origin, 0)

    def compress(self):
        pass

