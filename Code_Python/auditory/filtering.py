'''

 Copyright (c) 2012, "Simon Brodeur"
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:
 
  - Redistributions of source code must retain the above copyright notice, 
    this list of conditions and the following disclaimer.
  - Redistributions in binary form must reproduce the above copyright notice, 
    this list of conditions and the following disclaimer in the documentation 
    and/or other materials provided with the distribution.
  - Neither the name of the NECOTIS research group nor the names of its contributors 
    may be used to endorse or promote products derived from this software 
    without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
 NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
 OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 POSSIBILITY OF SUCH DAMAGE.
'''

import math
import numpy
import pylab
from brian import *
from brian.hears import *
import scipy
from scipy import signal
from scipy import *
from util import *
import scipy.interpolate

class GainFilterbank(Filterbank):
    def __init__(self, source, gain=1.0):
        Filterbank.__init__(self, source)
        self.gain = gain

    def buffer_apply(self, input):
        return self.gain * input

class GainController(object):
    def __init__(self, target, target_rms, time_constant):
        self.target = target
        self.target_rms = target_rms
        self.time_constant = time_constant

    def reinit(self):
        self.sumsquare = 0
        self.numsamples = 0

    def __call__(self, input):
        T = input.shape[0] / self.target.samplerate
        self.sumsquare += sum(input ** 2)
        self.numsamples += input.size
        rms = sqrt(self.sumsquare / self.numsamples)
        g = self.target.gain
        g_tgt = self.target_rms / rms
        tau = self.time_constant
        self.target.gain = g_tgt + exp(-T / tau) * (g - g_tgt)

class AutomaticGainControl(CombinedFilterbank):

    def __init__(self, source, rms=0.2, tau=50 * ms, update=10 * ms):
        CombinedFilterbank.__init__(self, source)
        source = self.get_modified_source()

        gain_fb = GainFilterbank(source)
        updater = GainController(gain_fb, rms, tau)
        control = ControlFilterbank(gain_fb, source, gain_fb, updater, update)
        self.set_output(control)


class InstantaneousEnvelopFilterbank(Filterbank):
    '''
    Calculate the instantaneous envelop of a signal
    '''
    def __init__(self, source):
        '''
        Constructor
        '''
        Filterbank.__init__(self, source)

    def buffer_apply(self, input):

        # Transformation to analytic signals
        z = scipy.signal.hilbert(input, axis=0)

        # Instantaneous amplitude/envelop estimation
        output = numpy.abs(z)
        return output


class InstantaneousFrequencyFilterbank(Filterbank):
    '''
    Calculate the instantaneous frequency and envelop of a signal
    
    Method based on:
    A. E. Barnes, "The calculation of instantaneous frequency 
    and instantaneous bandwidth," GEOPHYSICS, vol. 57, no. 11, pp. 1520-1524, 1992.
    '''
    def __init__(self, source, cf, windowSize=5 * ms, frange=None):
        '''
        Constructor
        '''
        self.cf = cf
        self.windowSize = windowSize

        if frange == None:
            frange = [0, source.samplerate]
        self.frange = frange
        Filterbank.__init__(self, source)
        self.nchannels = 3

    def buffer_apply(self, input):

        # Transformation to analytic signals
        z = scipy.signal.hilbert(input, axis=0)
        x = z.real
        y = z.imag

        # Instantaneous frequency estimation
        y_pT = numpy.concatenate((y[1:, :], y[-1, :].reshape((1, y.shape[1]))), axis=0)
        x_pT = numpy.concatenate((x[1:, :], x[-1, :].reshape((1, x.shape[1]))), axis=0)
        T = 1.0 / self.source.samplerate
        f = 1 / (2 * numpy.pi * T) * numpy.arctan((x * y_pT - x_pT * y) / (x * x_pT + y * y_pT))

        # Instantaneous amplitude/envelop estimation
        env = numpy.abs(z)

        # Binning process to average instantaneous frequency across channels
        N_BINS = 256
        fmin = self.frange[0]
        fmax = self.frange[1]

        # Clip indexes within the histogram range
        idx = numpy.round((f - fmin) / (fmax - fmin) * (N_BINS - 1))
        idx = numpy.clip(idx, -1, N_BINS)

        # Calculate an averaged instantaneous frequency over each block
        output = zeros((input.shape[0], self.nchannels))
        nsamples = round(self.windowSize * self.source.samplerate)
        nblocks = int(numpy.ceil(idx.shape[0] / nsamples))
        for n in range(nblocks):
            idxStart = nsamples * n
            idxEnd = numpy.min((idxStart + nsamples, idx.shape[0]))
            idxblock = idx[idxStart:idxEnd, :]
            envblock = env[idxStart:idxEnd, :]

            # Calculate the histogram of frequency distribution, summing the instantaneous amplitudes 
            # TODO: possibility to vectorize code below or translate to Weave
            hist = numpy.zeros(N_BINS)
            for i in range(idxblock.shape[0]):
                for j in range(idxblock.shape[1]):
                    # Consider only instantaneous frequencies within the specified range
                    histIdx = idxblock[i, j]
                    if histIdx >= 0 and histIdx < N_BINS:
                        hist[histIdx] += envblock[i, j]

            # Normalize the histogram
            hist = hist / (numpy.sum(hist))

            # Find the first local maxima in the histogram
            # TODO: avoid picking harmonics for low pitches
            peakIdx = hist.argmax()
            favg = float(peakIdx) / (N_BINS - 1) * (fmax - fmin) + fmin

            # Normalize output in the interval [0,1]
            favg = (favg - fmin) / (fmax - fmin)

#            width = N_BINS / 16
#            overlap = width / 2
#            slices = array_slice(hist, width, overlap)
#            peakIdx = 0
#            std = numpy.std(hist)
#            for i in range(len(slices)):
#                idxStart = i * (width - overlap)
#                foundIdx = idxStart + slices[i].argmax()
#                if hist[foundIdx] > 4 * std:
#                    peakIdx = foundIdx
#                    break

            # Find the voicing envelop based on the peak found.
            # Average only envelop on the channels that responded strongly to the fundamental frequency.
            FENV_GAIN = 3
            fenv = FENV_GAIN * hist[peakIdx]
            fenv = numpy.clip(fenv, 0, 1.0)

            # The maximum envelop amplitude over all channels (warning this includes unvoiced segments)
            fenvMax = numpy.amax(envblock)
            fenvMax = numpy.clip(fenvMax, 0, 1.0)

#            if favg > 200 * Hz and favg < 300 * Hz:
#                p = pylab.figure()
#                pylab.plot(hist)
#                pylab.show()

            output[idxStart:idxEnd, 0] = favg
            output[idxStart:idxEnd, 1] = fenv
            output[idxStart:idxEnd, 2] = fenvMax

        return output

class ButterworthFilterbank(LinearFilterbank):
    '''
    Butterworth filterbank
    '''
    def __init__(self, source, cf, bandwidth, order=1):
        cf = numpy.atleast_1d(cf)
        bandwidth = numpy.atleast_1d(bandwidth)

        b = numpy.zeros((len(cf), 3, order))
        a = numpy.zeros((len(cf), 3, order))
        for n in range(len(cf)):
            fc = cf[n] * 2 / float(source.samplerate)
            hbw = bandwidth[n] / float(source.samplerate)
            flow = max(fc - hbw, 0)
            fhigh = min(fc + hbw, 1.0)
            b_n, a_n = scipy.signal.butter(1, [flow, fhigh], 'bandpass')
            b[n, :, :] = numpy.tile(b_n, (order, 1)).T
            a[n, :, :] = numpy.tile(a_n, (order, 1)).T

        LinearFilterbank.__init__(self, source, b, a)

    def calculateFrequencyResponse(self, samplerate):

        fig = pylab.figure()
        pylab.title('Gaussian filterbank frequency response')
        pylab.hold(True)

        # Loop for each channel
        for n in range(self.nchannels):
            b = self.filt_b[n, :, :]
            a = self.filt_a[n, :, :]
            (h, F) = calculateCascadedFrequencyResponse(b, a)

            f = (h * (self.source.samplerate / 2) / numpy.pi) / kHz

            ratio = samplerate / self.source.samplerate
            pylab.semilogy(f[0:round((len(f) - 1) * ratio)], F[0:round((len(f) - 1) * ratio)], 'b')

        pylab.ylabel('Amplitude [dB]')
        pylab.xlabel('Frequency [kHz]')
        pylab.ylim(ymin=0.001)
        pylab.grid()

class GaussianFilterbank(LinearFilterbank):
    '''
    FIR Gaussian filterbank
    '''
    def __init__(self, source, cf, bandwidth, order=512):
        cf = numpy.atleast_1d(cf)
        bandwidth = numpy.atleast_1d(bandwidth)

        funcGaussian = lambda f, f0, df: numpy.exp(-(f - f0) ** 2 / (df ** 2))
        b = numpy.zeros((len(cf), order))
        for n in range(len(cf)):
            # Generate fft spectrum
            freq = numpy.linspace(0, float(source.samplerate) / 2, num=(2 * order) - 1)
            mag = funcGaussian(freq, cf[n], bandwidth[n])
            a = numpy.concatenate((numpy.array([0]), mag[::], numpy.array([mag[-1]]), mag[::-1]))
            # Calculate the impulse response of the filter by the inverse Fast Fourier Transform
            ir = numpy.fft.ifft(a)
            ir = ir[0:order].real
            b[n, :] = ir

        a = numpy.zeros((len(cf), order))
        a[:, 0] = 1.0

        self.cf = cf
        LinearFilterbank.__init__(self, source, b, a)

    def calculateFrequencyResponse(self, samplerate):

        fig = pylab.figure()
        pylab.title('Gaussian filterbank frequency response')
        pylab.hold(True)

        # Loop for each channel
        for n in range(self.nchannels):
            b = self.filt_b[n, :, :]
            a = self.filt_a[n, :, :]
            (h, F) = calculateCascadedFrequencyResponse(b, a)

            f = (h * (self.source.samplerate / 2) / numpy.pi) / kHz

            ratio = samplerate / self.source.samplerate
            pylab.semilogy(f[0:round((len(f) - 1) * ratio)], F[0:round((len(f) - 1) * ratio)], 'b')

        pylab.ylabel('Amplitude [dB]')
        pylab.xlabel('Frequency [kHz]')
        pylab.ylim(ymin=0.001)
        pylab.grid()

class PreEmphasisFilterbank(LinearFilterbank):
    '''
    Compensation for the -6dB/octave spectral slope of speech
    '''
    def __init__(self, source, alpha=0.94):
        b = numpy.zeros((source.nchannels, 2))
        b[:, 0] = 1
        b[:, 1] = -alpha
        a = numpy.zeros((source.nchannels, 2))
        a[:, 0] = 1
        a[:, 1] = 0
        LinearFilterbank.__init__(self, source, b, a)


class OuterMiddleEarFilter(DoNothingFilterbank):
    '''
    Based on Huber et al. Ann.Otol.Rhinol.Laryngol. 110 31-35 (2001)
    <gain>  <lower cutoff>  <upper cutoff>  <filter order>
    '''
    human_data_Huber = numpy.array([
        [0, 1300, 3100, 1],
        [-13, 4000, 6000, 1],
        ])

    '''
    Based on Ruggero
    <gain>  <lower cutoff>  <upper cutoff>  <filter order>    
    '''
    human_data_Ruggero = numpy.array([
        [-2, 1900, 4200, 1],
        [-3, 4500, 6300, 1],
        [-19, 8000, 12000, 1]
        ])

    human_data_generic = numpy.array([
        [1, 700, 1200, 1],
        ])

    def __init__(self, source, model='human_Ruggero'):

        if model == 'human_Huber':
            human_data = OuterMiddleEarFilter.human_data_Huber
        elif model == 'human_Ruggero':
            human_data = OuterMiddleEarFilter.human_data_Ruggero
        elif model == 'human_generic':
            human_data = OuterMiddleEarFilter.human_data_generic
        else:
            raise "Unsupported model: %s" % (model)

        resultSource = None

        # Parallel mode
        for i in range(human_data.shape[0]):
            gain = human_data[i, 0] * dB
            flow = human_data[i, 1] * Hz
            fhigh = human_data[i, 2] * Hz
            order = human_data[i, 3]

            # Skip filter that would introduce aliasing due to the limited sampling rate of the source
            if float(fhigh) >= float(source.samplerate / 2):
                continue

            filter = Butterworth(source, source.nchannels, order, fc=numpy.array([flow, fhigh]), btype='bandpass')
            if resultSource == None:
                resultSource = gain * filter
            else:
                resultSource = resultSource + gain * filter

        DoNothingFilterbank.__init__(self, resultSource)

class OuterEarFilter(LinearFilterbank):
    '''
    For headphone-delivered sound
    '''
    pass


class MiddleEarFilter(FIRFilterbank):
    '''
    Human data estimated from
    The Journal of the Acoustical Society of America, vol. 110, no. 6, p. 3107, 2001.
    '''
    human_data = numpy.array([
        [0.0, 0.0],
        [100.793, 1.19613e-09],
        [201.257, 2.38659e-09],
        [401.855, 4.76187e-09],
        [598.021, 7.74264e-09],
        [802.59, 1.02591e-08],
        [999.511, 8.57696e-09],
        [1208.47, 6.64083e-09],
        [1419.69, 6.1502e-09],
        [1618.96, 5.01187e-09],
        [1820.06, 4.64159e-09],
        [2016.02, 4.08424e-09],
        [2199.67, 3.16228e-09],
        [2401.81, 3.0824e-09],
        [2509.23, 2.85467e-09],
        [2583.5, 2.71227e-09],
        [2820.22, 2.44844e-09],
        [3033.32, 2.10001e-09],
        [3510.75, 1.80117e-09],
        [4004.82, 1.62596e-09],
        [4433.48, 1.19613e-09],
        [5061.53, 1.39458e-09],
        [5366.47, 1.32502e-09],
        [5606.94, 1.25893e-09],
        [5942.32, 1.0525e-09],
        [6488.4, 1.02591e-09],
        [7082.34, 9.02725e-10],
        [7619.35, 8.36031e-10],
        [8078.39, 7.94328e-10],
        [8564.39, 7.35642e-10],
        [9079.63, 6.81292e-10],
        [9486.49, 6.47308e-10],
        [10058.8, 6.30957e-10],
        [64000.0, 0.0]
        ])

    def __init__(self, source, ncoff=512):

        interpolationFunc = scipy.interpolate.interp1d(MiddleEarFilter.human_data[:, 0], MiddleEarFilter.human_data[:, 1] / max(MiddleEarFilter.human_data[:, 1]), kind='cubic')

        # Generate fft spectrum
        freq = numpy.linspace(0, float(source.samplerate) / 2, num=(2 * ncoff / 2) - 1)
        mag = interpolationFunc(freq)
        a = numpy.concatenate((numpy.array([0]), mag[::], numpy.array([mag[-1]]), mag[::-1]))
        # Calculate the impulse response of the filter by the inverse Fast Fourier Transform
        impulse_response = numpy.fft.ifft(a)
        if max(impulse_response.imag) > 1e-10:
            raise "Complex impulse response: magnitude spectrum is not symmetric!"
        impulse_response = impulse_response[1:(ncoff / 2) + 1].real

        FIRFilterbank.__init__(self, source, impulse_response, use_linearfilterbank=True)

class EnvelopFilter(LowPass):
    '''
    Bank of envelop filters performing full- or half-wave rectification followed by a low-pass filtering
    
    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
        
    ``fc``
        Value, list or array (with length = number of channels) of cutoff
        frequencies.

    '''
    def __init__(self, source, fc, mode='full-wave'):
        if mode == 'full-wave':
            self.rectifier = FunctionFilterbank(source, lambda x: abs(x))
        elif mode == 'half-wave':
            self.rectifier = FunctionFilterbank(source, lambda x: clip(x, 0, Inf))
        else:
            raise 'Unsupported mode: %s' % (mode)
        LowPass.__init__(self, self.rectifier, fc)


class DownsamplingFilterbank(Filterbank):
    '''
    classdocs
    '''

    def __init__(self, source, samplerate):
        '''
        Constructor
        '''

        self.ratio = float(samplerate) / float(source.samplerate)
        if math.modf(1 / self.ratio)[0] > 0.00001:
            raise ValueError('Downsampling is currently only possible for an integer factor')

        self.lpfilter = Cascade(source, LowPass(source, samplerate / 2), 8)

        Filterbank.__init__(self, self.lpfilter)
        self.samplerate = samplerate

    def buffer_apply(self, input):
        x = input
        step = round(1 / self.ratio)
        y = x[::step]
        return y

    def buffer_fetch_next(self, samples):
        start = self.next_sample
        inputSamples = math.ceil(samples / self.ratio)
        end = start + inputSamples
        self.next_sample += inputSamples

        input = self.source.buffer_fetch(start, end)
        output = self.buffer_apply(input)
        if output.shape[0] != samples:
            raise ValueError('Sample rate conversion returns a wrong number of values')
        return output

class InterleaverFilterbank(Filterbank):
    '''
    classdocs
    '''

    def __init__(self, source, samplerate):
        '''
        Constructor
        '''
        Filterbank.__init__(self, source)
        self.samplerate = samplerate

        self.ratio = float(samplerate) / float(source.samplerate)
        if math.modf(self.ratio)[0] > 0.00001:
            raise ValueError('Upsampling is currently only possible for an integer factor')

    def buffer_apply(self, input):
        x = input
        step = round(self.ratio)
        y = numpy.zeros((x.shape[0] * step, x.shape[1]))
        y[::step] = input
        return y

    def buffer_fetch_next(self, samples):
        start = self.next_sample
        inputSamples = math.ceil(samples / self.ratio)
        end = start + inputSamples
        self.next_sample += inputSamples

        input = self.source.buffer_fetch(start, end)
        output = self.buffer_apply(input)
        if output.shape[0] != samples:
            raise ValueError('Sample rate conversion returns a wrong number of values')
        return output


class UpsamplingFilterbank(CombinedFilterbank):
    '''
    classdocs
    '''

    def __init__(self, source, samplerate):
        '''
        Constructor
        '''
        CombinedFilterbank.__init__(self, source)
        self.samplerate = samplerate
        source = self.get_modified_source()

        interleaver = InterleaverFilterbank(source, samplerate)
        lpfilter = Cascade(interleaver, LowPass(interleaver, source.samplerate / 2), 8)
        self.set_output(lpfilter)

class ResamplingFilterbank(Filterbank):
    '''
    Warning: Scipy uses a FFT-implementation and produces artefacts
    '''

    def __init__(self, source, samplerate):
        '''
        Constructor
        '''
        self.ratio = float(samplerate) / float(source.samplerate)
        Filterbank.__init__(self, source)
        self.samplerate = samplerate

    def buffer_apply(self, input):
        x = input
        quality = 'sinc_best'
        window = None
        func = (lambda x, ratio, quality, window: signal.resample(x, int(round(len(x) * ratio)), window=window))
        if len(x.shape) == 1 or x.shape[1] == 1 :
            y = func(x, self.ratio, quality, window)
        else :
            y = numpy.array([])
            for i in range(x.shape[1]) :
                y = numpy.concatenate((y, func(x[:, i], self.ratio, quality, window)))
            y = y.reshape(x.shape[1], -1).T
        return y

    def buffer_fetch_next(self, samples):
        start = self.next_sample
        inputSamples = math.ceil(samples / self.ratio)
        end = start + inputSamples
        self.next_sample += inputSamples

        input = self.source.buffer_fetch(start, end)
        output = self.buffer_apply(input)
        if output.shape[0] != samples:
            raise ValueError('Sample rate conversion returns a wrong number of values')
        return output
