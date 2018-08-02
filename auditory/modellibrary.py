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

from brian import *
from brian.library.ionic_currents import *
from brian.hears import *
from brian.hears import filtering
import scipy
import numpy
import pylab
import time
import gammatone
import random
from filtering import *
from warping import *
from util import *
import cochlea

def createSourceFromFile(filename, fs=16 * kHz, level=50 * dB, maxlen=None):
    from scipy.io import wavfile
    fsA, audio = wavfile.read(filename)
    if maxlen != None:
        audio = audio[1:round(maxlen * fsA)]

    audio = numpy.float32(audio)
    audio = audio / max(abs(audio))

    if fsA != int(fs):
        print 'Warning: sample rate converting from %ikHz to %ikHz is needed for file ''%s''' % (fsA, fs, filename)
        audio = resample(audio, float(fs) / fsA)
        print 'done.'

    sound = Sound(audio)
    sound.samplerate = fs
    sound.level = level
    return sound;

def simpleSourceModel(audioFilename, samplerate=16 * kHz, speechLevel=50 * dB, intervalMin=None, intervalMax=None, showInfo=False):

    # Load speech and noise signal
    speech = createSourceFromFile(audioFilename, fs=samplerate)
    speech.level = speechLevel
    source = speech

    if intervalMin == None or intervalMin < 0:
        intervalMin = 0 * ms
    if intervalMax == None or intervalMax > source.duration:
        intervalMax = source.duration

    sourceSeg = source[intervalMin:intervalMax]

    if showInfo:
        print 'Mixed source signal range: min = %f, max = %f' % (source.min(), source.max())
        p = figure()
        sourceSeg.spectrogram(low=0 * Hz, high=samplerate / 2, NFFT=int(source.samplerate * 4 * ms), noverlap=0)
        title('Mixed source wide-band spectrogram\n')
#        p.savefig('../output/%s_snr%ddB_wide_spectrum.png' % (noiseType, SNR), dpi=100)

        p = figure()
        sourceSeg.spectrogram(low=0 * Hz, high=samplerate / 2, NFFT=int(source.samplerate * 20 * ms), noverlap=0)
        title('Mixed source narrow-band spectrogram\n')
#        p.savefig('../output/%s_snr%ddB_narrow_spectrum.png' % (noiseType, SNR), dpi=100)

    return sourceSeg

def outerMiddleEarModel(source, type='OME', showInfo=False):
    # -------------------------
    # Outer/Middle ear model
    # -------------------------

    if type == 'OME':
        ome = OuterMiddleEarFilter(source, model='human_Huber')
    elif type == 'pre-emphasis':
        ome = PreEmphasisFilterbank(source)
    elif type == 'ME':
        ome = MiddleEarFilter(source)
    else:
        raise 'Unsupported type: %s' % (type)

    if showInfo:
        output = ome.process()
        output = Sound(output, samplerate=source.samplerate)
        print 'Middle-ear signal range: min = %f, max = %f' % (output.min(), output.max())
        p = figure()
        output.spectrogram(low=0 * Hz, high=ome.samplerate / 2, NFFT=int(source.samplerate * 4 * ms), noverlap=0)
        title('Middle-ear wide-band spectrogram\n (type = %s)' % (type))
        p = figure()
        output.spectrogram(low=0 * Hz, high=ome.samplerate / 2, NFFT=int(source.samplerate * 20 * ms), noverlap=0)
        title('Middle-ear narrow-band spectrogram\n (type = %s)' % (type))

#        title('Middle-ear Spectrogram\n filterbank=%s , noise=%s, SNR=%ddB' % (filterbankType, noiseType, SNR))
#        p.savefig('../output/%s_%s_snr%ddB_ome_spectrum.png' % (filterbankType, noiseType, SNR), dpi=100)

    return ome;

def getModelParam(modelParams, name, defaultValue=None):
    value = defaultValue
    if name in modelParams:
        value = modelParams[name]
    return value;

def basilarMembraneModel(source, type='drnl', nchannels=64, cfmin=20 * Hz, cfmax=8 * kHz, samplerate=64 * kHz, warping='universal', modelParams={}, showInfo=False):
    # -------------------------
    # Cochlear filterbank model
    # -------------------------

    # Resample the source signal to a higher sampling rate depending on the filterbank used
    source = UpsamplingFilterbank(source, samplerate)

    if warping == 'erbspace':
        cf = erbspace(cfmin, cfmax, nchannels)
    elif warping == 'melspace':
        cf = melspace(cfmin, cfmax, nchannels)
    elif warping == 'universal':
        cf = universalWarpingFunctionSpace(cfmin, cfmax, nchannels)
    else:
        raise 'Unsupported warping: %s' % (warping)

    if type == 'gammatoneSlaney':
        bw = 10 ** (0.03728 + 0.78563 * log10(cf))
        cfb = ApproximateGammatone(source, cf, bw, order=3)
    elif type == 'gammatoneHohmann':
        b = getModelParam(modelParams, 'b', 1.019)
        cfb = Gammatone(source, cf, b=b)
    elif type == 'gammatoneCustom':
        cfb = gammatone.Gammatone(source, cf)
    elif type == 'drnl':
        dataType = getModelParam(modelParams, 'dataType', 'Lopez-Paveda2001')
        cfb = cochlea.DRNL(source, cf, type=dataType)
#        dataType = getModelParam(modelParams, 'dataType', 'human')
#        cfb = DRNL(source, cf, type=dataType)
    else:
        raise 'Unsupported type: %s' % (type)

    if showInfo:
        print '##########################'
        print 'Filterbank spacing table'
        for i in range(nchannels):
            print 'Channel %d: %fHz' % (nchannels - i, cf[nchannels - i - 1])
        print '##########################'

        output = cfb.process()
        print 'CBF signal range: min = %f, max = %f' % (output.min(), output.max())
        p = figure()
#        title('Cochlear filterbank output\n (filterbank=%s , noise=%s, SNR=%ddB)' % (type, noiseType, SNR))
#        imshow(output.T, aspect='auto', origin='lower', extent=[float(intervalMin), float(intervalMax), 1, nchannels])
        title('Basilar membrane response\n (type = %s, warping = %s)' % (type, warping))
        imshow(output.T, aspect='auto', origin='lower', extent=[float(0), float(source.duration), 1, nchannels])
        ylabel('Channel number')
        xlabel('time [sec]')
        #p.savefig('../output/%s_%s_snr%ddB_cfb.png' % (type, noiseType, SNR), dpi=100)

    return cfb

def innerHairCellModel(source, type='root-cube', nchannels=None, samplerate=8 * kHz, modelParams={}, showInfo=False):
    # -------------------------
    # Inner-hair cell model
    # -------------------------

    if nchannels == None:
        nchannels = source.nchannels

    if type == 'root-cube':
        # Rectification and compression
        order = getModelParam(modelParams, 'order', 3.0)
        funcCompression = lambda x: order * clip(x, 0, Inf) ** (1.0 / order)
        ihcRect = FunctionFilterbank(source, funcCompression)
        # Low-pass filtering
        fc = getModelParam(modelParams, 'fc', 500 * Hz)
        ihc = LowPass(ihcRect, fc)
    elif type == 'lowpass':
        funcCompression = lambda x: x
        fc = getModelParam(modelParams, 'fc', 500 * Hz)
        ihc = LowPass(source, fc)
    elif type == 'envelop':
        funcCompression = lambda x: x
        fc = getModelParam(modelParams, 'fc', 500 * Hz)
        mode = getModelParam(modelParams, 'mode', 'full-wave')
        ihc = EnvelopFilter(source, fc, mode)
    else:
        raise 'Unsupported type: %s' % (type)

    # Resample the channels to a lower sampling rate
    ihc = DownsamplingFilterbank(ihc, samplerate)

    if showInfo:
        output = ihc.process()
        print 'IHC signal range: min = %f, max = %f' % (output.min(), output.max())
        p = figure()
        imshow(output.T, aspect='auto', origin='lower', extent=[float(0), float(source.duration), 1, nchannels])
        title('Inner-hair cells response\n (type = %s)' % (type))
        ylabel('Channel number')
        xlabel('time [sec]')

        # Compression analysis
        levels = range(0, 100, 1)
        x = numpy.zeros(len(levels))
        y = numpy.zeros(len(levels))
        for i in range(len(levels)):
            x[i] = float(0.05 * gain(float(levels[i]) * dB - 50 * dB))
            y[i] = funcCompression(x[i])
        p = figure()
        plot(x, y)
        xlabel('Amplitude in')
        ylabel('Amplitude out')
        grid()

    return ihc

def auditoryNerveModel(source, neuronModel='lif', modelParams={}, showInfo=False):
    # -------------------------
    # Auditory nerve model
    # -------------------------

    if neuronModel == 'lif':
        tau = getModelParam(modelParams, 'tau', 15 * ms)
        sigma = getModelParam(modelParams, 'sigma', 0.02)
        refractory = getModelParam(modelParams, 'refractory', 5 * ms)
        threshold = getModelParam(modelParams, 'threshold', 0.55)
        eqs = '''
        dv/dt = (I-v)/tau + sigma*(2/tau)**.5*xi : 1
        I : 1
        '''
        P = FilterbankGroup(source, 'I', eqs, reset=0, threshold=threshold, refractory=refractory)
    elif neuronModel == 'ipp':
        P = FilterbankGroup(source, 'x', model='x : Hz', threshold=PoissonThreshold(state='x'))
    else:
        raise 'Unsupported neuron model: %s' % (neuronModel)

    # -------------------------
    # Simulation and monitoring
    # -------------------------
    if showInfo:
        M = SpikeMonitor(P)
        p = figure()
        run(source.duration, report='text')
        raster_plot(M)

    return P
