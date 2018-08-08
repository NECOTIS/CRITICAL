'''
 Main entry

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

import os
import sys
from os.path import *
import numpy
import numpy.random
import scipy
from scipy.io import wavfile
import pylab
import time
import pickle
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib._pylab_helpers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import brian_no_units
from brian import *
from brian.hears import *
from brian.hears import filtering

from auditory.modellibrary import *
from auditory.util import *
from auditory.filtering import *

from util import *
from learning import *
from microcircuit import *
from monitors import *
from testing import *
import defaultparams

class LTSOCP_AudioMonitor(NetworkOperation):
    '''
    Realtime plotting of the mean and variance of neuron contribution
    '''

    def __init__(self, tuner, features, audioDuration, refresh=100 * ms, preallocate=10000, clock=None):
        self.tuner = tuner
        self.features = features
        self.refresh = refresh
        self.output = features.process(buffersize=4096, duration=audioDuration)

        self.fig = pylab.figure(facecolor='white')
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.fig.subplots_adjust(hspace=0.5)

        self.data = numpy.zeros((preallocate, 4))
        self.dataCounter = 0
        self.refreshCounter = 0.0

        NetworkOperation.__init__(self, lambda:None, clock=clock)

    def __call__(self):

        # Calculate average contributions for each neurons
        sumContrib = numpy.sum(self.tuner.neuronLogData, axis=1)
        avgContrib = numpy.zeros(self.tuner.neuronLogData.shape[0])
        globalAvgContrib = 0.0
        nbNonNulContrib = 0.0
        for i in range(self.tuner.neuronLogData.shape[0]):
            if self.tuner.neuronLogDataLen[i] > 0:
                avgContrib[i] = sumContrib[i] / float(self.tuner.neuronLogDataLen[i])
                globalAvgContrib += avgContrib[i]
                nbNonNulContrib += 1

        if nbNonNulContrib > 0:
            globalAvgContrib /= float(nbNonNulContrib)

        # Calculate the average variance across all neurons
        globalVarContrib = 0.0
        nbNonNulContrib = 0.0
        varDataFlat = numpy.array([])
        for i in range(self.tuner.neuronLogData.shape[0]):
            if self.tuner.neuronLogDataLen[i] > 0:
                curVarData = self.tuner.neuronLogData[i, 0:self.tuner.neuronLogDataLen[i]]
                varContrib = numpy.var(curVarData)
                varDataFlat = numpy.append(varDataFlat, curVarData)
                globalVarContrib += varContrib
                nbNonNulContrib += 1
        localVarContrib = numpy.var(varDataFlat)
        if nbNonNulContrib > 0:
            globalVarContrib /= float(nbNonNulContrib)

        if self.dataCounter >= self.data.shape[0]:
            # Warning: dynamically resizing array
            self.data.resize((2 * self.data.shape[0], self.data.shape[1]), refcheck=False)
        self.data[self.dataCounter, :] = [globalAvgContrib, globalVarContrib, localVarContrib, self.clock.t]
        self.dataCounter += 1

        self.refreshCounter += self.clock.dt
        if self.refreshCounter >= self.refresh and self.tuner.globalLogDataLen > 0:
            self.ax1.cla()
            t = self.tuner.logTimes[0:self.tuner.globalLogDataLen]
            tt = self.data[0:self.dataCounter - 1, 3]

            #self.ax1.plot(t / ms, self.tuner.globalLogData[0:self.tuner.globalLogDataLen, 0], color='blue')
            self.ax1.plot(tt, self.data[0:self.dataCounter - 1, 0], color='black', linestyle='solid')
            #self.ax1.set_title('Global average neural contribution')
            self.ax1.set_xlabel('Time [sec]')
            self.ax1.set_ylabel('Amplitude')
            self.ax1.set_xlim((0.0, tt[-1]))
            self.ax1.set_ylim((0.9, 1.1))
            self.ax1.grid(True)

            nbModulationFilters = 4
            audioDuration = tt[-1]
            self.ax2.cla()

            outputEnd = (self.clock.dt / defaultclock.dt) * (self.dataCounter - 1)
            self.ax2.imshow(self.output.T[:, 0:outputEnd], aspect='auto', origin='lower', extent=[float(0), float(audioDuration), 1, self.features.nchannels])
            self.ax2.set_ylabel('Channel number')
            self.ax2.set_xlabel('Time [sec]')
            for i in range(nbModulationFilters):
                y = i * (self.features.nchannels / nbModulationFilters) + 1
                if i > 0:
                    l = Line2D([float(0), float(audioDuration)], [y, y], linewidth=2, color='black', linestyle='--')
                    self.ax2.add_line(l)

            self.fig.canvas.draw()
            self.refreshCounter = 0.0

def getTimitAudioFileList(timitBasePath, excludeList=['SA']):

    # Get a list of female and male audio files
    timitdir = abspath(timitBasePath)
    femaleAudioFileList = []
    maleAudioFileList = []
    for dirpath, dirnames, filenames in os.walk(timitdir):
        folderName = os.path.basename(dirpath)
        if folderName.startswith("M"):
            sex = 'male'
        elif folderName.startswith("F"):
            sex = 'female'
        else:
            continue

        for filename in filenames:
            # Only search for audio files
            if filename.endswith(".MSWAV"):
                keepFilename = True
                for excludePrefix in excludeList:
                    if filename.startswith(excludePrefix):
                        keepFilename = False
                        break

                if keepFilename:
                    audioFilename = abspath(normpath(join(dirpath, filename)))
                    if sex == 'female':
                        femaleAudioFileList.append(audioFilename)
                    elif sex == 'male':
                        maleAudioFileList.append(audioFilename)
                    else:
                        raise Exception()

    print 'Number of audio files found for female: %d' % (len(femaleAudioFileList))
    print 'Number of audio files found for male: %d' % (len(maleAudioFileList))

    return (femaleAudioFileList, maleAudioFileList)

def concatenateTimitFiles(timitBasePath, fs=16 * kHz, silenceTime=500 * ms, totalTime=60 * second):

    print 'Recursively loading audio files list from TIMIT base path...'
    femaleAudioFileList, maleAudioFileList = getTimitAudioFileList(timitBasePath)

    # Shuffle lists
    numpy.random.shuffle(femaleAudioFileList)
    numpy.random.shuffle(maleAudioFileList)

    # Interleave arrays
    minLen = numpy.min((len(femaleAudioFileList), len(maleAudioFileList)))
    femaleAudioFileList = femaleAudioFileList[0:minLen]
    maleAudioFileList = maleAudioFileList[0:minLen]
    audioFileList = numpy.vstack((femaleAudioFileList, maleAudioFileList)).ravel([-1])

    print 'Concatenating random audio files...'
    # Concatenate audio files
    index = 0
    silenceCount = int(fs * silenceTime)
    silence = numpy.zeros(silenceCount)
    targetBufferCount = int(fs * totalTime)
    buffer = numpy.zeros(targetBufferCount)
    bufferCount = 0
    while bufferCount < targetBufferCount:

        if index >= len(audioFileList):
            raise Exception('Not enough audio files for concatenation to specified time!')

        filename = audioFileList[index]
        index += 1
        fsA, audio = wavfile.read(filename)

        audio = numpy.float32(audio)
        audio = audio / max(abs(audio))

        if fsA != int(fs):
            print 'Warning: sample rate converting from %ikHz to %ikHz is needed for file ''%s''' % (fsA, fs, filename)
            audio = resample(audio, float(fs) / fsA)

        bufferStart = bufferCount
        bufferEnd = bufferStart + len(audio) + silenceCount
        if bufferEnd >= buffer.shape[0]:
            # Resize buffer
            buffer.resize((2 * buffer.shape[0],))

        # Append the audio signal and the silence
        buffer[bufferStart:bufferEnd] = numpy.concatenate((audio, silence))
        bufferCount = bufferEnd

    # Trim buffer
    buffer = buffer[0:bufferCount]

    return buffer

def auditoryFeatureExtractionModel(audioFilename, audioDuration=None, offset=0 * ms, extractParams={}, modelParams={}, showPlot=False):

    cfbmin = getParameter(extractParams, 'cfbmin', 100 * Hz)
    cfbmax = getParameter(extractParams, 'cfbmax', 4 * kHz)
    mfbmin = getParameter(extractParams, 'mfbmin', 5 * Hz)
    mfbmax = getParameter(extractParams, 'mfbmax', 15 * Hz)
    nbModulationFilters = getParameter(extractParams, 'nbModulationFilters', 4)
    testDuration = getParameter(extractParams, 'testDuration', 5 * second)
    fs = getParameter(extractParams, 'fs', 1 / defaultclock.dt)

    # Auditory model
    source = simpleSourceModel(audioFilename, samplerate=8 * kHz, speechLevel=50 * dB, intervalMin=offset, intervalMax=offset + audioDuration, showInfo=False)
    ome = outerMiddleEarModel(source, type='pre-emphasis', showInfo=False)
    cfb = basilarMembraneModel(ome, type='gammatoneCustom', nchannels=16, cfmin=cfbmin, cfmax=cfbmax, warping='melspace', modelParams={}, showInfo=False)
    ihc = innerHairCellModel(cfb, type='root-cube', samplerate=4 * kHz, modelParams={'fc':500 * Hz, 'order':3}, showInfo=False)

    # Amplitude-modulation filters for modulation estimation
    ihcRestruct = RestructureFilterbank(ihc, numrepeat=nbModulationFilters)

    if mfbmin != mfbmax and nbModulationFilters > 1:
        cf = linspace(mfbmin, mfbmax, nbModulationFilters)
        bandwidth = (max(cf) - min(cf)) / (nbModulationFilters - 1) * numpy.ones(len(cf))
    else:
        cf = mfbmin * numpy.ones(nbModulationFilters)
        bandwidth = getParameter(extractParams, 'mfbbw', mfbmin) * numpy.ones(len(cf))

    gfb = ButterworthFilterbank(ihcRestruct, numpy.tile(cf, ihc.nchannels), numpy.tile(bandwidth, ihc.nchannels), order=3)

    # Extract instantaneous envelop and apply a compressive non-linearity
    instEnv = InstantaneousEnvelopFilterbank(gfb)
    order = 5
    funcCompression = lambda x: order * clip(x, 0, Inf) ** (1.0 / order)
    instCompEnv = FunctionFilterbank(instEnv, funcCompression)
    features = DownsamplingFilterbank(instCompEnv, fs)

    # Restructure to separate modulation frequency groups together 
    indexMapping = []
    for i in range(nbModulationFilters):
        idx = range(i, features.nchannels, nbModulationFilters)
        indexMapping.extend(idx)
    features = RestructureFilterbank(features, indexmapping=indexMapping)

    # Normalise features using the first few seconds of input
    print 'Simulating for normalization...'
    testDuration = numpy.min((audioDuration, testDuration))
    output = features.process(buffersize=4096, duration=testDuration)
    features /= numpy.max(output)
    print 'done.'

    # Generate spike trains
    neuronGroup = auditoryNerveModel(features, neuronModel='lif', modelParams=modelParams)

    if showPlot:
        print 'Simulating for visualization...'
        # Visualize the modulation patterns
        output = features.process(buffersize=4096, duration=audioDuration)
        print 'done.'

        p = pylab.figure(facecolor='white')
        pylab.imshow(output.T, aspect='auto', origin='lower', extent=[float(0), float(audioDuration), 1, features.nchannels])
        pylab.ylabel('Channel number')
        pylab.xlabel('Time [sec]')
        ax = pylab.gca()
        for i in range(nbModulationFilters):
            y = i * (features.nchannels / nbModulationFilters) + 1
            if i > 0:
                l = Line2D([float(0), float(audioDuration)], [y, y], linewidth=2, color='black', linestyle='--')
                ax.add_line(l)

            pylab.text(0.1 * float(audioDuration), y + 0.5 * features.nchannels / nbModulationFilters - 1, 'M = %2.0fHz' % (cf[i]), fontsize=14, color='black')

            yh = (i + 0.8) * (features.nchannels / nbModulationFilters) + 1
            yl = (i + 0.0) * (features.nchannels / nbModulationFilters) + 2
            pylab.text(0.95 * float(audioDuration), yh, '%4.0fHz' % (cfbmax), fontsize=12, color='black', horizontalalignment='center')
            pylab.text(0.95 * float(audioDuration), yl, '%4.0fHz' % (cfbmin), fontsize=12, color='black', horizontalalignment='center')
            x = 0.95 * float(audioDuration)
            l = Line2D([x, x], [yh - 1, yl + 2], linewidth=2, color='black', linestyle='--')
            ax.add_line(l)

    return (neuronGroup, features)

def testReservoirResponse(structure, audioFilename, cbf):

    extractParams, inputModelParams = defaultparams.getDefaultExtractParams()

    # Use only a single modulation channel at 8Hz
    extractParams['mfbmin'] = 8 * Hz
    extractParams['mfbmax'] = 8 * Hz
    extractParams['mfbbw'] = 12 * Hz
    extractParams['nbModulationFilters'] = 1

    # Increase the amount of noise
    inputModelParams['sigma'] = 0.1

    N, connectivity, connectParams = defaultparams.getDefaultConnectivityParams(structure)
    neuronModel, modelParams = defaultparams.getDefaultMicrocircuitModelParams()
    microcircuit = Microcircuit(N, neuronModel, modelParams, connectivity, connectParams)
    network = Network(microcircuit.microcircuit)
    network.add(microcircuit.connections)

    # Training
    audioDuration = 5 * second
    offset = 0 * second

    print "Training microcircuit for cbf %f..." % (cbf)
    method, trainParams = defaultparams.getDefaultTrainingParams(modelParams)
    trainParams['refractory'] = modelParams['refractory']
    tuner = LTSOCP(microcircuit, trainParams)
    network.add(tuner)

    # Input
    input, inputConnections, features = defaultparams.createDefaultTimitInput(audioDuration, offset, microcircuit, tuner, extractParams, inputModelParams)
    network.add(input)
    network.add(inputConnections)
    figures = [manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    figures[-1].savefig('results/latex/timit_modulation_patterns_train.eps')

    # Monitor
    steps = int(duration / defaultclock.dt)
    logMonitor = LTSOCP_Monitor(tuner, refresh=25 * ms, preallocate=steps, clock=EventClock(dt=25 * ms))
    network.add(logMonitor)
    spikeMonitor = RealtimeSpikeMonitor(microcircuit.microcircuit, refresh=50 * ms, showlast=500 * ms, unit=second)
    network.add(spikeMonitor)
    spikeMonitor = RealtimeSpikeMonitor(input, refresh=50 * ms, showlast=500 * ms, unit=second)
    network.add(spikeMonitor)

    # Simulation
    network.reinit(states=False)
    network.run(duration)

    pylab.close(logMonitor.fig)
    pylab.close(spikeMonitor.fig)

    # Testing
    print "Testing microcircuit with target %f on speech signal..." % (target)
    network = Network(microcircuit.microcircuit)
    network.add(microcircuit.connections)

    # Input
    audioDuration = 700 * ms
    offset = 8.1 * second
    inputModelParams['threshold'] = 0.65
    inputTest, features = auditoryFeatureExtractionModel(audioFilename, audioDuration, offset, extractParams, inputModelParams, showPlot=True)
    inputConnectionsTest = Connection(inputTest, microcircuit.microcircuit, state='v', microcircuit=microcircuit)
    for i in range(inputConnections.W.shape[0]):
        row = inputConnections.W[i, :]
        for j, w in izip(row.ind, row):
            inputConnectionsTest.W[i, j] = w
    network.add(inputTest)
    network.add(inputConnectionsTest)

    figures = [manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    figures[-1].savefig('results/latex/timit_modulation_patterns_test.eps')

    # Monitors
    resSpikeMonitor = RealtimeSpikeMonitor(microcircuit.microcircuit, refresh=audioDuration - defaultclock.dt, showlast=audioDuration, unit=second)
    network.add(resSpikeMonitor)
    inputSpikeMonitor = RealtimeSpikeMonitor(inputTest, refresh=audioDuration - defaultclock.dt, showlast=audioDuration, unit=second)
    network.add(inputSpikeMonitor)

    # Simulation
    microcircuit.microcircuit.v[:] = microcircuit.microcircuit.v0[:]
    microcircuit.microcircuit.vt[:] = microcircuit.microcircuit.vt0[:]
    network.reinit(states=False)
    network.run(audioDuration)

    resSpikeMonitor.fig.savefig('results/latex/timit_spikegram_reservoir_test.eps')
    inputSpikeMonitor.fig.savefig('results/latex/timit_spikegram_input_test.eps')

def testFeatureExtraction(audioFilename, audioDuration):

    defaultclock.dt = defaultparams.getDefaultSimulationTimeStep()

    # Auditory model providing spike trains
    offset = 0 * ms

    extractParams, inputModelParams = defaultparams.getDefaultExtractParams()
    auditoryFeatureExtractionModel(audioFilename, audioDuration, offset, extractParams, inputModelParams, showPlot=True)

    figures = [manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    figures[-1].savefig('results/latex/timit_modulation_patterns.eps')


def testReservoirRegime(structure, audioDuration):

    # Choose the duration of the training
    targetList = [0.5, 1.0, 1.5]
    for n in range(len(targetList)):

        # Microcircuit
        print 'Generating microcircuit for target %f...' % (targetList[n])
        N, connectivity, connectParams = defaultparams.getDefaultConnectivityParams(structure)
        neuronModel, modelParams = defaultparams.getDefaultMicrocircuitModelParams()
        microcircuit = Microcircuit(N, neuronModel, modelParams, connectivity, connectParams)
        network = Network(microcircuit.microcircuit)
        network.add(microcircuit.connections)

        # Learning rule
        print "Training microcircuit for target %f..." % (targetList[n])
        method, trainParams = defaultparams.getDefaultTrainingParams()
        trainParams['alpha'] = targetList[n]
        trainParams['refractory'] = modelParams['refractory']
        tuner = LTSOCP(microcircuit, trainParams)
        network.add(tuner)

        # Input
        input, inputConnections = defaultparams.createDefaultPoissonInput(microcircuit, tuner)
        network.add(input)
        network.add(inputConnections)

        # Monitor
        steps = int(audioDuration / defaultclock.dt)
        logMonitor = LTSOCP_Monitor(tuner, refresh=25 * ms, preallocate=steps, clock=EventClock(dt=25 * ms))
        network.add(logMonitor)
        spikeMonitor = RealtimeSpikeMonitor(microcircuit.microcircuit, refresh=50 * ms, showlast=300 * ms, unit=second)
        network.add(spikeMonitor)

        # Simulation
        network.reinit(states=False)
        network.run(audioDuration)

        filename = 'results/latex/spikegram_timit_target_%1.1f_%s' % (targetList[n], structure)
        filename = filename.replace('.', '_') + '.eps'
        spikeMonitor.fig.savefig(filename)

        pylab.close(logMonitor.fig)


def testTargetConvergence(structure, audioFilename, audioDuration):

    # Choose the duration of the training
    offset = 0 * second

    fig = pylab.figure(facecolor='white', figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Average output contributions')
    ax.grid(True)

    targetList = [0.6, 0.8, 1.0, 1.2, 1.4]
    linestyleList = ['-', '--', '-.', ':', '-']
    linewidthList = [1, 1, 1, 2, 1]
    colorList = ['grey', 'black', 'black', 'black', 'black']
    for n in range(len(targetList) - 1, -1, -1):

        # Microcircuit
        print 'Generating microcircuit for target %f...' % (targetList[n])
        N, connectivity, connectParams = defaultparams.getDefaultConnectivityParams(structure)
        neuronModel, modelParams = defaultparams.getDefaultMicrocircuitModelParams()
        microcircuit = Microcircuit(N, neuronModel, modelParams, connectivity, connectParams)
        network = Network(microcircuit.microcircuit)
        network.add(microcircuit.connections)

        # Learning rule
        print "Training microcircuit for target %f..." % (targetList[n])
        method, trainParams = defaultparams.getDefaultTrainingParams()
        trainParams['alpha'] = targetList[n]
        trainParams['refractory'] = modelParams['refractory']
        tuner = LTSOCP(microcircuit, trainParams)
        network.add(tuner)

        # Input
        extractParams, inputModelParams = defaultparams.getDefaultExtractParams()
        input, inputConnections, features = defaultparams.createDefaultTimitInput(audioDuration, offset, microcircuit, tuner, extractParams, inputModelParams, False)
        network.add(input)
        network.add(inputConnections)

        # Monitors
        steps = int(audioDuration / defaultclock.dt)
        logMonitor = LTSOCP_Monitor(tuner, refresh=25 * ms, preallocate=steps, clock=EventClock(dt=25 * ms))
        network.add(logMonitor)
        #spikeMonitor = RealtimeSpikeMonitor(microcircuit.microcircuit, refresh=50 * ms, showlast=1000 * ms)
        #network.add(spikeMonitor)

        # Simulation
        network.reinit(states=False)
        network.run(audioDuration)

        pylab.close(logMonitor.fig)
        t = logMonitor.data[0:logMonitor.dataCounter, 3]
        ax.plot(t, logMonitor.data[0:logMonitor.dataCounter, 0], color=colorList[n], linestyle=linestyleList[n], linewidth=linewidthList[n], label='target = %1.1f' % (targetList[n]))
        fig.canvas.draw()

    # Visualization
    ax.legend(loc='lower right', ncol=2)
    fig.canvas.draw()
    fig.savefig('results/latex/convergence_timit_%s.eps' % (structure))

def testConvergenceFluctuation(structure, audioFilename):

    # Choose the duration of the training
    audioDuration = 30 * second
    offset = 0 * second

    # Microcircuit
    print 'Generating microcircuit...'
    N, connectivity, connectParams = defaultparams.getDefaultConnectivityParams(structure)
    neuronModel, modelParams = defaultparams.getDefaultMicrocircuitModelParams()
    microcircuit = Microcircuit(N, neuronModel, modelParams, connectivity, connectParams)
    network = Network(microcircuit.microcircuit)
    network.add(microcircuit.connections)

    # Learning rule
    target = 1.0
    print "Training microcircuit for target %f..." % (target)
    method, trainParams = defaultparams.getDefaultTrainingParams()
    trainParams['alpha'] = target
    trainParams['refractory'] = modelParams['refractory']
    tuner = LTSOCP(microcircuit, trainParams)
    network.add(tuner)

    # Input
    extractParams, inputModelParams = defaultparams.getDefaultExtractParams()
    input, inputConnections, features = defaultparams.createDefaultTimitInput(audioDuration, offset, microcircuit, tuner, extractParams, inputModelParams)
    network.add(input)
    network.add(inputConnections)

    # Monitors
    steps = int(audioDuration / defaultclock.dt)
    logMonitor = LTSOCP_AudioMonitor(tuner, features, audioDuration, refresh=200 * ms, preallocate=steps, clock=EventClock(dt=25 * ms))
    network.add(logMonitor)

    # Simulation
    network.reinit(states=False)
    network.run(audioDuration)

    logMonitor.fig.savefig('results/latex/convergence_timit_detailed_%s.eps' % (structure))


if __name__ == '__main__':

    numpy.random.seed(0)
    numpy.seterr(all='raise')
    pylab.ion()

    # Generate required audio files if not existing
    audioFilename = 'data/timitAudio.wav'
    regenerateAudioFile = False
    if not os.path.exists(audioFilename) or regenerateAudioFile:
        timitBasePath = '/home/simon/workspace/master/bros2405/workspace/data/corpus/TIMIT/TIMIT/TRAIN'

        # Generate or load input audio signal from TIMIT database
        fs = 8 * kHz
        buffer = concatenateTimitFiles(timitBasePath, fs=fs, silenceTime=500 * ms, totalTime=60 * second)

        # Saving the file to disk
        print 'Saving output file to disk: %s ...' % (audioFilename)
        # Buffer needs to be converted from float to 16bit integer values
        buffer = buffer * 32767
        buffer = buffer.astype(numpy.int16)
        wavfile.write(audioFilename, fs, buffer)
        print 'all done.'


    structure = 'small-world'
    #testReservoirRegime(structure, 5 * second)
    testConvergenceFluctuation(structure, audioFilename)
    #testTargetConvergence(structure, audioFilename, 10 * second)

    #testReservoirResponse(audioFilename)

    # Generate feature extraction examples and input spike trains 
    #testFeatureExtraction(audioFilename, 15 * second)

    print 'All done.'
    pylab.ioff()
    pylab.show()
    pylab.close()
