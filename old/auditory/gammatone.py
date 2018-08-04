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
from brian.hears import *
from util import *
import math
import numpy
import scipy
from util import *

__all__ = ['Gammatone']

class Gammatone(CombinedFilterbank):

    def __init__(self, source, cf, order=3):

        CombinedFilterbank.__init__(self, source)
        source = self.get_modified_source()

        cf = atleast_1d(cf)

        bandwidth_linear = 10 ** (0.03728 + 0.78563 * log10(cf))
        gammatone = ApproximateGammatone(source, cf, bandwidth_linear, order)
        self.gammatoneFilter = gammatone

        self.set_output(gammatone)

    def calculateFrequencyResponse(self, level, samplerate):
        fig = figure()
        title('Gammatone filterbank frequency response\n (level = %fdB)' % (level))
        hold(True)

        # Loop for each channel
        for n in range(self.output.nchannels):
            # Linear path
            b = self.gammatoneFilter.filt_b[n, :, :]
            a = self.gammatoneFilter.filt_a[n, :, :]
            h, mag = calculateCascadedFrequencyResponse(b, a)
            F = mag

            f = (h * (self.source.samplerate / 2) / numpy.pi) / kHz

            ratio = samplerate / self.source.samplerate
            semilogy(f[0:round((len(f) - 1) * ratio)], F[0:round((len(f) - 1) * ratio)], 'b')

        ylabel('Amplitude [dB]')
        xlabel('Frequency [kHz]')
        ylim(ymin=0.001)
        grid()
