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
import scipy.interpolate

def melspace(low, high, N):
    warpingFunc = (lambda freq: 2595.0 * math.log10(1.0 + freq / 700.0))
    unwarpingFunc = (lambda m: 700.0 * ((10 ** (m / 2595.0)) - 1.0))

    # Perform uniform sampling in the mel-scale
    melLow = warpingFunc(float(low))
    melHigh = warpingFunc(float(high))
    mels = numpy.linspace(melLow, melHigh, N)

    return unwarpingFunc(mels)


_region_data = numpy.array([
    [1, 240, 6.0],
    [240, 550, 4.3869],
    [550, 1280, 2.4629],
    [1280, 3000, 1.4616],
    [3000, 8001, 1.0]
    ])

_curve_data = numpy.array([
    [20, 0],
    [90.5753, 171.167],
    [104.405, 197.227],
    [120.341, 232.189],
    [141.19, 275.972],
    [162.74, 310.934],
    [181.035, 337.155],
    [208.656, 381.019],
    [236.277, 407.16],
    [267.527, 451.105],
    [302.909, 495.05],
    [349.106, 547.816],
    [431.986, 609.161],
    [472.055, 644.366],
    [534.431, 706.116],
    [615.807, 794.491],
    [709.575, 882.867],
    [832.245, 980.064],
    [893.316, 1033.15],
    [1066.44, 1148.08],
    [1185.89, 1236.61],
    [1295.47, 1325.23],
    [1440.57, 1413.77],
    [1659.49, 1546.66],
    [1780.88, 1635.36],
    [2015.67, 1741.62],
    [2241.33, 1839.06],
    [2491.98, 1954.3],
    [2770.37, 2087.35],
    [2972.4, 2211.66],
    [3189.51, 2318.16],
    [3362.33, 2415.85],
    [3737.75, 2557.8],
    [3871.44, 2628.85],
    [4154.87, 2708.65],
    [4458.58, 2806.25],
    [4784.49, 2903.85],
    [4955.36, 2983.81],
    [5411.84, 3116.94],
    [6342.08, 3356.58],
    [6566.49, 3489.95],
    [7000.00, 3500.00],
    [8000.00, 3700.00]
    ])

def universalWarpingFunctionSpace(low, high, N):

    x = _curve_data[:, 0]
    y = _curve_data[:, 1]
    warpingFunc = scipy.interpolate.interp1d(x, y, kind='linear')
    unwarpingFunc = scipy.interpolate.interp1d(y, x, kind='linear')

    # Perform uniform sampling in the warped-scale
    vLow = warpingFunc(low)
    vHigh = warpingFunc(high)
    vs = numpy.linspace(vLow, vHigh, N)

    return unwarpingFunc(vs)

