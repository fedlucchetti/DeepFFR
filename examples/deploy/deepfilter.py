# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:48:46 2021

@author: DELPAU
"""

from deepffr import DeepFilter, Utils
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

df = DeepFilter.DeepFilter()

_utils = Utils.Utils()
fs        = 24414
Nt        = 2048
dt        = 1/fs
t         = dt*np.array([x for x in range(Nt)])
ton       = round(0.01  *fs)              # convert to #samples
toff      = round(0.070 *fs)
risefall  = round(0.01  *fs)              # convert to #samples
frequency = 220
SNR       = 10

# convert to #samples
args  = {"NSamples":Nt,'sampling':fs,'frequency':frequency,"onset":ton,"offset":toff,"risefall":risefall,\
                                    "SNR":SNR,"phase0":0,"modulator":None}
sinusoid, envelope, phase, noise, t10percent, gate = _utils.insilico_EFR(args)
outpV = df.apply_filter(sinusoid)

plt.figure()
plt.plot(t,sinusoid)
plt.plot(t,outpV)
plt.show()
