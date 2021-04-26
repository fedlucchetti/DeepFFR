import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from scipy.signal import hilbert
from scipy import signal


import NeuralNet
import Utils

target_frequency = float(sys.argv[1])
utils  = Utils.Utils()
nn     = NeuralNet.NeuralNet()
trutil = NeuralNet.TrainUtils()
fs     = utils.fs

Nt     = utils.Nt
t      = utils.t*1000


envelope_model = load_model("results/Models/Envelope_model.h5",compile=False)
autoencoder    = load_model("results/Models/EFR_Autoencoder_v01.h5",compile=False)


EFRs, onsets, offsets = trutil.load_real_efr(target_frequency)
print("Loaded ", EFRs.shape, " recorded EFRs")
for idx,efr in enumerate(EFRs): EFRs[idx,:]=efr/np.max(efr)



filtEFR     = autoencoder.predict(EFRs)
filtEFR   = np.reshape(filtEFR,[filtEFR.shape[0],filtEFR.shape[1]])
print("filtEFR shape", filtEFR.shape)
for idx,efr in enumerate(filtEFR):
    efr           = efr/np.max(efr)
    efr           = efr*2-1
    efr           = efr - np.mean(efr)
    filtEFR[idx,:]= efr


envelopes   = envelope_model.predict(filtEFR)
envelopes   = np.reshape(envelopes,[envelopes.shape[0],envelopes.shape[1]])
# gates       = onsetmodel.predict(envelopes)
print("envelopes shape",envelopes.shape )

delta_ton  = np.zeros([envelopes.shape[0]])
delta_toff = np.zeros([envelopes.shape[0]])
for idx, envelope in enumerate(envelopes):
    envelope = envelope/max(envelope)

    ton  = onsets[idx]
    toff = offsets[idx]

    plt.figure(figsize=((40)/2.54,(20)/2.54))

    plt.plot(t,filtEFR[idx,:]/max(filtEFR[idx,:]),'r',linewidth=1)
    plt.plot(t,EFRs[idx,:]/max(EFRs[idx,:]),'k',linewidth=1)
    plt.plot(t,envelope,'r-')
    analytic_signal = hilbert(filtEFR[idx,:])
    diff_phase = np.diff(np.abs((np.unwrap( np.angle(analytic_signal))-2*np.pi*target_frequency*t/1000)))
    # plt.plot(t[0:diff_phase.size],diff_phase/max(diff_phase))

    plt.grid()

    ton_pred,probon   = utils.measure_onset(envelope,target_frequency,theta=0.1)
    toff_pred,proboff = utils.measure_onset(envelope[::-1],target_frequency,theta=0.1)
    toff_pred         = 2048*np.ones([len(toff_pred)])-toff_pred
    print(ton_pred,toff_pred)
    # ton_pred,toff_pred = ton_pred*utils.dt*1000,toff_pred*utils.dt*1000
    print(ton_pred,toff_pred)
    for id in range(len(ton_pred)):
        plt.plot([ton_pred[id]*utils.dt*1000,ton_pred[id]*utils.dt*1000],[0,probon[id]],'r-',linewidth=3)
    for id in range(len(toff_pred)):
        plt.plot([toff_pred[id]*utils.dt*1000,toff_pred[id]*utils.dt*1000],[0,proboff[id]],'b-',linewidth=3)
    #     if np.abs(ton-ton_pred)<5 or np.abs(ton-ton_pred)<5:
    #         delta_ton[idx] = np.abs(ton-ton_pred)
    #         delta_toff[idx] =np.abs(toff-toff_pred)
    #     else:
    #         delta_ton[idx] = 0
    #         delta_toff[idx] =0
    # except Exception as e: print(e)
    plt.plot([ton,ton],[0,1],'k-',linewidth=2)
    plt.plot([toff,toff],[0,1],'k-',linewidth=2)
    # plt.plot([envelope,envelope],[0,1],'r--',linewidth=2)
    # plt.legend()
    #
    plt.show()

plt.close()
plt.plot(delta_ton,'g.',label = 'ton')
plt.plot(delta_toff,'r.',label = 'toff')
print("Average error on ton", delta_ton.mean())
print("Average error on ton", delta_toff.mean())
plt.grid()
plt.ylim([0,6])
plt.legend()
plt.show()
