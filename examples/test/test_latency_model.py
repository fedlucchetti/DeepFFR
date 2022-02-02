import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.keras.models import load_model
import math

from deepffr import NeuralNet, DeepFilter
from deepffr import Utils

target_frequency = float(sys.argv[1])
utils  = Utils.Utils()
nn     = NeuralNet.NeuralNet()
trutil = NeuralNet.TrainUtils()
df     = DeepFilter.DeepFilter()
fs     = utils.fs

Nt     = utils.Nt
t      = utils.t*1000
dt     = utils.dt


def gaussian(mu,sigma=1):
    y=1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (t - mu)**2 / (2 * sigma**2))
    return y/max(y)

def get_ton(gate):
    ton, toff=[],[]
    for id, t1 in enumerate(gate):
        try:
            if gate[id]==0 and gate[id+1]==1:
                ton.append(id+1)
            elif gate[id]==1 and gate[id+1]==0:
                toff.append(id+1)
        except: pass
    return ton, toff



model = load_model("../train/results/Models/OnsetNetwork_v01.h5",compile=False)
filtermodel = load_model("../train/results/Models/EFR_Autoencoder_v01.h5",compile=False)
model.summary()
EFRs, onsets, offsets = trutil.load_real_efr(target_frequency)
print("Loaded ", EFRs.shape, " recorded EFRs")
for idx,efr in enumerate(EFRs):
    EFRs[idx,:] = df.apply_filter(efr/np.max(efr))

predictions1 = model.predict(EFRs)
predictions2 = model.predict(EFRs[:,::-1])
predictions = 0.5*(predictions1+predictions2[:,::-1])

predictions = np.reshape(predictions,[predictions.shape[0],predictions.shape[1]])
print("predictions shape",predictions.shape )


for idx, prediction in enumerate(predictions):
    # prediction = (prediction+1)/2
    wfm = EFRs[idx,:]
    ongate,offgate = np.zeros([Nt]),np.zeros([Nt])
    diston,distoff  = np.zeros([Nt]),np.zeros([Nt])
    prediction=prediction*2-1
    id_on = np.where(prediction>0.95)[0]
    id_off = np.where(prediction<=0.95)[0]
    ongate[id_on] = 1
    offgate[id_on] = 0
    ton,toff=get_ton(ongate)
    plt.figure(figsize=((40)/2.54,(20)/2.54))
    plt.plot(t,wfm/max(wfm),'k',linewidth=1,alpha=0.5,label='Original')
    plt.plot(t,prediction,'r-',label='Gate')
    plt.plot(t,ongate,'b')
    plt.plot(t,offgate,'k')
    for idt in ton:
        diston+=gaussian(idt*dt*1000)
    for idt in toff:
        distoff+=gaussian(idt*dt*1000)
    plt.fill(t,diston,'g',alpha=0.3)
    plt.fill(t,distoff,'r',alpha=0.3)
    print("prediction shape",prediction.shape )
    print("prediction shape",prediction.reshape([prediction.size,1]).shape )
    plt.grid()
    plt.legend(fontsize=16)
    plt.show()
