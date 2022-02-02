import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.keras.models import load_model
import math

from deepffr import NeuralNet
from deepffr import Utils

target_frequency = float(sys.argv[1])
utils  = Utils.Utils()
nn     = NeuralNet.NeuralNet()
trutil = NeuralNet.TrainUtils()
fs     = utils.fs

Nt     = utils.Nt
t      = utils.t*1000


model = load_model("../../results/Models/EFR_Autoencoder_v05.h5",compile=False)
model.summary()
EFRs, onsets, offsets = trutil.load_real_efr(target_frequency)
print("Loaded ", EFRs.shape, " recorded EFRs")
for idx,efr in enumerate(EFRs):
    EFRs[idx,:]=efr/np.max(efr)

predictions = model.predict(EFRs)
predictions = np.reshape(predictions,[predictions.shape[0],predictions.shape[1]])
print("predictions shape",predictions.shape )


for idx, prediction in enumerate(predictions):
    prediction = prediction/max(prediction)
    wfm = EFRs[idx,:]
    plt.figure(figsize=((40)/2.54,(20)/2.54))
    plt.plot(t,wfm/max(wfm),'k',linewidth=1,alpha=0.5,label='Original')
    plt.plot(t,prediction*2-1,'r-',label='Autoencoded')
    print("prediction shape",prediction.shape )
    print("prediction shape",prediction.reshape([prediction.size,1]).shape )
    plt.grid()
    plt.legend(fontsize=16)
    plt.show()
