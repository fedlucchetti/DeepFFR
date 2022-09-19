print("-----------------IMPORT LBR--------------------")
import json,  sys, os, random
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from deepffr import NeuralNet
from deepffr import Utils
print("-----------------INIT CLASSES--------------------")
utils         = Utils.Utils()
nn            = NeuralNet.NeuralNet()
trutil        = NeuralNet.TrainUtils()

print("-----------------INIT CONSTANTS--------------------")
Nt            = utils.Nt
if sys.argv[1]=='all':
    print("Target all frequencies")
    trutil.frequencies   = np.arange(50,4000,50)
else:
    fc = float(sys.argv[1])
    print("Target all",fc, "frequency")
    trutil.frequencies   = np.arange(fc-25,fc+25,1)
trutil.SNR_array     = np.arange(-4,3,0.1)
trutil.ton_array     = np.round(np.linspace(0.005,  0.025 , 40, endpoint=True) *  utils.fs)  # convert to #samples
trutil.toff_array    = np.round(np.linspace(0.045 , 0.07,   40, endpoint=True) *  utils.fs)  # convert to #samples
trutil.rise_array    = np.round(np.linspace(0.001,  0.01 ,  40, endpoint=True) *  utils.fs)
B                    = 50
SHOWPLOT             = True
SAVEPLOT             = True
train_ratio          = 0.8
trutil.learning_rate = 0.0001
trutil.batch_size    = 128
trutil.epochs        = 75

print("-----------------GENERATE DATA SET--------------------")
trutil.generate_EFR_data_set(B)
# trutil.generate_analyticalEFR_data_set(B)



print("-----------------ARCHITECTURE--------------------")
trutil.model = nn.architecture4()
# trutil.model = nn.architecture4_analytical()
print("loaded",trutil.model.name)
trutil.model.summary()
print("----------------- TRAIN --------------------")

history = trutil.run_experiment()



if sys.argv[1]=='all':
    path  = "results/Models/Envelope_model_v01.h5"
else:
    fc = int(sys.argv[1])
    path  = "results/Models/Envelope_model_"+str(fc)+"_v01.h5"
print("Saving submodel to ", path)
trutil.model.save(path)
