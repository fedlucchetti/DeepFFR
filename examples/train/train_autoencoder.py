print("-----------------IMPORT LBR--------------------")
import json,  sys, os, random
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus and sys.argv[1]!='full':
    print("Allocating GPU memory")
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], \
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(sys.argv[1]))])
        print("Allocating GPU memory: DONE")
    except RuntimeError as e:
        print(e)
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
trutil.frequencies   = np.arange(110,4000,50)
trutil.SNR_array     = np.arange(-4,3,0.1)
trutil.ton_array     = np.round(np.linspace(0.005,  0.025 , 40, endpoint=True) *  utils.fs)  # convert to #samples
trutil.toff_array    = np.round(np.linspace(0.045 , 0.07,   40, endpoint=True) *  utils.fs)  # convert to #samples
trutil.rise_array    = np.round(np.linspace(0.001,  0.01 ,  40, endpoint=True) *  utils.fs)
B                    = 10
SHOWPLOT             = True
SAVEPLOT             = True
train_ratio          = 0.8
trutil.learning_rate = 0.0001
trutil.batch_size    = 64
trutil.epochs        = 50

print("-----------------GENERATE DATA SET--------------------")
trutil.generate_autoencoder_data_set(B)

print("-----------------ARCHITECTURE--------------------")
trutil.model = nn.autoencoder3()
print("loaded",trutil.model.name)
trutil.model.summary()
print("----------------- TRAIN --------------------")

history = trutil.run_experiment()




path  = "results/Models/EFR_Autoencoder_v03.h5"
print("Saving submodel to ", path)
trutil.model.save(path)
