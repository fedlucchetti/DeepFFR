import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os, random
from tqdm import tqdm
from scipy.signal import hilbert
from tensorflow.keras.models import load_model



import utils, NeuralNet
utils = utils.utils()



class DeepFilter():
    def __init__(self):
        print("Initializing DeepFilter  Class with default parameters")
        self.filtermodel = load_model("results/Models/EFR_Autoencoder_v01.h5",compile=False)
        self.filtermodel.summary()
        self.fs          = 24414
        self.Nt          = 2048

    def set_frequency(self,target_frequency):
        self.target_frequency = target_frequency

    def lowpass_filter(self,waveform):

        pass

    def apply_filter(self,waveform):
        scale       = waveform.amax()
        waveform    = waveform/scale
        filtered    = self.filtermodel.predict(EFRs)
        return filtered*scale


if __name__ == "__main__":
    trutil = NeuralNet.TrainUtils()
    deepfilter = DeepFilter()
    # EFRs, onsets, offsets = trutil.load_real_efr(target_frequency=220)
