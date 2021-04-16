import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.keras.models import load_model
import sys
import numpy as np

class DeepFilter():
    def __init__(self):
        print("Initializing DeepFilter  Class with default parameters")
        self.filtermodel = load_model("results/Models/EFR_Autoencoder_v01.h5",compile=False)
        # self.filtermodel.summary()
        self.fs          = 24414
        self.Nt          = 2048

    def set_frequency(self,target_frequency):
        self.target_frequency = target_frequency

    def lowpass_filter(self,waveform):
        pass

    def apply_filter(self,waveform):
        # waveform    = self.lowpass_filter(waveform)
        scale       = waveform.max()
        waveform    = waveform/scale
        filtered    = self.filtermodel.predict(waveform)
        return filtered*scale

if __name__ == "__main__":
    deepfilter = DeepFilter()
    input = np.array(sys.argv[1::])
    flag  = False
    try:
        assert input.size==deepfilter.Nt
        flag==True
    except:
        print("DeepFilter: Input signal size needs to be 2048, input received:", input.size)

    if flag:

        waveform   = input
        waveform = waveform.astype('float')
        waveform = waveform.reshape([2048,1])
        filtered = deepfilter.apply_filter(waveform)
        print(filtered)
