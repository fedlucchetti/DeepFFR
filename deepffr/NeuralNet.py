import tensorflow as tf
from tensorflow.keras.layers import Flatten,Conv1D,LSTM,MaxPooling1D,Dropout, \
                                    GlobalAveragePooling1D,concatenate,Dense,Input,BatchNormalization, Conv1DTranspose,Lambda, \
                                    ThresholdedReLU,GlobalMaxPool1D, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os, random
from tqdm import tqdm
from scipy.signal import hilbert


from deepffr import Utils
utils = Utils.Utils()



class NeuralNet():
    def __init__(self):
        print("Initializing neural net Class with default parameters")
        self.optimizer        = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn          = tf.losses.MeanSquaredError()
        self.metrics          = [tf.keras.metrics.MeanSquaredError(name="MSE")]
        self.Nt    = utils.Nt

        return None

    def architecture4(self):
        modelname    = "architecture2"
        input        = 2048
        label        = 2048

        visible  = Input(shape=(input,1))
        layer    = Conv1D(filters=32, kernel_size=51,padding='same', activation='relu')(visible)
        layer    = Conv1D(filters=64, kernel_size=37,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1D(filters=128, kernel_size=25,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=128, kernel_size=19,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1D(filters=256, kernel_size=11,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=128, kernel_size=7,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1D(filters=64, kernel_size=5,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=32, kernel_size=3,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layerout    = Conv1D(filters=1, kernel_size=21,padding='same', activation='relu')(layer)
        self.model = Model(inputs=visible, outputs=layerout)
        return self.model




    def autoencoder(self):
        visible  = Input(shape=(self.Nt,1))
        layer    = Conv1D(filters=128, kernel_size=101,padding='same', activation='relu')(visible)
        layer    = Conv1D(filters=64, kernel_size=51,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1D(filters=32, kernel_size=5,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=16, kernel_size=5,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=16, kernel_size=3,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1DTranspose(16, kernel_size=3, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(16, kernel_size=3, padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1DTranspose(32, kernel_size=5, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(64, kernel_size=51, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(128, kernel_size=101, padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1D(1, kernel_size=1, padding='same', activation='relu')(layer)
        model    = Model(inputs=visible, outputs=layer)
        model._name="autoencoder"
        return model

    def autoencoder(self):
        visible  = Input(shape=(self.Nt,1))
        layer    = Conv1D(filters=128, kernel_size=101,padding='same', activation='relu')(visible)
        layer    = Conv1D(filters=64, kernel_size=51,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1D(filters=32, kernel_size=5,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=16, kernel_size=5,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=16, kernel_size=3,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1DTranspose(16, kernel_size=3, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(16, kernel_size=3, padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1DTranspose(32, kernel_size=5, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(64, kernel_size=51, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(128, kernel_size=101, padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1D(1, kernel_size=1, padding='same', activation='relu')(layer)
        model    = Model(inputs=visible, outputs=layer)
        model._name="autoencoder"
        return model

    def autoencoder2(self):
        visible  = Input(shape=(self.Nt,1))
        layer    = Conv1D(filters=128, kernel_size=101,padding='same', activation='relu')(visible)
        layer    = Conv1D(filters=64, kernel_size=51,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=16, kernel_size=5,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1DTranspose(16, kernel_size=5, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(64, kernel_size=51, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(128, kernel_size=101, padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = LSTM(64    ,return_sequences=True)(layer)
        layer    = Dropout(0.5                                                           )(layer)
        layer    = TimeDistributed(Dense(1 ,activation='sigmoid'                          ))(layer)
        model    = Model(inputs=visible, outputs=layer)
        model._name="autoencoder2"
        return model

    def autoencoder3(self):
        visible  = Input(shape=(self.Nt,1))
        layer    = Conv1D(filters=128, kernel_size=101,padding='same', activation='relu')(visible)
        layer    = Conv1D(filters=64, kernel_size=51,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=16, kernel_size=5,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = LSTM(64    ,return_sequences=True)(layer)
        layer    = Dropout(0.5                                                           )(layer)
        layer    = TimeDistributed(Dense(1 ,activation='sigmoid'                          ))(layer)
        model    = Model(inputs=visible, outputs=layer)
        model._name="autoencoder3"
        return model

    def onsetNN(self):
        visible  = Input(shape=(self.Nt,1))
        layer    = Conv1D(filters=128, kernel_size=101,padding='same', activation='relu')(visible)
        layer    = Conv1D(filters=64, kernel_size=51,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1D(filters=32, kernel_size=5,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=16, kernel_size=5,padding='same', activation='relu')(layer)
        layer    = Conv1D(filters=16, kernel_size=3,padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1DTranspose(16, kernel_size=3, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(16, kernel_size=3, padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        layer    = Conv1DTranspose(32, kernel_size=5, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(64, kernel_size=51, padding='same', activation='relu')(layer)
        layer    = Conv1DTranspose(128, kernel_size=101, padding='same', activation='relu')(layer)
        layer    = Dropout(0.5)(layer)
        outlayer    = Conv1D(filters=1, kernel_size=3,padding='same', activation='sigmoid')(layer)
        # # layer    = Flatten()(layer)
        # layer    = Dense(2048,activation='relu')(layer)

        # layer    = Conv1D(1,activation='relu')(layer)
        # layer    = Conv1D(filters=1, kernel_size=3,padding='same', activation='relu')(layer)
        # layer    = Conv1D(2048,activation='relu')(layer)
        # layer    = ThresholdedReLU(theta=0.5)(layer)
        model    = Model(inputs=visible, outputs=outlayer)
        model._name="onsetNN"
        return model


class TrainUtils():
    def __init__(self):
        self.model   = None
        self.train_dataset = None
        self.val_dataset   = None
        self.learning_rate = 0.001
        self.trainX        = None
        self.trainY        = None
        self.valX          = None
        self.valY          = None
        self.train_ratio   = 0.8
        self.Nt            = utils.Nt
        self.mode          = "filter"
        print("Initializing Train Utils Class with default parameters")
        return None





    def generate_EFR_data_set(self,B=10):
        X = np.zeros([B,self.frequencies.size,self.Nt])
        Y = np.zeros([B,self.frequencies.size,self.Nt])
        args             = {"NSamples":self.Nt,'sampling':utils.fs,"modulator":None}
        for b in tqdm(range(B)):
            for idf, frequency in enumerate(self.frequencies):
                args["onset"]       = self.ton_array[np.random.randint(0,len(self.ton_array),1)[0]]
                args["offset"]      = self.toff_array[np.random.randint(0,len(self.toff_array),1)[0]]
                args["risefall"]    = self.rise_array[np.random.randint(0,len(self.rise_array),1)[0]]
                args["SNR"]         = self.SNR_array[np.random.randint(0,len(self.SNR_array),1)[0]]
                args["phase0"]      = random.random()*2*np.pi
                args["frequency"]   = frequency
                efr, modulator, phase, _, _, gate    = utils.insilico_EFR(args)
                X[b,idf,:] = efr
                Y[b,idf,:] = modulator
        X=np.reshape(X,[B*self.frequencies.size,self.Nt])
        Y=np.reshape(Y,[B*self.frequencies.size,self.Nt])
        print("-----------------TOTAL DATA--------------------")
        print(X.shape, '---->',Y.shape)
        train_size                 = round(X.shape[0]*self.train_ratio)
        self.train_X, self.train_labels      = X[0:train_size,:],Y[0:train_size,:]
        self.val_X,self.val_labels           = X[train_size::,:],Y[train_size::,:]
        print("-----------------TRAINING DATA--------------------")
        print(self.train_X.shape, '---->',self.train_labels.shape)
        print("-----------------VALIDATION DATA--------------------")
        print(self.val_X.shape, '---->',self.val_labels.shape)

    def generate_analyticalEFR_data_set(self,B=10):
        X = np.zeros([B,self.frequencies.size,self.Nt,2])
        Y = np.zeros([B,self.frequencies.size,self.Nt])
        args             = {"NSamples":self.Nt,'sampling':utils.fs,"modulator":None}
        for b in tqdm(range(B)):
            for idf, frequency in enumerate(self.frequencies):
                args["onset"]       = self.ton_array[np.random.randint(0,len(self.ton_array),1)[0]]
                args["offset"]      = self.toff_array[np.random.randint(0,len(self.toff_array),1)[0]]
                args["risefall"]    = self.rise_array[np.random.randint(0,len(self.rise_array),1)[0]]
                args["SNR"]         = self.SNR_array[np.random.randint(0,len(self.SNR_array),1)[0]]
                args["phase0"]      = random.random()*2*np.pi
                args["frequency"]   = frequency
                efr, modulator, phase, _, _, gate    = utils.insilico_EFR(args)
                X[b,idf,:,0] = efr
                X[b,idf,:,1] = hilbert(efr).imag
                Y[b,idf,:] = modulator
        X=np.reshape(X,[B*self.frequencies.size,self.Nt,2])
        Y=np.reshape(Y,[B*self.frequencies.size,self.Nt])
        print("-----------------TOTAL DATA--------------------")
        print(X.shape, '---->',Y.shape)
        train_size                 = round(X.shape[0]*self.train_ratio)
        self.train_X, self.train_labels      = X[0:train_size,:],Y[0:train_size,:]
        self.val_X,self.val_labels           = X[train_size::,:],Y[train_size::,:]
        print("-----------------TRAINING DATA--------------------")
        print(self.train_X.shape, '---->',self.train_labels.shape)
        print("-----------------VALIDATION DATA--------------------")
        print(self.val_X.shape, '---->',self.val_labels.shape)

    def generate_autoencoder_data_set(self,B=10):
        X = np.zeros([B,self.frequencies.size,self.Nt])
        Y = np.zeros([B,self.frequencies.size,self.Nt])
        args             = {"NSamples":self.Nt,'sampling':utils.fs,"modulator":None}
        for b in tqdm(range(B)):
            for idf, frequency in enumerate(self.frequencies):
                args["onset"]       = self.ton_array[np.random.randint(0,len(self.ton_array),1)[0]]
                args["offset"]      = self.toff_array[np.random.randint(0,len(self.toff_array),1)[0]]
                args["risefall"]    = self.rise_array[np.random.randint(0,len(self.rise_array),1)[0]]
                args["SNR"]         = self.SNR_array[np.random.randint(0,len(self.SNR_array),1)[0]]
                args["phase0"]      = random.random()*2*np.pi
                args["frequency"]   = frequency
                efr, modulator, phase, _, _, gate    = utils.insilico_EFR(args)
                filtered = np.multiply(np.sin(phase),modulator)
                X[b,idf,:] = efr
                Y[b,idf,:] = (filtered+1)/2
        X=np.reshape(X,[B*self.frequencies.size,self.Nt])
        Y=np.reshape(Y,[B*self.frequencies.size,self.Nt])
        print("-----------------TOTAL DATA--------------------")
        print(X.shape, '---->',Y.shape)
        train_size                 = round(X.shape[0]*self.train_ratio)
        self.train_X, self.train_labels      = X[0:train_size,:],Y[0:train_size,:]
        self.val_X,self.val_labels           = X[train_size::,:],Y[train_size::,:]
        print("-----------------TRAINING DATA--------------------")
        print(self.train_X.shape, '---->',self.train_labels.shape)
        print("-----------------VALIDATION DATA--------------------")
        print(self.val_X.shape, '---->',self.val_labels.shape)

    def generate_onset_data_set(self,B=10):
        X = np.zeros([B,self.frequencies.size,self.Nt])
        Y = np.zeros([B,self.frequencies.size,self.Nt])
        args             = {"NSamples":self.Nt,'sampling':utils.fs,"modulator":None}
        for b in tqdm(range(B)):
            for idf, frequency in enumerate(self.frequencies):
                args["onset"]       = self.ton_array[np.random.randint(0,len(self.ton_array),1)[0]]
                args["offset"]      = self.toff_array[np.random.randint(0,len(self.toff_array),1)[0]]
                args["risefall"]    = self.rise_array[np.random.randint(0,len(self.rise_array),1)[0]]
                args["SNR"]         = self.SNR_array[np.random.randint(0,len(self.SNR_array),1)[0]]
                args["phase0"]      = random.random()*2*np.pi
                args["frequency"]   = frequency
                efr, modulator, phase, _, _, gate    = utils.insilico_EFR(args)
                filtered                  = np.multiply(np.sin(phase),modulator)
                phase                     = utils.instantaneous_phase(filtered)
                diff_phase                = np.abs(phase - 2*np.pi*frequency*utils.t)
                diff_frequency            = utils.instantaneous_diff_frequency(filtered,frequency)
                X[b,idf,:]   = modulator*np.sin(diff_frequency)/np.max(modulator*np.sin(diff_frequency))
                Y[b,idf,:]   = gate
        X=np.reshape(X,[B*self.frequencies.size,self.Nt])
        Y=np.reshape(Y,[B*self.frequencies.size,self.Nt])
        print("-----------------TOTAL DATA--------------------")
        print(X.shape, '---->',Y.shape)
        train_size                 = round(X.shape[0]*self.train_ratio)
        self.train_X, self.train_labels      = X[0:train_size,:],Y[0:train_size,:]
        self.val_X,self.val_labels           = X[train_size::,:],Y[train_size::,:]
        print("-----------------TRAINING DATA--------------------")
        print(self.train_X.shape, '---->',self.train_labels.shape)
        print("-----------------VALIDATION DATA--------------------")
        print(self.val_X.shape, '---->',self.val_labels.shape)

    def compile(self):
        if self.mode=='filter':
            self.model.compile(loss=tf.losses.MeanSquaredError(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                               metrics=tf.keras.metrics.MeanSquaredError())
        elif self.mode=='classification':
            self.model.compile(loss=tf.losses.BinaryCrossentropy(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                               metrics=tf.keras.metrics.MeanSquaredError())

    def train(self):
        history     = self.model.fit(x=self.train_X,y=self.train_labels\
                                ,batch_size=self.batch_size            \
                                ,epochs=self.epochs                    \
                                ,validation_data=(self.val_X,self.val_labels)
                                )
        return history

    def run_experiment(self):
        self.compile()
        history = self.train()
        return self.model, history


    def plot_results(self,history,path,showplot=False,saveplot=False):
        history_dict = history.history
        loss         = history_dict['loss']
        val_loss     = history_dict['val_loss']
        epochs       = range(1, len(loss) + 1)
        plt.plot(epochs, loss     ,'k-', label='Training Loss')
        plt.plot(epochs, val_loss ,'r-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if saveplot:
            print("save figure to",path )
            plt.savefig(path)
        if showplot: plt.show()
        return 1


    def save_model_history(self,path,model,history):
        hist_json_path = path
        print("Saving history to ", hist_json_path)
        hist_df = pd.DataFrame(history.history)
        with open(hist_json_path, mode='w') as f:
            hist_df.to_json(f)
        return 1



    def load_real_efr(self,fc):
        path_to_json = '../../data/real/control/'
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        jsons_data = pd.DataFrame(columns=['Ex_number', 'Length', 'Latency', 'Frequency', 'EFR'])
        n_efr = 0
        f     = []
        for index, js in enumerate(json_files):
            with open(os.path.join(path_to_json, js)) as json_file:
                json_text = json.load(json_file)
                frequency = json_text['FFR']['Channel-H']['EFR']['Analysis']['Frequency']
                if frequency!='':
                    frequency = int(frequency)
                    n_efr+=1
                    f         = np.append(f,frequency)
                    ex_number = json_text['MetaData']['Patient']['Number']
                    length    = json_text['FFR']['Channel-H']['EFR']['Analysis']['Lenght']
                    latency   = json_text['FFR']['Channel-H']['EFR']['Analysis']['Latency']
                    efr       = json_text['FFR']['Channel-H']['EFR']['AVG']['Waveform']
                    jsons_data.loc[index] = [ex_number, length, latency, frequency, efr]
        waveforms = []
        onsets    = []
        offsets   = []
        for row in range(len(jsons_data.iloc[:])):
            f = float(jsons_data.iloc[row]["Frequency"])
            if  f < fc+20 and f > fc-20:
                waveforms.append(jsons_data.iloc[row]["EFR"])
                onsets.append(jsons_data.iloc[row]["Latency"])
                offsets.append(float(jsons_data.iloc[row]["Latency"]) + float(jsons_data.iloc[row]["Length"]))
        waveforms = np.array(waveforms).astype(float)
        onsets    = np.array(onsets).astype(float)
        offsets   = np.array(offsets).astype(float)
        return waveforms,onsets,offsets
