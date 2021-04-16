import tensorflow as tf
from tensorflow.keras.layers import Flatten,Conv1D,LSTM,MaxPooling1D,Dropout, \
                                    GlobalAveragePooling1D,concatenate,Dense,Input,BatchNormalization, Conv1DTranspose,Lambda, \
                                    ThresholdedReLU,GlobalMaxPool1D
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os, random
from tqdm import tqdm
from scipy.signal import hilbert


import utils
utils = utils.utils()



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

    def architecture4_analytical(self):
        modelname    = "architecture4_analytical"
        input        = 2048
        label        = 2048

        visible  = Input(shape=(input,2))
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

    def onsetNN(self):
        visible  = Input(shape=(self.Nt,2))
        # layer    = Flatten()(visible)
        # layer    = tf.keras.layers.Reshape((4096,1))(visible)
        layer    = Conv1D(filters=256, kernel_size=255,padding='same', activation='relu')(visible)
        # layer     = MaxPooling1D()(layer)
        # layer    = Dropout(0.5)(layer)
        layer    = LSTM(128)(layer)
        # layer    = Conv1D(filters=1, kernel_size=3,padding='same', activation='relu')(visible)
        # layer    = Flatten()(layer)
        layer    = Dense(2048,activation='relu')(layer)

        # layer    = Conv1D(1,activation='relu')(layer)
        # layer    = Conv1D(filters=1, kernel_size=3,padding='same', activation='relu')(layer)
        # layer    = Conv1D(2048,activation='relu')(layer)
        # layer    = ThresholdedReLU(theta=0.5)(layer)
        model    = Model(inputs=visible, outputs=layer)
        model._name="onsetNN"
        return model

    def onsetNN_2(self):
        visible  = Input(shape=(self.Nt,2))
        layer    = Conv1D(filters=1, kernel_size=101, activation='relu')(visible)
        layer    = LSTM(128)(layer)
        layer    = Dense(2048,activation='softmax')(layer)
        layer    = tf.keras.layers.Reshape(target_shape=(2048,1))(layer)
        layerout = tf.math.argmax(layer,axis=1)
        model    = Model(inputs=visible, outputs=layerout)
        model._name="onsetNN_2"
        return model

    def onsetNN_3(self):
        visible  = Input(shape=(self.Nt,1))
        layer    = Conv1D(filters=1, kernel_size=21, activation='relu')(visible)
        layer    = LSTM(256)(layer)
        layerout = Dense(2048,activation='sigmoid')(layer)
        model    = Model(inputs=visible, outputs=layerout)
        model._name="onsetNN_3"
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
        print("Initializing Train Utils Class with default parameters")
        return None


    def load_synth_data(self,frequency,snr):
        root = "/home/ergonium/Downloads/synthetic/EFR"
        root = "./data/synthetic/EFR_f" + str(frequency)+"_SNR" + str(int(snr))
        # root = "./data/synthetic/EFR"

        try:
            path = root + "X.npy"
            X = np.load(path)
        except:
            print(path, " not present")
            X = None
        try:
            path = root + "labels.npy"
            y    = np.load(path)
        except:
            print(path, " not present")
            X = None
        return X, y

    def load_dataset(self,frequency,size,SNR,train_ratio=0.9):
        X, y = self.load_synth_data(frequency,SNR[0])
        for idx in range(1,len(SNR)):
            _X, _y = self.load_synth_data(frequency,SNR[idx])
            print(_X.shape)
            try:
                X = tf.concat([X, _X], 0)
                y = tf.concat([y, _y], 0)
            except: print("skipping SNR = ",snr)

        if size=='all': N = X.shape[0]
        else: N = size
        shuffle_id = np.random.randint(X.shape[0], size=N)
        X          = X.numpy()[shuffle_id,:]
        y          = y.numpy()[shuffle_id,:]

        train_size = round(np.size(X,0)*train_ratio)
        train_X      = X[0:train_size,:]
        train_labels = y[0:train_size,:]
        val_X       = X[train_size::,:]
        val_labels  = y[train_size::,:]
        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_labels))
        self.train_dataset = self.train_dataset.shuffle(train_X.shape[0])
        self.val_dataset   = tf.data.Dataset.from_tensor_slices((val_X,  val_labels))
        self.val_dataset   = self.val_dataset.shuffle(val_X.shape[0])
        # train_labels = labelclass[0:train_size,:]
        # test_labels  = labelclass[train_size::,:]
        return train_X, train_labels, val_X, val_labels

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
                filtered                  = np.multiply(np.sin(phase),modulator)
                phase                     = utils.instantaneous_phase(filtered)
                diff_phase                = np.abs(phase - 2*np.pi*frequency*utils.t)
                X[b,idf,:,0] = efr
                X[b,idf,:,1] = diff_phase
                Y[b,idf,:]   = gate/max(gate)
        X=np.reshape(X,[B*self.frequencies.size,self.Nt,2])
        Y=np.reshape(Y,[B*self.frequencies.size,self.Nt])
        print("-----------------TOTAL DATA--------------------")
        print(X.shape, '---->',Y.shape)
        train_size                 = round(X.shape[0]*self.train_ratio)
        self.train_X, self.train_labels      = X[0:train_size,:,:],Y[0:train_size,:]
        self.val_X,self.val_labels           = X[train_size::,:,:],Y[train_size::,:]
        print("-----------------TRAINING DATA--------------------")
        print(self.train_X.shape, '---->',self.train_labels.shape)
        print("-----------------VALIDATION DATA--------------------")
        print(self.val_X.shape, '---->',self.val_labels.shape)

    def compile(self):
        # tf.nn.sigmoid_cross_entropy_with_logits(
        # tf.losses.MeanSquaredError()
        # tf.keras.metrics.BinaryAccuracy()
        # tf.keras.metrics.MeanSquaredError()
        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           metrics=tf.keras.metrics.MeanSquaredError())

    def train(self):
        # history     = self.model.fit((self.train_X, self.train_labels)           \
        #                             ,batch_size=self.batch_size                  \
        #                             ,epochs=self.epochs                          \
        #                             ,validation_data=(self.val_X,self.val_labels)\load_real_efr
        #                             )
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
        path_to_json = '../data/real/control/'
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
