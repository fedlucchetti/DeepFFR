import numpy as np
from numpy.random import random as rand
from random import gauss
import json, os, sys, math
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.signal import hilbert





class utils():
    def __init__(self):
        self.fs        = 24414.0
        self.dt        = 1/self.fs
        self.Nt        = 2048
        self.Nf        = int(self.Nt/2)
        self.df        = self.fs/self.Nt
        self.t         = self.dt*np.array([x for x in range(self.Nt)])
        self.f         = self.df*np.array([x for x in range(self.Nf)])
        self.initwaveforms = {'Channel-V':{'EFR':[],'EFR**':[],'EFR***':[],'CDT':[],'CDT*':[],'F1':[],'F2':[],'ABR':[],'Noise':[]},\
                     'Channel-H':{'EFR':[],'EFR**':[],'EFR***':[],'CDT':[],'CDT*':[],'F1':[],'F2':[],'ABR':[],'Noise':[]} }


    def halfwaveRectifier(self,waveform,threshold=0):
        for idy in range(len(waveform)):
            if waveform[idy]<threshold:waveform[idy]=threshold
        return waveform

    def envelope_function(self,args):

        ton      = int(args["onset"])
        toff     = int(args["offset"])
        risefall = int(args["risefall"])
        duration = toff-ton
        # sample time scale

        t          = np.linspace(0,self.Nt,self.Nt,endpoint=True)
        gaussrise  = np.exp(-0.5*np.square((t-risefall)/(risefall/4)))
        gaussrise  = gaussrise/np.amax(gaussrise)

        t10percent = np.argwhere(gaussrise > 0.1)
        t10percent = int(t10percent[0])
        rise       = gaussrise[0:np.argmax(gaussrise)-1]
        t_rise     = risefall - t10percent

        x = np.array([0 ,ton-t10percent])
        x = np.append(x,np.linspace(max(x)+1,ton,20))
        x = np.append(x,np.linspace(max(x)+1,ton+t_rise,20))
        x = np.append(x,np.sort(np.random.uniform(low=max(x)+1,high=toff-t_rise,size=(10,))))
        x = np.append(x,np.linspace(max(x)+1,toff,20))
        x = np.append(x,np.linspace(max(x)+1,toff+t_rise,20))
        x = np.append(x,np.array([max(x)+1,self.Nt]))

        # sample amplitude scale
                         # only retain the rising part of the gaussian
        y          = np.array([0 ,0])                                                               # initial pause before onset
        _y         = rise[np.around(np.linspace(0,len(rise)-1-int(rand()*0.5*len(rise)),40)).astype(int)]        # rise function with random peak
        y          = np.append(y,_y)
        y          = np.append(y,np.random.uniform(low=0.1,high=1,size = (10,)))                    # sample random intervall between rise and fall
        _y         = rise[np.around(np.linspace(0,len(rise)-1-int(rand()*0.5*len(rise)),40)).astype(int)]
        y          = np.append(y,_y[::-1])                                                          # fall function
        y          = np.append(y,[0 ,0])                                                            # final break after offset

        #interpolate samples
        xnew       = np.linspace(0,self.Nt,self.Nt,endpoint=True)
        ynew       = np.interp(xnew, x, y)

        # apply smoothing functions
        ynew       = self.halfwaveRectifier(ynew,0.0)
        ynew       = signal.savgol_filter(ynew, 101, 10)

        # normalize
        yenvelope = ynew/max(ynew)
        return yenvelope, t10percent

    def pink_noise(self,k=1):
        spectrum     = k*1/self.f[4::]
        inv_spectrum = np.fft.ifft(spectrum)
        gaussnoise   = [gauss(0.0, 1) for i in range(self.Nt)]
        #gaussnoise = gaussnoise/max(gaussnoise)
        noise        = np.convolve(inv_spectrum,gaussnoise)
        noise        = noise.real
        noise        = self.filter(noise[0:self.Nt],1,50,1,"bandstop")            # bandstop for evoked potenial Applications
        noise        = noise-np.mean(noise)
        return noise

    def get_waveform(self,json_path,waveform_name=None,channel='V'):
        if waveform_name == None:
            channel       = "Channel-V"
            SC = "EFR"
        else:
            channel       = "Channel-"+channel
            SC = waveform_name
        with open(json_path) as data_file: data = json.load(data_file)
        waveform   = np.array(data["FFR"][channel][SC]["AVG"]["Waveform"])
        # for sample,amplitude in enumerate(waveform):
        #     print(amplitude)
        #     waveform[sample] = float(amplitude.replace(',','.'))
        ton        = data["FFR"][channel][SC]["Analysis"]["Latency"]
        length     = data["FFR"][channel][SC]["Analysis"]["Lenght"]
        ton        = float(ton.replace(',','.'))
        toff       = ton+float(length.replace(',','.'))
        return waveform, ton, toff

    def filter(self,waveform,fmin=1,fmax=50,order=4,type='bandpass'):
        fmin           = (math.ceil(fmin))/(self.fs/2)
        fmax           = (fmax)/(self.fs/2)
        b, a           = signal.butter(order, [fmin,fmax], type)
        filtwaveform   = signal.filtfilt(b, a, waveform, padlen=150)
        return filtwaveform

    def measure_onset(self,envelope,frequency,theta=0.1):
        on=[]
        window = int(1/frequency*self.fs)
        count,score=0,0
        probability=[]
        for sample,idy in enumerate(envelope):
            if idy>theta:
                count+=1
                try:
                    score+=(envelope[sample+1]-envelope[sample])
                except:pass
                if count>window:
                    on.append(sample-window)
                    probability.append(score)
                    count,score=0,0
                    # break
                else: continue
            else: count,score=0,0

        # print(probability)
        on=on[0:3]
        probability=probability[0:3]
        probability=softmax(probability)
        # print(on,probability)
        return on,probability

    def gauge_noise(self,args,sinusoid,noise):
        f                = float(args["frequency"])
        noiselevel       = np.std(sinusoid)*pow(10,-args["SNR"]/20)
        noise            = noise/np.std(noise)
        signal           = sinusoid+(noise*noiselevel)
        return signal, noise

    def env_to_square(self,ton,toff):
        gate                     = np.zeros([self.Nt])
        gate[ton:toff]           = np.ones([toff-ton])
        return gate

    def instantaneous_phase(self,signal):
        analytical = hilbert(signal)
        phase = np.unwrap(np.angle(analytical))
        return phase

    def instantaneous_diff_frequency(self,signal):
        plv   = np.zeros([signal.size])
        phase = self.instantaneous_phase(signal)
        plv[0:phase.size-1]   = np.diff(phase)
        return plv

    def insilico_EFR(self,args):
        fs       = args["sampling"]
        f        = args["frequency"]                                 # sinusoid frequency [Hz]
        phi0     = args["phase0"]                                    # starting phase
        fc       = f+2*np.sin(2*np.pi*100*self.t)                    # frequency fluctuation +-2hz
        phase    = 2*np.pi*fc*self.t+phi0
        carrier  = np.sin(phase)
        ton      = int(args["onset"])
        toff     = int(args["offset"])

        # create gating/envelope function

        try:
            bool_modulator = args["modulator"]
            if args["modulator"]==None:
                envelope, t10percent = self.envelope_function(args)                      # custom envelope array function
            else:
                envelope = args["modulator"]
        except:
            envelope = np.ones([self.Nt])

        sinusoid = np.multiply(carrier,envelope)

        if args["SNR"]!=None:
            noise           = self.pink_noise()
            sinusoid, noise = self.gauge_noise(args,sinusoid,noise)

        gate = np.zeros([sinusoid.size])
        gaussian = signal.gaussian(int(10/1000/self.dt), std=21)
        gaussian = gaussian/np.amax(gaussian)
        gate[ton-int(len(gaussian)/2):ton+int(len(gaussian)/2)]=gaussian
        gate[toff-int(len(gaussian)/2):toff+int(len(gaussian)/2)]=gaussian
        # gate[ton:toff]=1

        insilico_EFR = sinusoid/np.amax(sinusoid)
        return insilico_EFR, envelope, phase, noise, t10percent, gate

if __name__ == "__main__":
    utils     = utils()
    fs        = 24414
    Nt        = 2048
    ton       = round(0.01  *fs)              # convert to #samples
    toff      = round(0.060 *fs)
    risefall  = round(0.01  *fs)              # convert to #samples
    frequency = 220
    SNR       = -4

    # convert to #samples
    args                             = {"NSamples":Nt,'sampling':fs,'frequency':frequency,"onset":ton,"offset":toff,"risefall":risefall,\
                                        "SNR":SNR,"phase0":0,"modulator":None}
    print(args)
    sinusoid, envelope, phase, noise, t10percent, gate = utils.insilico_EFR(args)

    # compute spectra
    signalSpectrum            = np.abs(np.fft.fft(sinusoid))
    noiseSpectrum             = np.abs(np.fft.fft(noise))
    signal                    = np.multiply(np.sin(phase),envelope)
    phase                     = utils.instantaneous_phase(signal)
    diff_phase                = np.abs(phase - 2*np.pi*frequency*utils.t)
    inst_diff_frequency       = utils.instantaneous_diff_frequency(signal)



    # plot results
    plt.subplot(311)
    plt.plot(utils.t,sinusoid,label='EFR')
    plt.plot(utils.t,envelope,'r--',label='envelope')
    plt.plot([ton/fs,ton/fs],[-1,1],'k',label=("$t_{ON} & t_{Off}$"))
    plt.plot([toff/fs,toff/fs],[-1,1],'k')
    plt.legend()
    plt.title('In silico EFR  f = ' + str(frequency) + ' Hz   ' + 'SNR = ' + str(SNR) +' dB' )
    plt.xlabel("Time [s]")
    plt.ylabel("A.U")

    plt.subplot(312)
    plt.plot(utils.f,signalSpectrum[0:len(utils.f)],label='EFR')
    plt.fill_between(utils.f,0,noiseSpectrum[0:len(utils.f)], color=(0, 0, 1, 0.1),label='Noise')
    plt.xlim(0,1000)
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("A.U")

    plt.subplot(313)
    plt.plot(utils.t,diff_phase/max(diff_phase),label='diff phase')
    plt.plot(utils.t,envelope,'r--',label='envelope')
    # plt.plot(utils.t,inst_diff_frequency/max(inst_diff_frequency),label='plv')
    plt.fill_between(utils.t,0,gate, color=(0, 0, 1, 0.1),label='gate')
    # plt.plot(utils.t,gate,label='plv')


    # plt.xlim(0,1000)
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [rad]")
    plt.show()
