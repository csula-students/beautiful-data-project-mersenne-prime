#from scipy.io.wavfile import read
import wave
from subprocess import call
from features import mfcc
import glob
import numpy as np
from numpy.lib.stride_tricks import as_strided

class collector():
    def __init__(self,home_dir="./"):
        self.file_list = []
        self.train_file_map = {}
        self.home_dir=home_dir
        self.init_file_map()

    def collect(self):
        for key,value in self.train_file_map.items():
            wav = wave.open(key)
            rate = wav.getframerate()
            #nframes = wav.getnframes()
            #buf = wav.readframes(nframes)

            #dt = np.dtype(np.int8)
            #dt = dt.newbyteorder('<')
            #sig = np.frombuffer(buf,dtype=dt)

            #numpy doesn't support int24 yet so had to use this:
            #http://stackoverflow.com/questions/12080279/how-do-i-create-a-numpy-dtype-that-includes-24-bit-integers

            rawdatamap = np.memmap(key,dtype=np.dtype('u1'),mode='r')
            usablebytes = rawdatamap.shape[0]-rawdatamap.shape[0]%12
            frames = int(usablebytes/12)
            rawbytes = rawdatamap[:usablebytes]

            realdata = as_strided(rawbytes.view(np.int32), strides=(12,3,), shape=(frames,4))

            sig= realdata&0x00ffffff

            #mfcc_feat needs to be stored in MongoDB, it is a numpy array that is 5999 in length
            #Each Audio file is a scene which is being classified, one of feature vectors
            #used to classfy the scene is
            #the mfcc_feat array
            mfcc_feat = mfcc(sig,rate)
            #print("MFCC: ",mfcc_feat)

    def init_file_map(self):
        files_present = glob.glob(self.home_dir + "audio/*")
        evalu_present = glob.glob(self.home_dir + "evaluation_setup/fold*_evaluate.txt")
        for eval_file in evalu_present:
            f = open(eval_file,'r')
            for line in f:
                file_loc,acoustic_scene_name = line.rstrip('\n').split('\t')
                if file_loc not in self.train_file_map and (self.home_dir + file_loc) in files_present:
                    self.train_file_map[self.home_dir + file_loc] = acoustic_scene_name

        
