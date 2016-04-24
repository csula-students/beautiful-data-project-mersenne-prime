#from scipy.io.wavfile import read
import wave
from subprocess import call
from features import mfcc
import glob
import numpy as np
from numpy.lib.stride_tricks import as_strided
from pymongo import MongoClient
from bson.binary import Binary
import pickle

class source():
    def __init__(self,home_dir="./"):
        self.file_list = []
        self.train_file_map = {}
        self.home_dir=home_dir
        self.init_file_map()
        self.mongo_client = MongoClient()
        self.features_db = self.mongo_client.task2_features
        self.mfcc_fv = self.features_db.mfcc_fv

    '''
    Source and munge data, then store into MongoDB.
    Paramters: None
    Return: None
    '''
    def source_save(self):
        for key,value in self.train_file_map.items():
            wav = wave.open(key)
            rate = wav.getframerate()

            #The initial way I was hoping to munge the raw byte data
            #nframes = wav.getnframes()
            #buf = wav.readframes(nframes)

            #data is 24 bits in 3 bytes.  np.int24 does not exist!
            #dt = np.dtype(np.int24)
            #data is in little endian format
            #dt = dt.newbyteorder('<')
            #sig = np.frombuffer(buf,dtype=dt)

            #numpy doesn't support int24 yet so had to use this:
            #http://stackoverflow.com/questions/12080279/how-do-i-create-a-numpy-dtype-that-includes-24-bit-integers

            rawdatamap = np.memmap(key,dtype=np.dtype('u1'),mode='r')
            usablebytes = rawdatamap.shape[0]-rawdatamap.shape[0]%12
            frames = int(usablebytes/12)
            rawbytes = rawdatamap[:usablebytes]

            #This line is the difficult part which required stackoverflow, it makes the data into 32bit data,
            #but because it is actually 24bit data there is included redundant data in the first byte.
            realdata = as_strided(rawbytes.view(np.int32), strides=(12,3,), shape=(frames,2))

            #This ANDs the bits by a byte mask of the last 24bits, to get rid of the redundant data
            sig = realdata&0x00ffffff

            #mfcc is mel frequency cepstral coefficent
            #http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
            #mfcc_feat needs to be stored in MongoDB, it is a numpy array that is 5999 in length
            #Each Audio file is a scene which is being classified, one of feature vectors
            #used to classfy the scene is
            #the mfcc_feat array

            #mfcc will return an array of that is 5999 rows by 13 columns
            #Each column is a feature vector for training the classifier for that audio sample's
            #class (i.e. tram, park)
            #The window length for analysis is 0.025 seconds,
            #the window step between windows is 0.01 seconds.
            #This is the entire array of feature vectors for each audio sample.
            #Additional feature vectors might be added later but this is good for inital tests.
            mfcc_feat = mfcc(sig,samplerate=rate)
            
            #Insert records into mongodb
            #self.insert_mongo(self.mfcc_fv,mfcc_feat,key,value)
            print("MFCC: ",mfcc_feat.shape)            
            

    def init_file_map(self):
        files_present = glob.glob(self.home_dir + "audio/*")
        evalu_present = glob.glob(self.home_dir + "evaluation_setup/fold*_evaluate.txt")
        for eval_file in evalu_present:
            f = open(eval_file,'r')
            for line in f:
                file_loc,acoustic_scene_name = line.rstrip('\n').split('\t')
                if file_loc not in self.train_file_map and (self.home_dir + file_loc) in files_present:
                    self.train_file_map[self.home_dir + file_loc] = acoustic_scene_name

        
    def insert_mongo(self,col,np_array,key,value):
        #First need to serialize mfcc_feat using pickle,
        #then store in a Binary container for mongodb
        store_np = Binary( pickle.dumps( np_array, protocol=2) )
        fv = {"mfcc_array":store_np,
              "class":value,
              "file_name_id":key
        }
        result = col.insert_one(fv).inserted_id
