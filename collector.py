import numpy as np
import pickle
from pymongo import MongoClient

class collector():
    def __init__(self,):
        self.working = []
        self.mongo_client = MongoClient()
        self.features_db = self.mongo_client.task2_features
        self.mfcc_fv = self.features_db.mfcc_fv

    def get_scenes(self):
        res = self.mfcc_fv.distinct("class.scene")
        return res
        
    def get_feature_vector_array(self,sound_scene,limit_num=None):
        feature_array_list = []

        if limit_num == None:
            pickled = self.mfcc_fv.find({"class.scene" : sound_scene})
            for pic in pickled:
                feature_array_list.append( pickle.loads( pic['class']['mfcc_array'] ) )
            return feature_array_list
        else:
            pickled = self.mfcc_fv.find({"class.scene" : sound_scene}).limit(limit_num)
            for pic in pickled:
                feature_array_list.append( pickle.loads( pic['class']['mfcc_array'] ) )
            return feature_array_list
        
