import unittest
import numpy as np
from pymongo import MongoClient
from pymongo import results
from source import source
from collector import collector

class TestMethods(unittest.TestCase):
        
    def test_insert(self):
        mongo_client = MongoClient()
        features_db = mongo_client.task2_features
        mfcc_fv = features_db.mfcc_fv
        so = source()
        
        array = [1,2,3,4]
        cname = 'test_name'      
        result = so.insert_mongo(mfcc_fv,np.asarray(array),"fake_file",cname)
        self.assertIsInstance(result,results.InsertOneResult)
      
    def test_get_feature(self):
        mongo_client = MongoClient()
        features_db = mongo_client.task2_features
        mfcc_fv = features_db.mfcc_fv
        co = collector()
        
        array = [1,2,3,4]
        
        result = co.get_feature_vector_array(sound_scene="test_name",limit_num=1)
        self.assertEqual(result[0].tolist(),array)


if __name__ == '__main__':
    unittest.main()
