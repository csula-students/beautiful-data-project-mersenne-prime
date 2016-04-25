import unittest
import numpy as np
from source import source
from collector import collector

class TestMethods(unittest.TestCase):
    def __init__(self,):
        self.working = []
        self.mongo_client = MongoClient()
        self.features_db = self.mongo_client.task2_features
        self.mfcc_fv = self.features_db.mfcc_fv
        self.so = source()
        self.co = collector()
        
  def test_insert(self):
      array = [1,2,3,4]
      cname = 'test_name'      
      result = self.so.insert_mongo(self.mfcc,np.asarray(array),"fake_file",cname)
      self.assertEqual(result['class']['mfcc_array'],array)
      
  def test_get_feature(self):
      self.co.get_feature_vector_array(sound_scene="test_name",limit_num=1)
      self.assertEqual(result['class']['mfcc_array'],array)


if __name__ == '__main__':
    unittest.main()
