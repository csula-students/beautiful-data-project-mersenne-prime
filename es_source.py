from collector import collector

from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import certifi

class es_source():
    def __init__(self,):
        f = open('es_link.txt','r')
        url_file = f.readline().strip()
        self.es = Elasticsearch([url_file],
                                port=443,
                                use_ssl=True,
                                verify_certs=True,
                                ca_certs=certifi.where())
        self.co = collector()
        self.mfcc_insert_dict={}

    def insert_es_mfcc(self,mfcc_insert_dict):
        self.es.index(index='audioscene',
                      doc_type='mfcc',
                      id=mfcc_insert_dict["id"],
                      body={
                          'scene':mfcc_insert_dict["scene"],
                          'sample':mfcc_insert_dict["sample"],
                          'frame':mfcc_insert_dict["frame_id"],
                          'mfcc':mfcc_insert_dict["mfcc"],
                          'avg':mfcc_insert_dict["avg"],
                          'std':mfcc_insert_dict["std"],
                          'min':mfcc_insert_dict["min"],
                          'max':mfcc_insert_dict["max"]
                      })
        
    def bulk_builder_mfcc(self,array,mfcc_insert_dict):
        builder = {
            "_index":"audioscene",
            "_type":"mfcc",
            "_id":mfcc_insert_dict["id"],
            "_source":{
                'scene':mfcc_insert_dict["scene"],
                'sample':mfcc_insert_dict["sample"],
                'frame':mfcc_insert_dict["frame_id"],
                'mfcc':mfcc_insert_dict["mfcc"],
                'avg':mfcc_insert_dict["avg"],
                'std':mfcc_insert_dict["std"],
                'min':mfcc_insert_dict["min"],
                'max':mfcc_insert_dict["max"]
            }
        }

        array.append(builder)

    def insert_es_classifier(self,mfcc_array,sound_scene,classifier_data):
        pass

    def train_classifier(self,):
        clf_dict = {}
        mfcc_dict = {}

        scenes = self.co.get_scenes()
        print(scenes)
        for scene in scenes:
            result = np.asarray(self.co.get_feature_vector_array(sound_scene=scene))
            neg_result = np.asarray(self.co.get_feature_vector_array(sound_scene=scenes[(scenes.index(scene) + 1) % len(scenes)]))
            clf = svm.SVC(gamma=0.001, C=1500.)
            clf_dict[scene] = clf
            pos_train = []
            neg_train = []
            for sample in result:
                mean = []
                var = []
                for mfcc in sample:
                    avg = np.average(mfcc)
                    variance = np.var(mfcc)
                    mean.append(avg)
                    var.append(variance)
                meanAndVar = mean+var
                pos_train.append(meanAndVar)
            for sample in neg_result:
                mean = []
                var = []
                for mfcc in sample:
                    mean.append(np.average(mfcc))
                    var.append(np.var(mfcc))
                negMeanAndVar = mean+var
                neg_train.append(negMeanAndVar)

            #print(mfcc_array[:,0].shape)
            print(pos_train)
            labels = np.concatenate([np.ones(len(pos_train)),np.zeros(len(neg_train))])
            to_train = pos_train + neg_train
            X_train, X_test, y_train, y_test = train_test_split(to_train,labels,test_size=25)
            cur_clf = clf_dict[scene]
            cur_clf.fit(X_train,y_train)
            output = cur_clf.predict(X_test)
            metrics.accuracy_score(y_test, output)
            target_names = ['Not ' + scene + ' scene','Yes ' + scene +' scene']
            print(classification_report(y_test, output, target_names=target_names))
            cm = confusion_matrix(y_test,output)
            # (Actual No, Predicted No)(True Negative), (Actual No, Predicted Yes)(False Positive)
            print(cm[0])
            # (Acutal Yes, Predicted No)(False Negative), (Acutal Yes, Predicted Yes)(True Positive)
            print(cm[1])

    def source_save_mfcc(self,to_remove=None):
        mfcc_dict = {}
        mfcc_total_id = 1

        scenes = self.co.get_scenes()
        scenes.remove('test_name')
        #if(to_remove != None):
        #    for remove_scene in to_remove:
        #        scenes.remove(remove_scene)
        #        
        #        id_increment = np.asarray(self.co.get_feature_vector_array(sound_scene=remove_scene))
        #
        #        # We need to keep consistent id numbers, so increment the id
        #        for to_incr in id_increment:
        #            for mfcc in to_incr:
        #                mfcc_total_id += 1
                        
        print(scenes)
        for scene in scenes:
            result = np.asarray(self.co.get_feature_vector_array(sound_scene=scene))
            pos_train = []
            sample_id = 0
            for sample in result:
                mean = []
                var = []
                bulk_insert = []
                frame_id = 0
                for mfcc in sample:
                    avg = np.average(mfcc)
                    variance = np.var(mfcc)
                    std = np.std(mfcc)
                    mfcc_min = min(mfcc)
                    mfcc_max = max(mfcc)
                    mean.append(avg)
                    var.append(variance)
                    self.mfcc_insert_dict["scene"] = scene
                    self.mfcc_insert_dict["id"] = mfcc_total_id
                    mfcc_total_id += 1
                    self.mfcc_insert_dict["sample"] = sample_id
                    self.mfcc_insert_dict["frame_id"] = frame_id
                    frame_id += 1
                    self.mfcc_insert_dict["mfcc"] = mfcc.tolist()
                    self.mfcc_insert_dict["avg"] = avg
                    self.mfcc_insert_dict["std"] = std
                    self.mfcc_insert_dict["min"] = mfcc_min
                    self.mfcc_insert_dict["max"] = mfcc_max
                    #self.insert_es_mfcc(self.mfcc_insert_dict)
                    self.bulk_builder_mfcc(bulk_insert,self.mfcc_insert_dict)
                #helpers.bulk(self.es,bulk_insert)
                pos_train.append(mean+var)
                sample_id += 1
            print(sample_id * len(sample))

if __name__ == '__main__':
    already_done = ['car', 'tram', 'library', 'city_center', 'home']
    es_so = es_source()
    #es_so.source_save_mfcc(already_done)
    es_so.train_classifier()
