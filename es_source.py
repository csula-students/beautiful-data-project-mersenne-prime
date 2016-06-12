from collector import collector

from sklearn import svm
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import certifi
import time

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


    def bulk_builder_classifier(self,array,classi_insert_dict):
        builder = {
            "_index":"audioscene",
            "_type":"classifier",
            "_id":classi_insert_dict["id"],
            "_source":{
                'scene':classi_insert_dict["scene"],
                'classifier':classi_insert_dict["classifier"],
                'numsample':classi_insert_dict["numsample"],
                'truepos':classi_insert_dict["truepos"],
                'falsepos':classi_insert_dict["falsepos"],
                'trueneg':classi_insert_dict["trueneg"],
                'falseneg':classi_insert_dict["falseneg"],
                'precision':classi_insert_dict["prec"],
                'recall':classi_insert_dict["rec"],
                'fittimems':classi_insert_dict["fittime"],
                'predicttimems':classi_insert_dict["predictime"]
            }
        }

        array.append(builder)

    def insert_es_classifier(self,mfcc_array,sound_scene,classifier_data):
        pass

    def train_gmm_classifier(self,):
        clf_dict = {}
        mfcc_dict = {}

        scenes = self.co.get_scenes()
        print(scenes)
        for scene in scenes:
            result = np.asarray(self.co.get_feature_vector_array(sound_scene=scene))
            neg_result = np.asarray(self.co.get_feature_vector_array(sound_scene=scenes[(scenes.index(scene) + 1) % len(scenes)]))
            clf = svm.SVC(gamma=0.001, C=1500.,cache_size=500)
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

    def train_classifier(self,):
        id_num = 1
        clf_dict = {}
        mfcc_dict = {}
        classi_insert_dict = {}
        bulk_insert = []
        scenes = self.co.get_scenes()
        scenes.remove('test_name')
        scenes=['car', 'tram', 'bus','residential_area']
        print(scenes)
        for scene in scenes:
            print(scene)
            result = np.asarray(self.co.get_feature_vector_array(sound_scene=scene))
            neg_result = np.asarray(self.co.get_feature_vector_array(sound_scene=scenes[(scenes.index(scene) + 1) % len(scenes)]))
            clf = svm.SVC(gamma=0.001, C=1500.,cache_size=500)
            forest = RandomForestClassifier(n_estimators = 100)
            clf_dict[scene] = clf
            i = 3
            j = 0
            total = len(result)
            test_sizes = np.arange(1,50,3)
            #test_sizes = np.append(test_sizes,[25,28,31])
            #print(test_sizes)
            #print(result.shape)
            while(i<total):
                pos_train = []
                neg_train = []
                fraction = result[0:i]
                #print(fraction.shape)
                for sample in fraction:
                    mean = []
                    var = []
                    for mfcc in sample:
                        avg = np.average(mfcc)
                        variance = np.var(mfcc)
                        mean.append(avg)
                        var.append(variance)
                    meanAndVar = mean+var
                    pos_train.append(meanAndVar)
                neg_fraction = neg_result[0:i]
                #print(neg_fraction.shape)
                for sample in neg_fraction:
                    mean = []
                    var = []
                    for mfcc in sample:
                        mean.append(np.average(mfcc))
                        var.append(np.var(mfcc))
                    negMeanAndVar = mean+var
                    neg_train.append(negMeanAndVar)
                i+=5
                #print(mfcc_array[:,0].shape)
                #print("Sample size: ",len(pos_train))
                num_sample = len(pos_train)
                labels = np.concatenate([np.ones(len(pos_train)),np.zeros(len(neg_train))])
                #print(labels)
                to_train = pos_train + neg_train
                X_train, X_test, y_train, y_test = train_test_split(to_train,labels,test_size=test_sizes[j])
                j+=1
                cur_clf = clf_dict[scene]
                start_forest_time = time.process_time()
                forest.fit(X_train,y_train)
                forest_time = time.process_time() - start_forest_time
                forest_time = forest_time * 1000
                #print("Forest time: ",forest_time)
                start_svm_time = time.process_time()
                cur_clf.fit(X_train,y_train)
                svm_time = time.process_time() - start_svm_time
                svm_time *= 1000
                #print("SVM time: ", svm_time)
                start_svm_time = time.process_time()
                output = cur_clf.predict(X_test)
                svm_predict_time = time.process_time() - start_svm_time
                svm_predict_time *= 1000
                #print("SVM Predict time: ", svm_predict_time)
                metrics.accuracy_score(y_test, output)
                target_names = ['Not ' + scene + ' scene','Yes ' + scene +' scene']
                start_forest_time = time.process_time() 
                forest_output = forest.predict(X_test)
                forest_predict_time = time.process_time() - start_forest_time
                forest_predict_time *= 1000
                #print("Forest Predict time: ", forest_predict_time)
                svm_prec = precision_score(y_test,output)
                svm_rec = recall_score(y_test,output)
                forest_prec = precision_score(y_test,forest_output)
                forest_rec = recall_score(y_test,forest_output)
                #print("SVM report: ",classification_report(y_test, output, target_names=target_names))
                #print("Random Forest report: ", classification_report(y_test, forest_output, target_names=target_names))
                cm = confusion_matrix(y_test,output)
                cm_forest = confusion_matrix(y_test,forest_output)

                # (Actual No, Predicted No)(True Negative), (Actual No, Predicted Yes)(False Positive)
                #print(cm[0])
                # (Acutal Yes, Predicted No)(False Negative), (Acutal Yes, Predicted Yes)(True Positive)
                #print(cm[1])

                classi_insert_dict["id"] = id_num
                classi_insert_dict["scene"] = scene
                classi_insert_dict["classifier"] = "svm"
                classi_insert_dict["numsample"] = num_sample
                classi_insert_dict["truepos"] = int(cm[1][1])
                classi_insert_dict["falsepos"] = int(cm[0][1])
                classi_insert_dict["trueneg"] = int(cm[0][0])
                classi_insert_dict["falseneg"] = int(cm[1][0])
                classi_insert_dict["prec"] = float(svm_prec)
                classi_insert_dict["rec"] = float(svm_rec)
                classi_insert_dict["fittime"] = svm_time
                classi_insert_dict["predictime"] = svm_predict_time
                id_num += 1
                self.bulk_builder_classifier(bulk_insert,classi_insert_dict)
                classi_insert_dict["id"] = id_num
                classi_insert_dict["scene"] = scene
                classi_insert_dict["classifier"] = "randomforest"
                classi_insert_dict["numsample"] = num_sample
                if(len(cm_forest) == 2):
                    classi_insert_dict["truepos"] = int(cm_forest[1][1])
                    classi_insert_dict["falsepos"] = int(cm_forest[0][1])
                    classi_insert_dict["trueneg"] = int(cm_forest[0][0])
                    classi_insert_dict["falseneg"] = int(cm_forest[1][0])
                else:
                    print(cm_forest)
                    classi_insert_dict["truepos"] = 0
                    classi_insert_dict["falsepos"] = 0
                    classi_insert_dict["trueneg"] = 0
                    classi_insert_dict["falseneg"] = 0
                classi_insert_dict["prec"] = float(forest_prec)
                classi_insert_dict["rec"] = float(forest_rec)
                classi_insert_dict["fittime"] = forest_time
                classi_insert_dict["predictime"] = forest_predict_time
                id_num+=1
                self.bulk_builder_classifier(bulk_insert,classi_insert_dict)
                #print(bulk_insert)
        helpers.bulk(self.es,bulk_insert)


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
