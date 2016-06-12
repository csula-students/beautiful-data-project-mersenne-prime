from collector import collector
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from pymongo import MongoClient

import matplotlib.pyplot as plt
import numpy as np

#Modified from:
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm,conf_prefix = "", title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(conf_prefix + "confus_matrix.png")

def train_features():
    mongo_client = MongoClient()
    features_db = mongo_client.task2_features
    mfcc_fv = features_db.mfcc_fv
    co = collector()

    clf_dict = {}

    scenes = co.get_scenes()
    for scene in scenes:
        result = np.asarray(co.get_feature_vector_array(sound_scene=scene))
        neg_result = np.asarray(co.get_feature_vector_array(sound_scene=scenes[(scenes.index(scene) + 1) % len(scenes)]))
        clf = svm.SVC(gamma=0.001, C=1500.)
        clf_dict[scene] = clf
        pos_train = []
        neg_train = []
        for mfcc_array in result:
            mean = []
            var = []
            for row in mfcc_array:
                mean.append(np.average(row))
                var.append(np.var(row))
            posMeanVar = mean + var
            pos_train.append(posMeanVar)            
        for mfcc_array in neg_result:
            mean = []
            var = []
            for row in mfcc_array:
                mean.append(np.average(row))
                var.append(np.var(row))
            negMeanVar = mean+var
            neg_train.append(negMeanVar)
            
        #print(mfcc_array[:,0].shape)
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
        #plot_confusion_matrix(cm,conf_prefix=scene,title='assets/' + scene[0].upper() + scene[1:] + ' Scene Confusion matrix')
        #break
            


if __name__ == '__main__':
    train_features()
