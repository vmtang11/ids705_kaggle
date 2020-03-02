import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import keras

dir_train_images  = './data/training/'
dir_test_images   = './data/testing/'
dir_train_labels  = './data/labels_training.csv'
dir_test_ids      = './data/sample_submission.csv'

def load_data(dir_data, dir_labels, training=True):
    ''' Load each of the image files into memory 

    While this is feasible with a smaller dataset, for larger datasets,
    not all the images would be able to be loaded into memory

    When training=True, the labels are also loaded
    '''
    labels_pd = pd.read_csv(dir_labels)
    ids       = labels_pd.id.values
    data      = []
    for identifier in ids:
        fname     = dir_data + identifier.astype(str) + '.tif'
        image     = mpl.image.imread(fname)
        data.append(image)
    data = np.array(data) # Convert to Numpy array
    
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids

def plot_roc(labels, prediction_scores):
    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction_scores)
    legend_string = 'AUC = {:0.3f}'.format(auc)
   
    plt.plot([0,1],[0,1],'--', color='gray', label='Chance')
    plt.plot(fpr, tpr, label=legend_string)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid('on')
    plt.axis('square')
    plt.legend()
    plt.tight_layout()
    
# load data
X_train, y_train = load_data(dir_train_images, dir_train_labels, training=True)
X_train =(X_train)/255
y_train_cnn = keras.utils.to_categorical(y_train, num_classes = 2)

# cnn
cnn_filename = 'cnn_5d400_1000e_4e4lrdecay_wbest.sav'
cnn = pickle.load(open(cnn_filename, 'rb'))

cnn_scores = cnn.predict(X_train).ravel()
plot_roc(y_train_cnn.ravel(), cnn_scores)

# svm
svm_scores = pd.read_csv('svm_scores.csv')['0']
plot_roc(y_train, svm_scores)
