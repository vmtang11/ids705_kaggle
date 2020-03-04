# -*- coding: utf-8 -*-
'''Sample script for solar array image classification

Author:       Kyle Bradbury
Date:         January 30, 2018
Organization: Duke University Energy Initiative
'''

'''
Import the packages needed for classification
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from skimage.feature import hog
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.close()

'''
Set directory parameters
'''
# Set the directories for the data and the CSV files that contain ids/labels
dir_train_images  = './data/training/'
dir_test_images   = './data/testing/'
dir_train_labels  = './data/labels_training.csv'
dir_test_ids      = './data/sample_submission.csv'

'''
Include the functions used for loading, preprocessing, features extraction, 
classification, and performance evaluation
'''

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

def preprocess_and_extract_features(data):
    '''Preprocess data and extract features
    
    Preprocess: normalize, scale, repair
    Extract features: transformations and dimensionality reduction
    '''
    new_data = []

    for i in data:
        grayimg = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) # .833
                
        new_data.append(grayimg)
        
    new_data = np.array(new_data)
        
    data_hog=[]
    for d in new_data:
         data_hog.append(hog(d,orientations=15, transform_sqrt=True, pixels_per_cell=(16, 16),cells_per_block=(2,2)))

    return data_hog

def set_classifier():
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    return SVC(gamma='scale', probability = True, C = 10)

def cv_performance_assessment(X,y,k,clf):
    '''Cross validated performance assessment
    
    X   = training data
    y   = training labels
    k   = number of folds for cross validation
    clf = classifier to use
    
    Divide the training data into k folds of training and validation data. 
    For each fold the classifier will be trained on the training data and
    tested on the validation data. The classifier prediction scores are 
    aggregated and output
    '''
    # Establish the k folds
    prediction_scores = np.empty(y.shape[0],dtype='object')
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, val_index in kf.split(X, y):
        # Extract the training and validation data for this fold
        X_train, X_val   = X[train_index], X[val_index]
        y_train          = y[train_index]
        clf              = clf.fit(X_train,y_train)
        cpred            = clf.predict_proba(X_test)
        prediction_scores[val_index] = cpred[:,1]
    return prediction_scores

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

'''
Sample script for cross validated performance
'''
# Set parameters for the analysis
# num_training_folds = 10

# Load the data
data, labels = load_data(dir_train_images, dir_train_labels, training=True)
data = preprocess_and_extract_features(data)

data_test, ids_test = load_data(dir_test_images, dir_test_ids, training=False)
data_test = preprocess_and_extract_features(data_test)
        
# pca
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(data)

# Apply transform to both the training set and the test set.
data_hog = scaler.transform(data)
data_hog_test = scaler.transform(data_test)

pca = PCA(.85)
pca.fit(data)
data_hog = pca.transform(data)
data_hog_test = pca.transform(data_test)

'''
Cross validation
'''
# clf = set_classifier()
# prediction_scores = cv_performance_assessment(np.array(data),labels,num_training_folds,clf)
# plot_roc(labels, prediction_scores)

'''
Train Test Split
'''
X_train, X_val, y_train, y_val = train_test_split(data_hog, labels, test_size=0.3, random_state=23)
clf               = clf.fit(X_train,y_train)
prediction_scores = clf.predict_proba(X_val)[:,1]
plot_roc(y_val, prediction_scores)

# save to csv for results
pd.DataFrame(prediction_scores).to_csv("svm_scores.csv")

'''
Sample script for producing a Kaggle submission
'''

produce_submission = False # Switch this to True when you're ready to create a submission for Kaggle

if produce_submission:
    # Load data, extract features, and train the classifier on the training data
    training_data, training_labels = load_data(dir_train_images, dir_train_labels, training=True)
    training_features              = preprocess_and_extract_features(training_data)
    clf                            = set_classifier()
    clf.fit(np.array(data_hog),training_labels)

    # Load the test data and test the classifier
    test_data, ids = load_data(dir_test_images, dir_test_ids, training=False)
    test_features  = preprocess_and_extract_features(test_data)
    test_scores    = clf.predict_proba(np.array(data_test))[:,1]

    # Save the predictions to a CSV file for upload to Kaggle
    submission_file = pd.DataFrame({'id':    ids,
                                   'score':  test_scores})
    submission_file.to_csv('submission.csv',
                           columns=['id','score'],
                           index=False)

'''
For testing image preprocessing
Most methods did not improve AUC
'''

# data, labels = load_data(dir_train_images, dir_train_labels, training=True)

# grayscale 
# grayimg = cv2.cvtColor(data[2], cv2.COLOR_BGR2GRAY)

# binary thresholding 
# ret,bin_thresh_inv = cv2.threshold(grayimg,125,255,cv2.THRESH_BINARY_INV)

# adaptive thresholding
# adaptive_thresh = cv2.adaptiveThreshold(grayimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)

# canny edge detection
# edges = cv2.Canny(grayimg,125,200)

# image blurring
# blur = cv2.GaussianBlur(grayimg,(5,5),0)
# ret3,thresh_blur = cv2.threshold(blur,225,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# image blending
# dst0 = cv2.addWeighted(grayimg,0.7,bin_thresh_inv,0.3,0)
# dst1 = cv2.addWeighted(dst0,0.7,thresh_blur,0.3,0)

# dft = cv2.dft(np.float32(grayimg),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)

# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# rows, cols = grayimg.shape
# crow,ccol = round(rows/2) , round(cols/2)

# # create a mask first, center square is 1, remaining all zeros
# mask = np.zeros((rows,cols,2),np.uint8)
# mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# # apply mask and inverse DFT
# fshift = dft_shift*mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# visualize images
# plt.axis('off')
# plt.imshow(img_back)