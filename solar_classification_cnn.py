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
from sklearn.preprocessing import StandardScaler
import cv2

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
        
from skimage.feature import hog

    # relative luminance
#     lum = []
#     for i in data:
#         r_lum = i[:,:,0] * 0.2126
#         g_lum = i[:,:,1] * 0.7152
#         b_lum = i[:,:,2] * 0.0722
#         img_lum = np.stack((r_lum, g_lum, b_lum), axis = -1)
#         lum.append(img_lum)

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
 
from keras import layers, models, utils, preprocessing, optimizers
from keras import backend as K
import tensorflow as tf

def set_classifier():
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    print(K.image_data_format())
    
    model = models.Sequential()
    model.add(layers.Conv2D(128, (10, 10),padding = 'same', activation='relu', input_shape=(101, 101, 3)))
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Conv2D(128, (5,5),padding = 'same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2),strides = 2))
    model.add(layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(layers.MaxPooling2D((1, 1),strides = 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    #model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(128, activation='relu'))
    #model.add(keras.layers.Dense(16, activation='relu'))
    #model.add(keras.layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))
    sgd = optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
              loss='hinge', #set hinge as your loss function
              metrics=['accuracy'])

    
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), input_shape=(101, 101, 3)))
#     model.add(layers.Activation('relu'))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(layers.SpatialDropout2D(0.1, noise_shape=None, seed=None))


#     model.add(layers.Conv2D(32, (3, 3)))
#     model.add(layers.Activation('relu'))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(layers.SpatialDropout2D(0.1, noise_shape=None, seed=None))

#     model.add(layers.Conv2D(64, (3, 3)))
#     model.add(layers.Activation('relu'))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(layers.SpatialDropout2D(0.1, noise_shape=None, seed=None))
    
#     model.add(layers.Conv2D(64, (3, 3)))
#     model.add(layers.Activation('relu'))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(layers.SpatialDropout2D(0.1, noise_shape=None, seed=None))
    
#     model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
#     model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
#     model.add(layers.Dense(64))
#     model.add(layers.Activation('relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(1))
#     model.add(layers.Activation('sigmoid'))
    
#     model.compile(loss='binary_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy']) 
    return model

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
        X_train = X_train.reshape(-1,101,101,3)
        X_val = X_val.reshape(-1,101,101,3)
        y_train          = y[train_index]
        y_train          = utils.to_categorical(y[train_index])
        print(X_train.shape)
        
        # Train the classifier
        X_train_features = preprocess_and_extract_features(X_train)
        
        datagen = preprocessing.image.ImageDataGenerator(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=20,
                    brightness_range = (0.2, 0.8),
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train_features)

        print(X_train_features.shape)
        print(y_train.shape)
        history          = clf.fit(datagen.flow(X_train_features, y_train, batch_size=400), epochs=3)
        
        # Test the classifier on the validation data for this fold
#         X_val_features   = preprocess_and_extract_features(X_val)
        cpred            = clf.predict(X_val)
        
        # Save the predictions for this fold
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
num_training_folds = 5

# Load the data
data, labels = load_data(dir_train_images, dir_train_labels, training=True)

# Choose which classifier to use
clf = set_classifier()

# Perform cross validated performance assessment
# prediction_scores = cv_performance_assessment(data,labels,num_training_folds,clf)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30)
X_train_features = preprocess_and_extract_features(X_train)

history          = clf.fit(X_train_features, y_train, batch_size=200, epochs=10)

X_test   = preprocess_and_extract_features(X_test)
prediction_scores = clf.predict(X_test)

# Compute and plot the ROC curves
plot_roc(y_test, prediction_scores)


'''
Sample script for producing a Kaggle submission
'''

produce_submission = False # Switch this to True when you're ready to create a submission for Kaggle

if produce_submission:
    # Load data, extract features, and train the classifier on the training data
    training_data, training_labels = load_data(dir_train_images, dir_train_labels, training=True)
    training_features              = preprocess_and_extract_features(training_data)
    clf                            = set_classifier()
    clf.fit(training_features,training_labels)

    # Load the test data and test the classifier
    test_data, ids = load_data(dir_test_images, dir_test_ids, training=False)
    test_features  = preprocess_and_extract_features(test_data)
    test_scores    = clf.predict_proba(test_features)[:,1]

    # Save the predictions to a CSV file for upload to Kaggle
    submission_file = pd.DataFrame({'id':    ids,
                                   'score':  test_scores})
    submission_file.to_csv('submission.csv',
                           columns=['id','score'],
                           index=False)

