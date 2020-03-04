import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import keras
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, roc_auc_score, accuracy_score, confusion_matrix

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
    
# load data
X_train, y_train = load_data(dir_train_images, dir_train_labels, training=True)
X_train =(X_train)/255

indxs = np.arange(0, 1500, 1)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(X_train, y_train, indxs, test_size=0.3, random_state=23)

y_train_cnn = keras.utils.to_categorical(y_train, num_classes = 2)
y_val_cnn = keras.utils.to_categorical(y_val, num_classes = 2)

### ROC Curves ###

# load cnn
cnn_filename = 'cnn_5d400_1000e_4e4lrdecay_wbest.sav'
cnn = pickle.load(open(cnn_filename, 'rb'))

# cnn
cnn_scores = cnn.predict(X_val)[:,1]
cnn_scores_auc = cnn.predict(X_val).ravel()
fpr_cnn, tpr_cnn, t_cnn = roc_curve(y_val_cnn.ravel(), cnn_scores_auc, pos_label=1)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
plt.plot(fpr_cnn, tpr_cnn, color = 'r', label = 'CNN AUC = %0.4f' % (roc_auc_cnn))

optimal_idx_cnn = np.argmax(tpr_cnn - fpr_cnn)
optimal_threshold_cnn = t_cnn[optimal_idx_cnn]
print(optimal_threshold_cnn)

cnn_preds = np.where(cnn_scores >= optimal_threshold_cnn, 1, 0)

# svm
svm_scores = pd.read_csv('svm_scores.csv')['0']
fpr_svm, tpr_svm, t_svm = roc_curve(y_val, svm_scores, pos_label=1) 
roc_auc_svm = auc(fpr_svm, tpr_svm)
plt.plot(fpr_svm, tpr_svm, color = 'b', label = 'SVM AUC = %0.4f' % (roc_auc_svm))

optimal_idx_svm = np.argmax(tpr_svm - fpr_svm)
optimal_threshold_svm = t_svm[optimal_idx_svm]
print(optimal_threshold_svm)

svm_preds = np.where(svm_scores >= optimal_threshold_svm, 1, 0)

# random guess
y_rand = np.random.randint(2, size = y_val.shape)
fpr_r, tpr_r, t_r = roc_curve(y_val, y_rand)
roc_auc_r = auc(fpr_r, tpr_r)
plt.plot(fpr_r, tpr_r, color = 'g', linestyle = '--', label='Chance AUC = %0.4f' % (roc_auc_r))

plt.title("ROC Curves", fontsize=16) 
plt.xlabel("False Positive Rate", fontsize = 13)
plt.ylabel("True Positive Rate", fontsize = 13)
plt.legend(loc = 'lower right')
plt.savefig('roc.png')
plt.show()

### Precision Recall Curves ###

# cnn
prec_cnn, recall_cnn, t_cnn = precision_recall_curve(y_val_cnn.ravel(), cnn_scores_auc)
prec_cnn = np.insert(prec_cnn, 0, 0.37, axis = 0)
recall_cnn = np.insert(recall_cnn, 0, 1, axis = 0)
plt.plot(recall_cnn, prec_cnn, color = 'r', label='CNN')

# svm
prec_svm, recall_svm, t_svm = precision_recall_curve(y_val, svm_scores)
plt.plot(recall_svm, prec_svm, color = 'b', label='SVM')

# random guess
p_rand, r_rand = [.3355, .3355], [0, 1]
plt.plot(r_rand, p_rand, color = 'g', linestyle = '--', label='Chance')

plt.title("Precision Recall Curves", fontsize=16) 
plt.xlabel("Recall", fontsize = 13)
plt.ylabel("Precision", fontsize = 13)
plt.legend(loc = 'lower left')
plt.savefig('pr.png')
plt.show()

########################

### SVM ###
# fn: predict 0 but actually 1
fn_svm = pd.DataFrame(idx_val[(y_val == 1) & (svm_preds.T == 0)])
fn_svm = fn_svm.sort_values(by = 0).values.ravel()

# fp: predict 1 but actually 0
fp_svm = pd.DataFrame(idx_val[(y_val == 0) & (svm_preds.T == 1)])
fp_svm = fp_svm.sort_values(by = 0).values.ravel()

# tp
tp_svm = pd.DataFrame(idx_val[(y_val == 1) & (svm_preds.T == 1)])
tp_svm = tp_svm.sort_values(by = 0).values.ravel()
svm_scores[tp_svm].dropna().sort_values()

# tn
tn_svm = pd.DataFrame(idx_val[(y_val == 0) & (svm_preds.T == 0)])
tn_svm = tn_svm.sort_values(by = 0).values.ravel()
svm_scores[tn_svm].dropna().sort_values()

# Confusion Matrix
pd.crosstab(y_val, svm_preds, rownames=['True'], colnames=['Predicted'], margins = True)

### CNN ###
# fn: predict 0 but actually 1
fn_cnn = pd.DataFrame(idx_val[(y_val == 1) & (cnn_preds.T == 0)])
fn_cnn = fn_cnn.sort_values(by = 0).values.ravel()

# fp: predict 1 but actually 0
fp_cnn = pd.DataFrame(idx_val[(y_val == 0) & (cnn_preds.T == 1)])
fp_cnn = fp_cnn.sort_values(by = 0).values.ravel()

# tp
tp_cnn = pd.DataFrame(idx_val[(y_val == 1) & (cnn_preds.T == 1)])
tp_cnn = tp_cnn.sort_values(by = 0).values.ravel()
pd.Series(cnn_scores)[tp_cnn].dropna().sort_values()

# tn
tn_cnn = pd.DataFrame(idx_val[(y_val == 0) & (cnn_preds.T == 0)])
tn_cnn = tn_cnn.sort_values(by = 0).values.ravel()
pd.Series(cnn_scores)[tn_cnn].dropna().sort_values()

# Confusion Matrix
pd.crosstab(y_val, cnn_preds, rownames=['True'], colnames=['Predicted'], margins = True)
