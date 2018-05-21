from string import punctuation

import numpy as np
from PIL import Image
import os
import pandas as pd

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix

data_dir = "stage1/"
patients = [f for f in os.listdir(data_dir) if not f.startswith('.')]
labels = pd.read_csv('./stage1_labels.csv', index_col=0)
######################################################################
# functions -- input
######################################################################
def generate_support_vector(patient):
    col = Image.open(data_dir + patient + "/2d.png")
    gray = col.convert('L')

    # Let numpy do the heavy lifting for converting pixels to pure black or white
    bw = np.asarray(gray).copy()
    sum = 0
    cnt = 0
    for r in bw:
        for c in r:
            if c != 255:
                sum += c
                cnt += 1
    avg = sum / cnt
    
    # Pixel range is 0...255
    bw[bw < avg] = 0    # Black
    bw[bw >= avg] = 1   # White
    return bw.flatten()
    #convert rgb vector into supportvector array for SVM
    #supportVector = np.zeros(shape = (len(bw),len(bw[0])))
    #for i in range(len(bw)):
    #    for j in range(len(bw[r])):
    #        if bw[i][j] == 0:
    #            supportVector[i][j] = 1
    #return supportVector


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        measure  -- float, performance measure
    """
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    measure = 0
    if metric == 'accuracy':
        measure = metrics.accuracy_score(y_true, y_label)
    elif metric == 'f1-score':
        measure = metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
        measure = metrics.roc_auc_score(y_true, y_label)
    elif metric == 'precision':
        measure = metrics.precision_score(y_true, y_label)
    else:
        cm = confusion_matrix(y_true, y_label)
        if metric == 'sensitivity':
            measure = cm[0,0]/(cm[0,0]+cm[0,1])
        elif metric == 'specificity':
            measure = cm[1,1]/(cm[1,0]+cm[1,1])
    return measure


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    scores = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        temp_clf = clf
        temp_clf.fit(X_train, y_train)
        y_pred = temp_clf.decision_function(X_test)
        print(y_pred)
        score = performance(y_test,y_pred,metric=metric)
        scores.append(score)

    return np.mean(scores)


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print ('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)

    scores = []
    for c in C_range:
        score = cv_performance(SVC(C=c, kernel='linear'),X,y,kf,metric=metric)
        print(str(c) + ":" + str(score))
        scores.append(score)
    best = np.max(scores)
    return C_range[scores.index(best)]


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C, score -- tuple of floats, optimal parameter values for an RBF-kernel SVM with their corresponding
    """
    
    print ('RBF SVM Hyperparameter Selection based on ' + str(metric) + ':')

    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-3, 3)
    best_score = 0
    best_c = 0
    best_gamma = 0
    for c in C_range:
        for gamma in gamma_range:
            score = cv_performance(SVC(C=c,gamma=gamma,kernel='rbf'),X,y,kf,metric=metric)
            print("c: " + str(c) + ",gamma: " + str(gamma) + " -->" + str(score))
            if best_score < score:
                best_c = c
                best_gamma = gamma
                best_score = score
    return best_c, best_gamma, best_score


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    y_pred = clf.decision_function(X)
    score = performance(y,y_pred,metric=metric)
    return score


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the images and their labels   
    X = np.zeros(shape=(502,480*640))
    y = []
    count = 0
    for num,patient in enumerate(patients):
        print (patient)
        try:
            label = labels.get_value(patient, 'cancer')
            if label == 0:
                label = -1
            sv = generate_support_vector(patient)
            y.append(label)
            X[count] = sv
            count += 1
        except:
            print ("fail: " + patient)
    y = np.asarray(y)
    
    train_x = X[0:450]
    test_x = X[450:]
    train_y = y[0:450]
    test_y = y[450:]
    kf=StratifiedKFold(train_y, n_folds=5)

    best_accuracy = select_param_linear(train_x, train_y, kf,metric='accuracy')
    print ("best: "+str(best_accuracy))
    
    best_accuracy = select_param_rbf(train_x, train_y, kf,metric='accuracy')
    print ("best: "+str(best_accuracy))
    """
    clf_linear = SVC(C=10, kernel='linear')
    clf_linear.fit(train_x, train_y)
    clf_rbf = SVC(C=100,gamma=0.01,kernel='rbf')
    clf_rbf.fit(train_x, train_y)
    
    accuracy = performance_test(clf_linear, test_x, test_y, metric='accuracy')
    f1 = performance_test(clf_linear, test_x, test_y, metric='f1-score')
    auroc = performance_test(clf_linear, test_x, test_y, metric='auroc')
    precision = performance_test(clf_linear, test_x, test_y, metric='precision')
    sensitivity = performance_test(clf_linear, test_x, test_y, metric='sensitivity')
    specificity = performance_test(clf_linear, test_x, test_y, metric='specificity')
    print ("linear SVC performance")
    print (accuracy)
    print (f1)
    print (auroc)
    print (precision)
    print (sensitivity)
    print (specificity)
    
    accuracy = performance_test(clf_rbf, test_x, test_y, metric='accuracy')
    f1 = performance_test(clf_rbf, test_x, test_y, metric='f1-score')
    auroc = performance_test(clf_rbf, test_x, test_y, metric='auroc')
    precision = performance_test(clf_rbf, test_x, test_y, metric='precision')
    sensitivity = performance_test(clf_rbf, test_x, test_y, metric='sensitivity')
    specificity = performance_test(clf_rbf, test_x, test_y, metric='specificity')
    print ("rbf SVC performance")
    print (accuracy)
    print (f1)
    print (auroc)
    print (precision)
    print (sensitivity)
    print (specificity)
    """
    
if __name__ == "__main__" :
    main()
