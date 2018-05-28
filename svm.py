from string import punctuation

import numpy as np
import os
import pandas as pd
import dicom
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image

data_dir = "sample_images/"
patients = [f for f in os.listdir(data_dir) if not f.startswith('.')]
labels = pd.read_csv('./stage1_labels.csv', index_col=0)
n_clusters = 40

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
        #print(y_pred)
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
    print(y_pred)
    score = performance(y,y_pred,metric=metric)
    return score

######################################################################
# lung segment
######################################################################
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path) if not s.startswith('.') and not s.endswith('.npy')]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def sample_stack(patient,stack, start_with, show_every):
    for i in range(6):
        ind = start_with + i*show_every
        plt.imsave(data_dir+patient+'/'+str(i+1)+'.png',stack[ind],cmap='gray')

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

#Standardize the pixel values                                                                                  
def make_lungmask(img):
    row_size= img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images  
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)   
    #                       
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.
    eroded = morphology.erosion(thresh_img,np.ones([9,9]))
    dilation = morphology.dilation(eroded,np.ones([4,4]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    #labels = measure.label(eroded)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/10 and B[2]<col_size/10*9:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0
    #
    #  After ju≈st the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation 
    return mask*img

######################################################################
# sift
######################################################################
def scale(histogram):
    sum = np.sum(histogram)
    for i in range(len(histogram)):
        histogram[i] = histogram[i]/sum * 100
    return histogram

def sift(img,sift):
    kp, des = sift.detectAndCompute(img,None)
    vStack = np.array(des)
    for remaining in des:
        vStack = np.vstack((vStack, remaining))
    descriptor_vstack = vStack.copy()

    kmeans_obj = KMeans(n_clusters = n_clusters)
    kmeans_ret = kmeans_obj.fit_predict(descriptor_vstack)

    mega_histogram = np.array(np.zeros(n_clusters))
    l = len(des)
    for j in range(l):
        idx = kmeans_ret[j]
        mega_histogram[idx] += 1
    return scale(mega_histogram)

######################################################################
# gabor filter
######################################################################

# define gabor filter bank with different orientations and at different scales
def build_filters():
    filters = []
    ksize = 9
    #define the range for theta and nu
    for theta in np.arange(0, np.pi, np.pi / 8):
        for nu in np.arange(0, 6*np.pi/4 , np.pi / 4):
            kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, nu, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

#function to convolve the image with the filters
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def garbor(imgg, filters):
    f = np.asarray(filters)
    #initializing the feature vector
    feat = []
    #calculating the local energy for each convolved image
    for j in range(40):
        res = process(imgg, f[j])
        temp = 0
        for p in range(128):
            for q in range(128):
                temp = temp + res[p][q]*res[p][q]
        feat.append(temp)
    #calculating the mean amplitude for each convolved image
    for j in range(40):
        res = process(imgg, f[j])
        temp = 0
        for p in range(128):
            for q in range(128):
                temp = temp + abs(res[p][q])
        feat.append(temp)
    return feat

######################################################################
# main
######################################################################
 
def main():
    np.random.seed(1234)
    index=0
    X = np.zeros(shape=(19,120))
    y = []
    for num,patient in enumerate(patients):
        print (patient)
        label = labels.get_value(patient, 'cancer')
        if label == 0:
            label = -1
        scans = load_scan(data_dir+patient)
        imgs = get_pixels_hu(scans)
        np.save(data_dir+patient+"/%s.npy" % (patient), imgs)
        imgs_to_process = np.load(data_dir+patient+"/%s.npy" % (patient))
        imgs_after_resamp, spacing = resample(imgs_to_process, scans, [1,1,1])
        masked_lung = []
        count = 0
        for img in imgs_after_resamp:
            try:
                masked_lung.append(make_lungmask(img))
                count+=1
            except:
                print(patient + ": masked lung exception")
        sample_stack(patient,masked_lung,int(count/4),int(count/10))
        os.system("rm -rf " + data_dir+patient+"/*.npy")
        SIFT = cv2.xfeatures2d.SIFT_create()
        filters = build_filters()
        sv = [0] * 120
        for i in range(6):
            img = cv2.imread(data_dir+patient+'/'+str(i+1)+'.png')
            imgg = cv2.imread(data_dir+patient+'/'+str(i+1)+'.png',0)
            vec = np.append(sift(img,SIFT),garbor(imgg,filters))
            sv = [sv[j]+vec[j] for j in range(120)]
        sv = [sv[i]/6 for i in range(120)]
        y.append(label)
        X[index] = sv
        index += 1
    y = np.asarray(y)
    train_x = X[0:15]
    test_x = X[15:]
    train_y = y[0:15]
    test_y = y[15:]
    kf=StratifiedKFold(train_y, n_folds=5)

    best_accuracy = select_param_linear(train_x, train_y, kf,metric='accuracy')
    print ("best linear: "+str(best_accuracy))
    best_accuracy = select_param_rbf(train_x, train_y, kf,metric='accuracy')
    print ("best rbf: "+str(best_accuracy))

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
