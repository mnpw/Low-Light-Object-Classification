import os, re, sys, glob, math, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, BRIEF, plot_matches)
from skimage.feature import hog
from skimage.feature import local_binary_pattern

import cv2
import scipy
import imageio
import scipy.misc, scipy.signal
# from eq import *
from he import *
from dhe import *
from ying import *

def split_k(data, labels, k):

    assert len(data) == len(labels)
    i = np.random.permutation(len(data))

    split_data = np.array_split(data[i], k)
    split_labels = np.array_split(labels[i], k)    
    return split_data, split_labels


def split(data, labels, s):

    assert s<1

    assert len(data) == len(labels)
    l = len(data)

    i = np.random.permutation(len(data))

    data_1, labels_1 = data[i][:int(l*s)], labels[i][:int(l*s)]    
    data_2, labels_2 = data[i][int(l*s):], labels[i][int(l*s):]    

    return [data_1, data_2], [labels_1, labels_2]


def numeric_labels(labels):

    mapping = {}
    unq = np.unique(labels)
    for i in range(len(unq)):
        mapping[i] = unq[i]
        mapping[unq[i]] = i
    print(mapping)

    return np.array([mapping[i] for i in labels])


def joindata(data_list, labels_list):

    assert len(data_list) > 0
    d = np.array(data_list[0])
    l = np.array(labels_list[0])
    for i in range(1,len(data_list)):
        d = np.concatenate((d, data_list[i]), axis=0)
        l = np.concatenate((l, labels_list[i]), axis=0)  
    return d, l

def get_hog_features(image):
    hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9, 1, 4., 0, 2.0000000000000001e-01, 0, 64)
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(image,winStride,padding,locations)
    return hist

def get_lbp_features(image):
    lbp = local_binary_pattern(image, 24, 3, 'uniform')
    lbp = np.reshape(lbp, lbp.shape[0] * lbp.shape[1])
    return lbp

def get_hsv_image(image):
    hsv = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    return hsv

def get_features(a_lbp, a_hog):
    # print(a_lbp.shape, a_hsv.shape)
    features = np.concatenate((a_lbp, a_hog))
    # print(features.shape)
    return features

def bovw_part_2(data, CLF = None):
    
    feature = []
    # SIFT = SIFTDescriptor(patchSize = 64)
    SIFT = cv2.xfeatures2d.SIFT_create()

    for i in range(len(data)):        
        print(data[i].shape)
        im = np.reshape(data[i], (30, 30, 3))
        print(im.shape)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY), (30, 30))
        # gray = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
        # phog = PHogFeatures()

        # phg = phog.get_features(gray)
        # print(phg.shape)
        # detector.detect(data[i])
        # kp = detector.keypoints
        # kp_len.append(len(kp))
        # extractor.extract(data[i], kp)
        # if extractor.descriptors == []:
        #     feature.append(np.fill(256, 1))
        # else:
        #     feature.append(extractor.descriptors.astype(int))
        descs = daisy(gray, step=10, radius=30, rings=2, histograms=8, normalization='daisy', orientations=8, visualize=False).flatten()
        out1 = SIFT.describe(gray)
        # print("SIFT", len(out1))
        out2 = skimage.feature.hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4),block_norm= 'L2', feature_vector = True)
        # out2 = skimage.feature.hog(gray, orientations=8, pixels_per_cell=(8,8), cells_per_block=(4, 4),block_norm= 'L2', feature_vector = True)
        # display(_)
        # print(out2.shape)
        # print("HOG", len(out2))
        lbp = local_binary_pattern(gray, n_points, radius, METHOD).flatten()
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        # print("lbp ",lbp.shape)
        # patches = image.extract_patches_2d(data[i], (4,4), max_patches=20, random_state=0)
        # tp = []
        # kp_len.append(len(patches))
        # for i in range(len(patches)):
        #     out = skimage.feature.hog(patches[i], orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2), feature_vector = True)
        #     # out = np.reshape(out, (4,-1))
        #     # print(out.shape)
        #     tp.append(out)
        # tp = np.vstack(tp)
        feature.append(np.hstack((descs, hist, out1, out2, get_features(data[i]))))
        # feature.append(np.hstack((out2)))
        # feature.append(np.hstack((descs, hist, out2)))
        # feature.append(np.hstack((descs, out1, out2, get_features(data[i]))))


        # feature.append(np.hstack((descs)))

    return np.array(feature)    


def read_face(size, flag = 0):

    files = [f for f in glob.glob("Dataset/FaceDataset/*/*.pgm*")]
    data = []
    labels = []
    for f in files:
        a = cv2.resize(cv2.imread(f, 0), size, interpolation = cv2.INTER_AREA)
        if flag == 0:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        if flag==1:
            a = he(cv2.cvtColor(a, cv2.COLOR_GRAY2RGB))
        elif flag==2:
            
            a = dhe(cv2.cvtColor(a, cv2.COLOR_GRAY2RGB))
        elif flag==3:
            a = enhance(cv2.cvtColor(a, cv2.COLOR_GRAY2RGB))
        
        #hsv
        a_hsv = get_hsv_image(a)
        a_hsv = np.reshape(a_hsv, (a_hsv.shape[0] * a_hsv.shape[1] * a_hsv.shape[2]))
        
        #hog
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        a_hog = get_hog_features(a)
        a_hog = np.reshape(a_hog, (a_hog.shape[0] * a_hog.shape[1]))
        
        #lbp
        a_lbp = get_lbp_features(a)
        
        #concatenate
        a_features = get_features(a_lbp, a_hsv, a_hog)

        data.append(a.ravel())
    
        i = [m.start() for m in re.finditer('/', f)]
        c = int(f[i[1]+1:i[2]])
        labels.append(c)
    labels = np.stack(np.array(labels))
    data = np.stack(np.array(data))
    return data, labels

def read_exdark(size, force = 0, flag = 0):
    
    exists = os.path.isfile("Dataset/exdark_images_{}.pkl".format(flag))

    # print(force, exists)
    # if not force and exists:
    #     print("Pickled files exist. Using pickled files.")
    #     data = np.load('Dataset/exdark_images.pkl', allow_pickle = True)
    #     labels = np.load('Dataset/exdark_labels.pkl', allow_pickle = True)
    #     return data, labels

    files = [f for f in glob.glob("Dataset/ExDark/*/*")]
    data = []
    labels = []
    for f in files:
        a = cv2.resize(cv2.imread(f, 0), size, interpolation = cv2.INTER_AREA)
        if(flag == 0):
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        if flag==1:
            # a = he(a)
            a = he(cv2.cvtColor(a, cv2.COLOR_GRAY2RGB))
        elif flag==2:
            # a = dhe(a)
            a = dhe(cv2.cvtColor(a, cv2.COLOR_GRAY2RGB))
        elif flag==3:
            # a = enhance(a)
            a = enhance(cv2.cvtColor(a, cv2.COLOR_GRAY2RGB))        
        # print(a.shape)
        # a_bow = bovw_part_2(a)
        # a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        # a_hog = get_hog_features(a)
        # a_hog = np.reshape(a_hog, (a_hog.shape[0] * a_hog.shape[1]))
        
        #lbp
        # a_lbp = get_lbp_features(a)
        
        #concatenate
        # a_features = get_features(a_lbp, a_hog)
        # data.append(a.ravel())
        data.append(a.ravel())
        i = [m.start() for m in re.finditer('/', f)]
        c = f[i[1]+1:i[2]]
        labels.append(c)
    # print(labels)
    labels = np.stack(np.array(labels))
    data = np.stack(np.array(data))
    labels = numeric_labels(labels)
    with open("Dataset/exdark_images_{}.pkl".format(flag),'wb') as f:
        pickle.dump(data, f)
    with open("Dataset/exdark_labels_{}.pkl".format(flag),'wb') as f:
        pickle.dump(labels, f)
    return data, labels


def k_fold(data, labels, k, model):

    data_split, labels_split = split_k(data, labels, k)
    best_clf = None
    scores = [0] * k
    for i in range(k):
        X_train, y_train = joindata(data_split[0:i]+data_split[i+1:], labels_split[0:i]+labels_split[i+1:])
        X_test, y_test = data_split[i], labels_split[i]
        clf = model
        clf.fit(X_train, y_train)
        sc = clf.score(X_test, y_test)
        if (max(scores) < sc):
            best_clf = clf
        scores[i] = clf.score(X_test, y_test)

    print("Scores:", scores)
    print("Mean:", np.mean(scores))
    print("Standard Deviation:", np.std(scores))
    return best_clf


def run_models(data, labels, models):
    
    [X_train, X_test], [y_train, y_test] = split(data, labels, 0.7)

    X_train = bovw_part_2(X_train)
    X_test = bovw_part_2(X_test)

    print("\n\n\n-----Running Models-----")
    for n, m in models:
        print("{}:".format(n))
        clf = m
        clf.fit(X_train, y_train)
        print("Accuracy:", clf.score(X_test, y_test))
        # print("- K-Fold (5 folds)")
        # best_model = k_fold(X_train, y_train, 5, clf)
        # print("- Using best k-fold model")
        # print("Accuracy:", best_model.score(X_test, y_test))
        # print()
