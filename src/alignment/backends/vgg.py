import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os

hideBottom = False
modelType = "vgg16" #vgg16, vgg19
layer = "block3_pool"
square = False

print("Loading VGG")
basemodel = None
if modelType == "vgg16":
    basemodel = VGG16(weights="imagenet")
elif modelType == "vgg19":
    basemodel = VGG19(weights="imagenet")
featureModels = []
for layers in basemodel.layers:
    print(layers.name)
for idx in range(len(basemodel.layers)):
    print(basemodel.get_layer(index=idx).name)

from tensorflow.keras import backend
#backend.clear_session()

sess = tf.keras.backend.get_session()
init = tf.global_variables_initializer()

graph = tf.get_default_graph()
with graph.as_default():
    sess.run(init)
    model = Model(inputs=basemodel.input, outputs=basemodel.get_layer(layer).output)
    model._make_predict_function()
print("Finished loading model")

#network comparison function
def getDiff(a, b):
    diff = 0
    flatA = a.flat
    flatB = b.flat
    #for i in range(len(flatA)):
    #    if square == False:
    #        diff += abs(flatA[i] - flatB[i])
    #else:
    #        diff += abs(flatA[i] - flatB[i]) ** 2
    diff = np.subtract(flatA, flatB)
    diff = np.abs(diff)
    diff = np.sum(diff)
    return diff

def eraseBottom(img):
    for i in range(240, 479):
        for j in range(len(img[0])):
            img[i][j] = [0, 0, 0]

def align(baseimg, img):
    global graph, sess

    print("Begin nn align")
    baseimgcrop = baseimg[:, 136:616]
    baseimgcrop = cv2.resize(baseimgcrop, (224, 224))
    baseimgcrop = img_to_array(baseimgcrop)
    baseimgcrop = np.expand_dims(baseimgcrop, axis=0)
    baseimgcrop = preprocess_input(baseimgcrop)
    with graph.as_default():
        set_session(sess)
        baseimgdescriptor = model.predict(baseimgcrop)
    
    imgcrop = img[:, 136:616]
    imgcrop = cv2.resize(imgcrop, (224, 224))
    imgcrop = img_to_array(imgcrop)
    imgcrop = np.expand_dims(imgcrop, axis=0)
    imgcrop = preprocess_input(imgcrop)
    with graph.as_default():
        set_session(sess)
        imgdescriptor = model.predict(imgcrop)

    bestOffset = -1
    bestOffsetValue = float('inf')
    offsetResults = []
    offsetValues = []

    for offset in range(0, 272):
    #for offset in range(100, 160):
        #if offset % 5 != 0:
        #    continue
        #print(offset)

        offset -= 136
        diff = getDiff(descriptor, baseimgdescriptor)

        #print(offset, diff)
        offsetValues.append(offset)
        offsetResults.append(diff)

        if diff < bestOffsetValue:
            bestOffsetValue = diff
            bestOffset = offset

    if bestOffset < -30 or bestOffset > 30:
        bestOffset = 0
    bestOffset = 0

    print("Best:", bestOffset, bestOffsetValue)

    return bestOffset, bestOffsetValue, offsetResults


def align_t(baseimg, img):
    global graph, sess

    if hideBottom:
        eraseBottom(baseimg)
    print("Begin nn align")
    baseimgcrop = baseimg[:, 136:616]
    baseimgcrop = cv2.resize(baseimgcrop, (224, 224))
    baseimgcrop = img_to_array(baseimgcrop)
    baseimgcrop = np.expand_dims(baseimgcrop, axis=0)
    baseimgcrop = preprocess_input(baseimgcrop)
    with graph.as_default():
        set_session(sess)
        baseimgdescriptor = model.predict(baseimgcrop)
    
    if hideBottom:
            eraseBottom(img)

    bestOffset = -1
    bestOffsetValue = float('inf')
    offsetResults = []
    offsetValues = []

    #for offset in range(0, 272):
    for offset in range(100, 160):
        if offset % 5 != 0:
            continue
        print(offset)
        imgcrop = img[:, offset:offset+480]
        imgcrop = cv2.resize(imgcrop, (224, 224))
        imgcrop = img_to_array(imgcrop)
        imgcrop = np.expand_dims(imgcrop, axis=0)
        imgcrop = preprocess_input(imgcrop)
        with graph.as_default():
            set_session(sess)
            descriptor = model.predict(imgcrop)

        offset -= 136
        diff = getDiff(descriptor, baseimgdescriptor)

        #print(offset, diff)
        offsetValues.append(offset)
        offsetResults.append(diff)

        if diff < bestOffsetValue:
            bestOffsetValue = diff
            bestOffset = offset

    print("Best:", bestOffset, bestOffsetValue)

    return bestOffset, bestOffsetValue, offsetResults

def align(baseimg, img):
    global graph, sess

    if hideBottom:
        eraseBottom(baseimg)
    print("Begin nn align")
    baseimgcrop = baseimg[:, 136:616]
    baseimgcrop = cv2.resize(baseimgcrop, (224, 224))
    baseimgcrop = img_to_array(baseimgcrop)
    baseimgcrop = np.expand_dims(baseimgcrop, axis=0)
    baseimgcrop = preprocess_input(baseimgcrop)
    with graph.as_default():
        set_session(sess)
        baseimgdescriptor = model.predict(baseimgcrop)
    
    if hideBottom:
            eraseBottom(img)

    bestOffset = -1
    bestOffsetValue = float('inf')
    offsetResults = []
    offsetValues = []

    #for offset in range(0, 272):
    for offset in range(100, 160):
        if offset % 5 != 0:
            continue
        print(offset)
        imgcrop = img[:, offset:offset+480]
        imgcrop = cv2.resize(imgcrop, (224, 224))
        imgcrop = img_to_array(imgcrop)
        imgcrop = np.expand_dims(imgcrop, axis=0)
        imgcrop = preprocess_input(imgcrop)
        with graph.as_default():
            set_session(sess)
            descriptor = model.predict(imgcrop)

        offset -= 136
        diff = getDiff(descriptor, baseimgdescriptor)

        #print(offset, diff)
        offsetValues.append(offset)
        offsetResults.append(diff)

        if diff < bestOffsetValue:
            bestOffsetValue = diff
            bestOffset = offset

    print("Best:", bestOffset, bestOffsetValue)

    return bestOffset, bestOffsetValue, offsetResults

