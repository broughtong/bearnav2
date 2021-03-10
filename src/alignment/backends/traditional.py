import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os
import histogram
from pathlib import Path

eraseLower = False
numberFeatures = 500

resultsDir = "./results/"

print("Loading descriptors")
bf = cv2.BFMatcher()
#can run all these without any params for default thresholds:
sift = cv2.xfeatures2d.SIFT_create(nfeatures=numberFeatures, contrastThreshold=0.0075) #default con thresh 0.04
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=40) #default 100
kaze = cv2.KAZE_create(threshold=0.000008) #default 0.001
akaze = cv2.AKAZE_create(threshold=0.000008) # default 0.001
brisk = cv2.BRISK_create(thresh=8) #default 30

#used to add folder structure
firstTime = True

def cropFeatures(kp, des):
    newkp = [None] * numberFeatures
    newdes = [None] * numberFeatures
    i = 0
    while i < numberFeatures:
        bestResponse = -sys.maxsize
        bestId = -1
        for idx in range(len(kp)):
            if kp[idx].response > bestResponse:
                bestReponse = kp[idx].response
                bestId = idx
        newkp[i] = kp[bestId]
        newdes[i] = des[bestId]
        i += 1
        kp = np.delete(kp, bestId, 0)
        des = np.delete(des, bestId, 0)
    newdes = np.array(newdes)
    return newkp, newdes

def getDisplacements(kpsA, kpsB, matches):
    arr = []
    for match in matches:
        fta = kpsA[match.queryIdx].pt[0]
        ftb = kpsB[match.trainIdx].pt[0]
        dist = ftb - fta
        arr.append(dist)
    return arr

def getHistPeak(histograms):
    maxPeak = 0
    maxOffset = 0
    for hist in histograms:
        for key, value in hist.items():
            if value > maxPeak:
                maxPeak = value
                maxOffset = key
    return maxOffset

for files in os.walk("./positioned/"):
    if files[0] == "./positioned/":
        continue
    position = files[0].split("/")[-1]
    print("==============")
    print("Position: " + str(position))
    print("==============")
    filenames = files[2]

    completedPositions = []
    if position in completedPositions:
        print("Position already completed, skipping..")
        continue

    baseImageFilename = ""
    for i in filenames:
        if "A000" in i:
            baseImageFilename = i

    #load base image
    baseimg = cv2.imread(os.path.join("positioned", position, baseImageFilename))
    if eraseLower:
        baseimg = baseimg[:240]
    basesiftkp, basesiftdes = sift.detectAndCompute(baseimg, None)
    basesurfkp, basesurfdes = surf.detectAndCompute(baseimg, None)
    basekazekp, basekazedes = kaze.detectAndCompute(baseimg, None)
    baseakazekp, baseakazedes = akaze.detectAndCompute(baseimg, None)
    basebriskkp, basebriskdes = brisk.detectAndCompute(baseimg, None)

    if len(basesiftkp) < numberFeatures:
        print("Low sift")
    if len(basesurfkp) < numberFeatures:
        print("Low surf")
    if len(basekazekp) < numberFeatures:
        print("Low kaze")
    if len(baseakazekp) < numberFeatures:
        print("Low akaze")
    if len(basebriskkp) < numberFeatures:
        print("Low brisk")

    if len(basesiftkp) > numberFeatures:
        basesiftkp, basesiftdes = cropFeatures(basesiftkp, basesiftdes)
    if len(basesurfkp) > numberFeatures:
        basesurfkp, basesurfdes = cropFeatures(basesurfkp, basesurfdes)
    if len(basekazekp) > numberFeatures:
        basekazekp, basekazedes = cropFeatures(basekazekp, basekazedes)
    if len(baseakazekp) > numberFeatures:
        baseakazekp, baseakazedes = cropFeatures(baseakazekp, baseakazedes)
    if len(basebriskkp) > numberFeatures:
        basebriskkp, basebriskdes = cropFeatures(basebriskkp, basebriskdes)

    for filename in filenames:
        if filename == baseImageFilename:
            continue
        
        img = cv2.imread(os.path.join("positioned", position, filename))
        if eraseLower:
            img = img[:240]
        siftkp, siftdes = sift.detectAndCompute(img, None)
        surfkp, surfdes = surf.detectAndCompute(img, None)
        kazekp, kazedes = kaze.detectAndCompute(img, None)
        akazekp, akazedes = akaze.detectAndCompute(img, None)
        briskkp, briskdes = brisk.detectAndCompute(img, None)

        if len(siftkp) < numberFeatures:
            print("Low sift")
        if len(surfkp) < numberFeatures:
            print("Low surf")
        if len(kazekp) < numberFeatures:
            print("Low kaze")
        if len(akazekp) < numberFeatures:
            print("Low akaze")
        if len(briskkp) < numberFeatures:
            print("Low brisk")

        if len(siftkp) > numberFeatures:
            siftkp, siftdes = cropFeatures(siftkp, siftdes)
        if len(surfkp) > numberFeatures:
            surfkp, surfdes = cropFeatures(surfkp, surfdes)
        if len(kazekp) > numberFeatures:
            kazekp, kazedes = cropFeatures(kazekp, kazedes)
        if len(akazekp) > numberFeatures:
            akazekp, akazedes = cropFeatures(akazekp, akazedes)
        if len(briskkp) > numberFeatures:
            briskkp, briskdes = cropFeatures(briskkp, briskdes)

        siftmatches = bf.match(basesiftdes, siftdes)
        surfmatches = bf.match(basesurfdes, surfdes)
        kazematches = bf.match(basekazedes, kazedes)
        akazematches = bf.match(baseakazedes, akazedes)
        briskmatches = bf.match(basebriskdes, briskdes)

        siftdisplacements = getDisplacements(basesiftkp, siftkp, siftmatches)
        surfdisplacements = getDisplacements(basesurfkp, surfkp, surfmatches)
        kazedisplacements = getDisplacements(basekazekp, kazekp, kazematches)
        akazedisplacements = getDisplacements(baseakazekp, akazekp, akazematches)
        briskdisplacements = getDisplacements(basebriskkp, briskkp, briskmatches)

        sifthists = histogram.variableHist(siftdisplacements, 10)
        surfhists = histogram.variableHist(surfdisplacements, 10)
        kazehists = histogram.variableHist(kazedisplacements, 10)
        akazehists = histogram.variableHist(akazedisplacements, 10)
        briskhists = histogram.variableHist(briskdisplacements, 10)

        siftpeak = getHistPeak(sifthists)
        surfpeak = getHistPeak(surfhists)
        kazepeak = getHistPeak(kazehists)
        akazepeak = getHistPeak(akazehists)
        briskpeak = getHistPeak(briskhists)

        print("sift", siftpeak, "surf", surfpeak, "kaze", kazepeak, "akaze", akazepeak, "brisk", briskpeak)

        settingsString = "_" + str(numberFeatures) + "feats"
        if eraseLower:
            settingsString += "_top_only"

        if firstTime:
            try:
                feats = ["sift", "surf", "kaze", "akaze", "brisk"]
                for i in feats:
                    #os.mkdir(resultsDir + i + settingsString)
                    for j in range(1, 33):
                        #os.mkdir(resultsDir + i + settingsString + "/" + str(j))
                        Path(resultsDir + i + settingsString + "/" + str(j)).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(e)
        firstTime = False

        #write results
        fn = os.path.join(resultsDir, "sift" + settingsString, position, filename + ".best")
        with open(fn, "w") as f:
            f.write(str(siftpeak) + "\n")

        fn = os.path.join(resultsDir, "surf" + settingsString, position, filename + ".best")
        with open(fn, "w") as f:
            f.write(str(surfpeak) + "\n")

        fn = os.path.join(resultsDir, "kaze" + settingsString, position, filename + ".best")
        with open(fn, "w") as f:
            f.write(str(kazepeak) + "\n")

        fn = os.path.join(resultsDir, "akaze" + settingsString, position, filename + ".best")
        with open(fn, "w") as f:
            f.write(str(akazepeak) + "\n")

        fn = os.path.join(resultsDir, "brisk" + settingsString, position, filename + ".best")
        with open(fn, "w") as f:
            f.write(str(briskpeak) + "\n")

