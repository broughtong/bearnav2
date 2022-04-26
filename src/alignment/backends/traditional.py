import cv2

bf = cv2.BFMatcher()
featureTypes = {}

#SIFT
try:
    featureTypes["SIFT"] = cv2.SIFT_create()
except:
    pass
try:
    featureTypes["SIFT"] = cv2.xfeatures2d.SIFT_create()
except:
    pass

#SURF
try:
    featureTypes["SURF"] = cv2.SURF_create()
except:
    pass
try:
    featureTypes["SURF"] = cv2.xfeatures2d.SURF_create()
except:
    pass

#KAZE
try:
    featureTypes["KAZE"] = cv2.KAZE_create()
except:
    pass

#AKAZE
try:
    featureTypes["AKAZE"] = cv2.AKAZE_create()
except:
    pass

#BRISK
try:
    featureTypes["BRISK"] = cv2.BRISK_create()
except:
    pass

#ORB
try:
    featureTypes["ORB"] = cv2.ORB_create()
except:
    pass

def detect(img, featureType):

    if featureType not in featureTypes:
        rospy.logwarn("Feature type unknown or unavailable on this machine/installation. Not correcting heading!")
        return None, None

    kps, des = featureTypes[featureType].detectAndCompute(img, None)
    return kps, des

def match(kpsA, desA, kpsB, desB):

    matches = bf.match(desA, desB)
    
    displacements = []
    for match in matches:
        xA = kpsA[match.queryIdx].pt[0]
        xB = kpsB[match.trainIdx].pt[0]
        dist = xB - xA
        displacements.append(dist)

    return displacements

