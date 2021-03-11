import cv2

bf = cv2.BFMatcher()
featuresTypes = [
	"SIFT": cv2.SIFT_create(),
	"SURF": cv2.xfeatures2d.SURF_create(),
	"KAZE": cv2.KAZE_create(),
	"AKAZE": cv2.AKAZE_create(),
	"BRISK": cv2.BRISK_create(),
	"ORB": cv2.ORB_create()
	]

def detect(img, featureType):

	if featureType not in featureTypes:
		print("Error: Unknown feature type")
		return

	kps, des = featureTypes[featureType].detectAndCompute(img, None)

def match(kpsA, desA, kpsB, desB, method):

	matches = bf.match(desA, desB)
	
	displacements = []
	for match in matches:
		xA = kpsA[match.queryIdx].pt[0]
		xB = kpsA[match.trainIdx].pt[0]
		dist = xB - xA
		displacements.append(dist)

	return displacements

