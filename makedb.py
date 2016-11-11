import sys	# for commandline arguments
import os
import cv2
import numpy as np

def getKeypoints(imagePath):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	return sift.detect(gray, None)

def makeOutputImage(input, output):
	img = cv2.imread(input)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray, None)
	cv2.drawKeypoints(gray,kp,img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite(output,img)

def getStringFromKeyPoint(kp):
	return str(kp.octave)

def getDescriptor(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img, None)
	return des

def getMatches(descriptor1, descriptor2):
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	return flann.knnMatch(descriptor1, descriptor2, k=2)

def getMatchingImage(queryImagePath, trainImagePath):
	# REFERENCE : http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(cv2.imread(queryImagePath,0), None)
	kp2, des2 = sift.detectAndCompute(cv2.imread(trainImagePath,0), None)

	matches = getMatches(des1, des2)

	# Need to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in range(len(matches))]

	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.7*n.distance:
	        matchesMask[i]=[1,0]

	draw_params = dict(matchColor = (0,255,0),
	                   singlePointColor = (255,0,0),
	                   matchesMask = matchesMask,
	                   flags = 0)

	return cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

def drawMatchingImage(queryImagePath, trainImagePath, outputPath):
	cv2.imwrite(outputPath, getMatchingImage(queryImagePath, trainImagePath))

def encode(descriptor):
	return descriptor.tostring()

def decode(descriptorString):
	return np.fromstring(descriptorString, dtype = np.float32)

def lsh(descriptor):
	# print(type(descriptor[0][0]))
	print(descriptor[0:2])
	print(encode(descriptor[0:2]))
	print(decode(encode(descriptor[0:2])))
	return descriptor[0:100]


	print(descriptor[::2][0:3])
	return descriptor[::2]
	output = []
	# print([descriptor[0].tolist()])
	# return descriptor
	for row in descriptor:
		output.append(row[0:4].toList())
	print(output)
	return output


def getSimilarity(queryImagePath, trainImagePath):
	des1 = getDescriptor(cv2.imread(queryImagePath,0))
	des2 = getDescriptor(cv2.imread(trainImagePath,0))

	# print(len(des1))
	# print(len(des1[0]))
	# print(len(des2))
	# print(len(des2[0]))

	matches = getMatches(lsh(des1), lsh(des2))

	goodCount = 0
	badCount = 0

	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.7 * n.distance:
	    	goodCount = goodCount + 1
	    else:
	    	badCount = badCount + 1

	return float(goodCount) / (goodCount + badCount)

arg1 = sys.argv[1]
arg2 = sys.argv[2]

# drawMatchingImage(arg2, arg1, 'output4.jpg')
print(getSimilarity(arg2, arg1))



# kp1 = getKeypoints(img1)
# print(len(kp1))
# kp2 = getKeypoints(img2)
# print(len(kp2))

# makeOutputImage(img1, 'output1.jpg')
# makeOutputImage(img2, 'output2.jpg')

# print(getStringFromKeyPoint(kp1[0]))
# print(getStringFromKeyPoint(kp2[0]))
# print(getStringFromKeyPoint(kp1[1]))
# print(getStringFromKeyPoint(kp2[1]))
# print(getStringFromKeyPoint(kp1[2]))
# print(getStringFromKeyPoint(kp2[2]))

# m1 = MinHash(num_perm = 128)
# m2 = MinHash(num_perm = 128)

# for d in kp1:
#     m1.update(getStringFromKeyPoint(d).encode('utf8'))
# for d in kp2:
#     m2.update(getStringFromKeyPoint(d).encode('utf8'))

# print(m1)
# print(m2)

# lsh = MinHashLSH(threshold = 0.5, num_perm = 128)

# lsh.insert("m1", m1)
# lsh.insert("m2", m2)

# # print("m1" in lsh)
# # print("m2" in lsh)

# result = lsh.query(m1)
# print("Candidates with Jaccard similarity > 0.5", result)






# cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift result',img)

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT()
# keypoints, descriptors = sift.detectAndCompute(img_gray, None)
 







# rootDir = os.getcwd() + '/../images/'
# for dirName, subdirList, fileList in os.walk(rootDir):
#     # print('Found directory: %s' % dirName)
#     for fname in fileList:
#         print('\t%s' % dirName + '/' + fname)


# f = open(os.getcwd() + '/image.jpg', 'r')
# data = f.read()
# print(data)
# f.close()



# print sys.argv[1]