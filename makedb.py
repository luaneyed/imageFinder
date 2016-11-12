import sys	# for commandline arguments
import os
import cv2
import numpy as np
from operator import itemgetter

featureNum = 50

def getDescriptor(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img, None)

	if (des is None):
		return np.array([[0.0] * 128, [0.0] * 128], dtype = np.float32)
	if (len(des) == 1):
		return np.array(des.tolist()[0] * 2, dtype = np.float32)

	return des

def getMatches(descriptor1, descriptor2):
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	return flann.knnMatch(descriptor1, descriptor2, k=2)

def encode(descriptor):
	print('encoding length : %d' % (len(descriptor.tostring())))
	return descriptor.tostring()

def decode(descriptorString):
	d1arr = np.fromstring(descriptorString, dtype = np.float32)
	return d1arr.reshape((int(len(d1arr) / 128)), 128)

def compareKeyDescriptor(kp1, kp2):
	i = 0
	while i < 128:
		if kp1[i] > kp2[i]:
			return True
		if kp1[i] < kp2[i]:
			return False
		i = i + 1
	return True

def hash(descriptor):
	des = descriptor.tolist()
	result = []
	desLen = len(descriptor)
	for i in range(featureNum if (featureNum < desLen) else desLen):
		maxIndex = i
		j = maxIndex + i + 1
		while j < desLen:
			if compareKeyDescriptor(des[j], des[maxIndex]):
				maxIndex = j
			j = j + i + 1
		result.append(des[maxIndex])

	return np.array(result, np.float32)

	return descriptor[0:featureNum]



# def calcSimilarity(matches):
# 	goodCount = 0
# 	badCount = 0

# 	for i,(m,n) in enumerate(matches):
# 	    if m.distance < 0.7 * n.distance:
# 	    	goodCount = goodCount + 1
# 	    else:
# 	    	badCount = badCount + 1

# 	return float(goodCount) / (goodCount + badCount)

def getSimilarity(des1, des2):

	# goodCount = 0
	# badCount = 0

	point = 0

	# print('start ratio')

	for i,(m,n) in enumerate(getMatches(des1, des2)):

		# print('m %f , n %f' % (m.distance, n.distance))
		if m.distance == 0:
				if n.distance == 0:
					point = point + 10000
				else:
					point = point + (m.distance / n.distance)
		elif m.distance < 0.7 * n.distance:
			# print('%f %f %f' % (m.distance / n.distance, m.distance, n.distance))
			point = point + (n.distance / m.distance) ** 3
			# goodCount = goodCount + 1
		# else:
			# badCount = badCount + 1

	# print('end ratio')

	# print('good %d , bad %d' % (goodCount, badCount))
	# print(float(goodCount) / (goodCount + badCount))

	return point

	# return float(goodCount) / (goodCount + badCount)

def makePair(imagePath):
	return (imagePath, encode(hash(getDescriptor(cv2.imread(imagePath, 0)))))

def search(queryDescriptor, db):
	result = []
	# print(queryDescriptor)
	# print('it was query')
	for pair in db:
		similarity = getSimilarity(queryDescriptor, decode(pair[1]))
		if similarity > 0:
			result.append([pair[0], similarity])

	return result

# arg1 = sys.argv[1]
# arg2 = sys.argv[2]
# print(hash(getDescriptor(cv2.imread(arg2,0)))[0])
# print(decode(encode(hash(getDescriptor(cv2.imread(arg2,0)))))[0])
# print(getSimilarity(hash(getDescriptor(cv2.imread(arg1,0))), decode(encode(hash(getDescriptor(cv2.imread(arg2,0)))))))

# print(encode(hash(getDescriptor(cv2.imread(arg2,0)))))

db = []

rootDir = os.getcwd() + '/../images/4094'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
    	path = dirName + '/' + fname
    	extention = os.path.splitext(path)[1]
    	if (os.path.getsize(path) > 0) and (extention == '.jpg' or extention == '.jpeg' or extention == '.png') :
    		# print(path)
    		pair = makePair(path)
    		if pair:
    			db.append(pair)

print('start finding')

candidates = sorted(search(hash(getDescriptor(cv2.imread(sys.argv[1], 0))), db), key = itemgetter(1), reverse = True)
maxScore = candidates[0][1]
for i in range(1, len(candidates)):
	if candidates[i][1] < min(80, maxScore - 10):
		for answer in candidates[:i]:
			print('\tsimilarity point : %f , path : %s' % (answer[1], answer[0]))
		break


# for i in range(1, featureNum):
# 	print(i)

# drawMatchingImage(arg2, arg1, 'output4.jpg')

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
 





# f = open(os.getcwd() + '/image.jpg', 'r')
# data = f.read()
# print(data)
# f.close()



# print sys.argv[1]
