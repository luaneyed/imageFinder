import cv2
import numpy as np

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

def getSimilarity(des1, des2):
	point = 0

	for i,(m,n) in enumerate(getMatches(des1, des2)):
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