import sys	# for commandline arguments
import os
import cv2
import numpy as np
from operator import itemgetter

import myPackage.utils as ut
import elasticsearch
from elasticsearch import helpers

db = []

rootDir = os.getcwd() + '/../images/4094'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
    	path = dirName + '/' + fname
    	extention = os.path.splitext(path)[1]
    	if (os.path.getsize(path) > 0) and (extention == '.jpg' or extention == '.jpeg' or extention == '.png') :
    		# print(path)
    		pair = ut.makePair(path)
    		if pair:
    			db.append(pair)

print('start finding')

candidates = sorted(ut.search(ut.hash(ut.getDescriptor(cv2.imread(sys.argv[1], 0))), db), key = itemgetter(1), reverse = True)
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
