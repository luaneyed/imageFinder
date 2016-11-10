import sys	# for commandline arguments
import os
import cv2

img = cv2.imread(sys.argv[1])
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
 
cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift result',img)

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