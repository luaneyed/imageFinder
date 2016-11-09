import sys, os
# import cv

# img = cv2.imread(sys.argv[1])
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT()
# keypoints, descriptors = sift.detectAndCompute(img_gray, None)
 

rootDir = os.getcwd() + '/../images/'
for dirName, subdirList, fileList in os.walk(rootDir):
    # print('Found directory: %s' % dirName)
    for fname in fileList:
        print('\t%s' % dirName + '/' + fname)


# f = open(os.getcwd() + '/image.jpg', 'r')
# data = f.read()
# print(data)
# f.close()



# print sys.argv[1]