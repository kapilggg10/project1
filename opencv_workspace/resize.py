import cv2
import os

list_of_files = open("neg.txt","r")
for img in list_of_files:
	dim = (300,300)
	img = "/home/kapil/Desktop/opencv_workspace"+img[1:len(img)-1]
	image = cv2.imread(img)
	if(not image is None):
		print(img)
		height,width,channels = image.shape
		if((width >=300) & (height >=300)):
			resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
			cv2.imwrite(img,resized)
			#cv2.imshow("resized",resized)
			cv2.waitKey(0)
		else:
			os.remove(img)
	
