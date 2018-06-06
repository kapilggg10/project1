
import cv2
import os

face_cascade = cv2.CascadeClassifier('cascade/cascade.xml')


#Identify the image of interest to import. Ensure that when you import a file path
#that you do not use / in front otherwise it will return empty.
#img = cv2.imread('1.yogendra.1.jpg')


#Identify the face and eye using the haar-based classifiers.
#faces = face_cascade.detectMultiScale(img, 1.3, 5)
#i = 0
#for (x,y,w,h) in faces:
 #   cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
 #   sub_face = img[y:y+h, x:x+w]
  #  FaceFileName = "subject." + str(y)+ "." + "1" + ".jpg"
  #  path = "./unknownfaces"
   # cv2.imwrite(os.path.join(path , FaceFileName),sub_face)
    #cv2.imwrite(FaceFileName, sub_face)

#Display the bounding box for the face
#r = 500.0 / img.shape[1]
#dim = (500, int(img.shape[0] * r))
#resized = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
#cv2.imshow('img',resized)
#cv2.waitKey(5000)



list_of_files = open("images.txt","r")
for img in list_of_files:
	img = "/home/kapil/Desktop/photos"+img[1:len(img)-1]
	image = cv2.imread(img)
	if(not image is None):
		print img
		faces = face_cascade.detectMultiScale(image, 1.3, 5)
		i = 0
		for (x,y,w,h) in faces:
		    #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		    sub_face = image[y-100:y+h+100, x-100:x+w+100]
		    FaceFileName = img.split("/")[-1]
		    path = "./unknownfaces"
		    cv2.imwrite(os.path.join(path , FaceFileName),sub_face)
		    #cv2.imwrite(FaceFileName, sub_face)
	
