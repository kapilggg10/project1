import cv2
import time
import numpy as np
import os 
from PIL import Image

#import trained classifier for face detection
face_cascade = cv2.CascadeClassifier('./src/cascade/cascade.xml')

#create face recognizer of LBP Histogram
recognizer = cv2.face.LBPHFaceRecognizer_create()

Names = {1: 'yogendra', 2: 'Omprakash_C', 3: 'jarnail', 4: 'Kaushal', 5: 'Kaushal', 6: 'Dikshit', 7: 'Deepak', 8: 'puneet', 9: 'Mahesh', 10: 'Krishna', 11: 'Naresh', 12: 'Niraj', 13: 'Tanmay', 14: 'Kapil', 15: 'Omprakash'}

#names = ['none', 'Kapil', 'Omprakash D','Kaushal','Omprakash C','Deekshith','Krishna','Ankit','Tanmay']
font = cv2.FONT_HERSHEY_SIMPLEX
#function for face detection
def face_detect_recognize(image_path,Names):
	# frame value is stored into img so original input does not get effected
	img = image_path
	# dimensions of frame is resized
	r = 800.0 / img.shape[1]
	dim = (800, int(img.shape[0] * r))
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	#img = cv2.imread(image_path)
	# convert RGB color image to grey
	grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	
	#Detect faces from the input frame using face_cascade
	faces = face_cascade.detectMultiScale(grey, 1.3, 5)
	# draw rectangle across each face and name of the person over that rectangle
	for (x,y,w,h) in faces:
   		cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
		Id, confidence = recognizer.predict(grey[y:y+h,x:x+w])
		if (confidence > 25):
        		Id = Names[Id]
        		confidence = "  {0}%".format(round(100 - confidence))
        	else:
        		Id = "unknown"
        		confidence = "  {0}%".format(round(100 - confidence))
        	cv2.putText(resized, str(Id), (x+5,y-5), font, 1, (255,255,255), 2)
        	cv2.putText(resized, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
		cv2.imshow('img',resized)
	cv2.waitKey(1)

def get_images_and_labels(database_path):

    #get the path of all the files in the folder
    imagePaths=[os.path.join(database_path,f) for f in os.listdir(database_path)] 

    #create empth face list
    sample_faces=[]
    #create empty ID list
    Ids=[]
    Names = {}
    #now looping through all the image paths and loading the Ids and the images
    for image_path in imagePaths:

        # ignore if the file does not have jpg extension :
        if(os.path.split(image_path)[-1].split(".")[-1]!='jpg'):
            continue

        #loading the image and converting it to gray scale
        pilImage=Image.open(image_path).convert('L')

        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(image_path)[-1].split(".")[0])
		name = os.path.split(image_path)[-1].split(".")[1]
        # extract the face from the training image sample
        faces= face_cascade.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            sample_faces.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
	    Names[Id] = name
    return sample_faces,Ids,Names

def train_face_recognizer(database_path):
	faces,Ids,Names = get_images_and_labels(database_path)
	recognizer.train(faces, np.array(Ids))
	recognizer.save('trainer/trainer.yml')
	print "Total number of faces feeded in recognizer :",len(faces)
	return Names

#train_face_recognizer('./src/training_faces')

#import trained faces of database for face recognition
recognizer.read('trainer/trainer.yml')

cap = cv2.VideoCapture("./src/input/video1.mp4")
print(Names)
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detect_recognize(frame,Names)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

