from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
from time import sleep

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))

display_window = cv2.namedWindow("Detecting Faces...")

faceRecognizer = cv2.face.createLBPHFaceRecognizer()
faceRecognizer.load('trainer/trainer.yml')

# Detect object in video stream using Haarcascade Frontal Face
faceCascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
sleep(1)

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin =30

nametagColor = 255,0,0
nametagHeight = 50

faceRectangleBorderColor = nametagColor
faceRectangleBorderSize = 2

sleep(1)

# Start looping
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array
	
	# Convert frame to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	# Detect frames of different sizes, list of faces rectangles
    faces = faceCascade.detectMultiScale(gray, 1.1, 5)
	
	
	
    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), faceRectangleBorderColor, faceRectangleBorderSize)

        # Recognize the face belongs to which ID
        ID, Confidence = faceRecognizer.predict(gray[y:y+h,x:x+w])

        if((round(Confidence-100, 0)) < 20):
            if(ID==1):
                ID="Mya"
            
                
            elif(ID==2):
                ID="Moe"
                
            elif(ID==3):
                ID="Ines"


                
        else:
            ID="Unknown"

        # Put text describe who is in the picture
        cv2.rectangle(image, (x-22,y-nametagHeight), (x+w+22, y-22), nametagColor, -1)
        cv2.putText(image, str(ID)+": "+str(Confidence)+"%", (x,y-fontBottomMargin),fontFace, fontScale, fontColor, fontWeight)
        
        
        
	 
    #display on windows
    cv2.imshow("Detecting Faces...", image)
    
    
    rawCapture.truncate(0)

		
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
 
camera.close()
cv2.destroyAllWindows()


