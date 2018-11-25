from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))

display_window = cv2.namedWindow("Dataset Generating...")

# Detect object in video stream using Haarcascade Frontal Face
face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
time.sleep(1)

# For each person, one face id
ID=input('Enter your ID: ')

# Initialize sample face image
count = 0

# Start looping
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array
	
	# Convert frame to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
	
	# Detect frames of different sizes, list of faces rectangles
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)     
    
	# Loops for each faces
    for (x,y,w,h) in faces:
	
		# Crop the image frame into rectangle
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		
		# Increment sample face image
        count = count+1
		
		# Save the captured image into the datasets folder
        cv2.imwrite("dataset/user."+str(ID)+'.'+str(count)+".jpg",gray[y:y+h,x:x+w])


        
    # Display the video frame, with bounded rectangle on the person's face
    cv2.imshow("Dataset Generating...", image)
    
	# empty the output with truncate(0) between captures and then re-use it in order to produce multiple arrays 
    rawCapture.truncate(0)
	
	# To stop taking video, press 'q' key for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count>50:
        break

camera.close()
cv2.destroyAllWindows()
