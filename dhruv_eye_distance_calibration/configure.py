#THIS IS FOR LIVE TRACKING
#HAVE NOT ADDED THE .TXT CONFIG FILES
#FIRST WORKING WITH ONE FRAME SIMULATION

from __future__ import print_function
import cv2 as cv
import math
import argparse

#constants used in gaze angle calculations
#~3.80 pixels to 1 mm
pixel_conversion = 3.8
ave_eye_size = 30
#ideal distance for computer screen users
z_distance = 400


#Identify/Detect the face
def detectAndDisplay(frame):
	#Simplify the image (color isn't really necessary)
	frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	#graphical rep of the gray picture
	frame_gray = cv.equalizeHist(frame_gray)
	height, width, _ = frame.shape
    
    #-- Detect faces
	faces = face_cascade.detectMultiScale(frame_gray)
	
	left_eye = None
	right_eye = None
	pupilliary_distance = None
	left_pitch = None
	left_yaw = None
	right_pitch = None
	right_yaw = None

	#the pupil distance that is returned
	ret_val = 0
	
    #Look through each face on the screen
	for (x,y,w,h) in faces:
		center = (x + w//2, y + h//2)
        #bounding the face frame
		frame = cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 4)
        #frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
		faceROI = frame_gray[y:y+h,x:x+w]
		
        #-- In each face, detect eyes
		eyes = eyes_cascade.detectMultiScale(faceROI)
		for (ex,ey,ew,eh) in eyes:
			#filtering potential false eyes (mouth and nose)
			if ey+eh < int(2*h/3):
				#create eye region
				eyeROI = frame[ey+y:ey+eh+y, ex+x:ex+ew+x]
				eyeROI_center = (x + ex + ew//2, y + ey + eh//2)
				radius = int(round((ew + eh)*0.25))
				#eye drawing
				#frame = cv.circle(frame, eyeROI_center, radius, (255, 0, 0 ), 4)
				x_pos, y_pos = get_iris_region(eyeROI, x, y, ex, ey, frame)
				if x_pos < x + (w//2) and x_pos != -1:
					left_eye = (x_pos, y_pos)
				elif x_pos != -1:
					right_eye = (x_pos, y_pos)
				
				if left_eye != None and right_eye != None:
					#temp is a tuple with the "pulipalry distance" <- idk what that is 
					# the second element is the distance between the pupils in pixels
					temp = get_pupil_distance(left_eye, right_eye, ew)
					pupilliary_distance = temp[0]
					cv.line(frame, left_eye, right_eye, (0, 0, 255), 2)
					left_pitch, left_yaw = get_pitch_yaw_angle(left_eye, height, width)
					right_pitch, right_yaw = get_pitch_yaw_angle(right_eye, height, width)
			
					print("left_eye: (", math.degrees(left_pitch), math.degrees(left_yaw), ")")
					print("right_eye: (", math.degrees(right_pitch), math.degrees(right_yaw), ")")
					print(pupilliary_distance)
					ret_val = temp[1]

	#cv.imshow('Capture - Face detection', frame)
	return ret_val


def get_iris_region(eyeROI, face_x, face_y, eye_x, eye_y, frame):
	grayEye = cv.cvtColor(eyeROI, cv.COLOR_BGR2GRAY)
	grayEye = cv.GaussianBlur(grayEye, (7, 7), 0)
	_, threshold = cv.threshold(grayEye, 30, 255, cv.THRESH_BINARY_INV)
	contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x:cv.contourArea(x), reverse = True)
	
	for cnt in contours:
		(x, y, w, h) = cv.boundingRect(cnt)
		x_pos = x+int(w/2)+face_x+eye_x
		y_pos = y+int(h/2)+face_y+eye_y
		cv.circle(frame, (x_pos, y_pos), 5, (255, 255, 255), 3)
		return x_pos, y_pos
	
	return -1, -1
	

def get_pupil_distance(left, right, eye_width):
	pixel_dist = right[0] - left[0]
	screen_dist = pixel_dist / pixel_conversion
	ratio = eye_width / pixel_conversion / ave_eye_size
	return (screen_dist / ratio, ((right[0] - left[0])**2 + (right[1] - left[1])**2)**(0.5))


def get_pitch_yaw_angle(eye, screen_height, screen_width):
	#if
	pitch_angle = math.atanh(abs(eye[1] - int(screen_height/2)) / pixel_conversion / z_distance)
	yaw_angle = math.atanh(abs(eye[0] - int(screen_width/2)) / pixel_conversion / z_distance)
	return pitch_angle, yaw_angle


def detectSquare(img):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	ret, thresh = cv.threshold(gray,50,255,0)
	contours, hierarchy = cv.findContours(thresh, 1, 2)
	#print("Number of contours detected:", len(contours))

	for cnt in contours:
		x1,y1 = cnt[0][0]
		approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
		if len(approx) == 4:
			#print("I see an object with 4 points")
			x, y, w, h = cv.boundingRect(cnt)
			ratio = float(w)/h
			if ratio >= 0.9 and ratio <= 1.1 and w > 50:
				#print("SQUAREEEEEEEEEEE")
				cv.drawContours(img, [cnt], -1, (0,255,255), 3)
				return w
				#cv.putText(img, 'Square', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


#Defining Haar Cascade classifier (eyes and face)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade_eye.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)


args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera

#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
    


while True:
	
	ret, frame = cap.read()
    
	if frame is None:
		print('--(!) No captured frame -- Break!')
		break
        
	currPupilDist = detectAndDisplay(frame)
	squareWidth = detectSquare(frame)
	cv.imshow('Capture - Face detection', frame)

	if cv.waitKey(10) == ord('a'):
		f = open("config1.txt", 'w')
		f.writelines([str(currPupilDist) + "\n", str(squareWidth)])
		f.close()

		cv.imwrite("config_img1.jpg", frame)

