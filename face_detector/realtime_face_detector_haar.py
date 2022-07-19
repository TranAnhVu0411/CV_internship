import cv2
import time
import imutils
from imutils.video import VideoStream
import numpy as np

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

# list of fps in one frame
fps_list = []

vs = VideoStream(src=0).start()
# Change path of haarcascade_frontalface_default.xml
detector = cv2.CascadeClassifier("/Users/trananhvu/Documents/CV/venv/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml")
time.sleep(2.0)
while True:
    # grab the frame from the video stream, resize it, and convert it
	# to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# perform face detection
	rects = detector.detectMultiScale(gray, scaleFactor=1.05,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
    	# loop over the bounding boxes
	for (x, y, w, h) in rects:
		# draw the face bounding box on the image
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
	# save fps
	new_frame_time = time.time()
	fps = 1/(new_frame_time-prev_frame_time)
	fps_list.append(fps)
	prev_frame_time = new_frame_time

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed
	# or limit the frame count (only capture 500 frames in this case)
	# break from the loop
	if key == ord("q") or len(fps_list)==500:
	    break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
# show avg fps
print("average fps")
print(np.array(fps_list).mean())