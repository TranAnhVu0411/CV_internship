import cv2
import dlib
import time
import imutils
from imutils.video import VideoStream
import time
import numpy as np

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

# list of fps in one frame
fps_list = []

vs = VideoStream(src=0).start()

# change from (left, top, right, bottom) to (x, y, w, h)
def convert_and_trim_bb(image, rect):
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()

    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    
    w = endX - startX
    h = endY - startY

    return (startX, startY, w, h)
detector = dlib.get_frontal_face_detector()
time.sleep(2.0)

while True:
    # grab the frame from the video stream, resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

	# perform face detection
    rects = detector(frame, 0)
    rects = [convert_and_trim_bb(frame, r) for r in rects]

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