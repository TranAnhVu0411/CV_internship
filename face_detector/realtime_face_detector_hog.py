import cv2
import dlib
import time
import imutils
from imutils.video import VideoStream
vs = VideoStream(src=0).start()
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
    # grab the frame from the video stream, resize it, and convert it
	# to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# perform face detection
    rects = detector(frame, 0)
    rects = [convert_and_trim_bb(image, r) for r in rects]

    # loop over the bounding boxes
    for (x, y, w, h) in rects:
		# draw the face bounding box on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the output frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
    	break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()