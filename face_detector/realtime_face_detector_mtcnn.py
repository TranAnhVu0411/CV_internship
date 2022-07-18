import cv2
from matplotlib import cm
import numpy as np
from PIL import Image
from torch_mtcnn import detect_faces
import time
import imutils
from imutils.video import VideoStream
vs = VideoStream(src=0).start()

time.sleep(2.0)
while True:
    # grab the frame from the video stream, resize it, and convert it
	# to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    img = Image.fromarray(np.uint8(frame)).convert('RGB')
    # print(frame.shape)
    # img=frame
	# perform face detection
    try:
        bounding_boxes, landmarks = detect_faces(img)

        # loop over the bounding boxes
        for i in bounding_boxes:
            x1=int(i[0])
            y1=int(i[1])
            x2=int(i[2])
            y2=int(i[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i in landmarks:
            for j in range(int(len(i)/2)):
                x=int(i[j])
                y=int(i[j+5])
                cv2.circle(frame, (x,y), radius=3, color=(0, 0, 255), thickness=-1)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    except:
        print("Error")
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()