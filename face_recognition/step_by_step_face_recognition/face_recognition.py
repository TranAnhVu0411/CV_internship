import cv2
import dlib
import openface
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
import json
import os
import timeit

path = "/Users/trananhvu/Documents/CV/CV_internship"
preprocess_model_path = os.path.join(path, "model")
recognition_model_path = os.path.join(path, "face_recognition/step_by_step_face_recognition/model")

class Face_Recognition:
    def __init__(self, type, clf_type):
        self.type = type
        self.clf_type = clf_type

        if self.type == "hog_openface":
            self.feature_size = 128
        elif self.type == "mtcnn_facenet":
            self.feature_size = 512

        # Bộ phát hiện và xử lý khuôn mặt sử dụng HOG và aligned face của Openface
        self.hog_detector = dlib.get_frontal_face_detector()
        self.face_aligner = openface.AlignDlib(os.path.join(preprocess_model_path, "shape_predictor_68_face_landmarks.dat"))

        # Bộ phát hiện và xử lý khuôn mặt sử dụng MTCNN và aligned face của facenet_pytorch
        self.mtcnn_detector = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], 
                                   factor=0.709, post_process=True, keep_all=True
                              )
        
        # Mô hình lấy đặc trưng khuôn mặt của openface
        self.openface = cv2.dnn.readNetFromTorch(os.path.join(preprocess_model_path, "nn4.small2.v1.t7"))

        # Mô hình lấy đặc trưng khuôn mặt của facenet
        self.facenet = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.5, num_classes=6, classify=False).eval()

        # Load model
        if self.type == "hog_openface":
            if self.clf_type == "svm":
                self.model = joblib.load(os.path.join(recognition_model_path, "hog_openface_svm_model.sav"))
            elif self.clf_type == "knn":
                self.model = joblib.load(os.path.join(recognition_model_path, "hog_openface_knn_model.sav"))
        elif self.type == "mtcnn_facenet":
            if self.clf_type == "svm":
                self.model = joblib.load(os.path.join(recognition_model_path, "mtcnn_facenet_svm_model.sav"))
            elif self.clf_type == "knn":
                self.model = joblib.load(os.path.join(recognition_model_path, "mtcnn_facenet_knn_model.sav"))

        # idx to face
        with open(os.path.join(recognition_model_path, 'face2idx.json')) as json_file:
            face2idx = json.load(json_file)
        self.idx2face = dict([(value, key) for key, value in face2idx.items()])

    def face_detection(self, frame):
        self.preprocess_face = []
        self.bounding_box = []
        if self.type == "hog_openface":
            rects = self.hog_detector(frame, 3)
            for i in rects:
                preprocess = self.face_aligner.align(imgDim = 96, rgbImg = frame, bb = i, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                self.bounding_box.append((i.left(), i.top(), i.right(), i.bottom()))
                self.preprocess_face.append(preprocess)
        elif self.type == "mtcnn_facenet":
            rects, _ = self.mtcnn_detector.detect(frame)
            if not rects is None:
                for i in rects:
                    self.bounding_box.append(i)
                self.preprocess_face.append(self.mtcnn_detector(frame))
    
    def extract_feature(self):
        self.feature_list = []
        if self.type == "hog_openface":
            for image in self.preprocess_face:
                blob = cv2.dnn.blobFromImage(image, 1./255, (96, 96), (0,0,0))
                self.openface.setInput(blob)
                feature = self.openface.forward()
                self.feature_list.append(feature.reshape(self.feature_size).tolist())
        elif self.type == "mtcnn_facenet":
            features = self.facenet(self.preprocess_face[0])
            for i in features:
                self.feature_list.append(i.tolist())
    
    def face_recognition(self):
        predict_idx = self.model.predict(self.feature_list)
        self.predict = []
        for i in predict_idx:
            self.predict.append(self.idx2face[i])  
    
    def draw_bb_box(self, frame):
        # start_extract_feature = timeit.default_timer()
        self.extract_feature()
        # start_face_recognition = timeit.default_timer()
        # print("EXTRACT FEATURE RUNTIME: "+str(start_face_recognition - start_extract_feature))
        self.face_recognition()
        # end_face_recognition = timeit.default_timer()
        # print("FACE RECOGNITION RUNTIME: "+str(end_face_recognition - start_face_recognition))
        for idx, _ in enumerate(self.bounding_box):
            startX = int(self.bounding_box[idx][0])
            startY = int(self.bounding_box[idx][1])
            endX = int(self.bounding_box[idx][2])
            endY = int(self.bounding_box[idx][3])

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(endX, frame.shape[1])
            endY = min(endY, frame.shape[0])

            w = endX - startX
            h = endY - startY

            predict_name = self.predict[idx]
            # print(predict_name)
            cv2.rectangle(frame, (startX, startY), (startX+w, startY+h), (0, 255, 0), 2)
            cv2.putText(frame, predict_name, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        return frame

    def face_recognition_video(self, video_path):
        cap = cv2.VideoCapture(video_path) 
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video file")

        # Read until video is completed
        while(cap.isOpened()):
            
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                self.face_detection(frame)
                # Nếu phát hiện đc khuôn mặt, ta mới thực hiện việc nhận diện khuôn mặt
                if len(self.bounding_box)!=0:
                    frame=self.draw_bb_box(frame)
                
                # Display the resulting frame
                cv2.imshow('Frame', frame)
            
                # Press Q on keyboard to exit 
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break   
            
            # Break the loop
            else: 
                break
            
        # When everything done, release 
        # the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
    
    def face_recognition_video_save(self, video_path, save_path):
        cap = cv2.VideoCapture(video_path) 
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video file")
        save_file_name = os.path.splitext(os.path.basename(video_path))[0]+"_"+self.type+"_"+self.clf_type+".mp4"
        writer = cv2.VideoWriter(os.path.join(save_path, save_file_name),
                                 int(cap.get(cv2.CAP_PROP_FOURCC)),int(cap.get(cv2.CAP_PROP_FPS)), 
                                 (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

        count=0
        # Read until video is completed
        while(cap.isOpened()):
            
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                self.face_detection(frame)
                # Nếu phát hiện đc khuôn mặt, ta mới thực hiện việc nhận diện khuôn mặt
                if len(self.bounding_box)!=0:
                    frame=self.draw_bb_box(frame)
                
                # Save process frame to list
                writer.write(frame)
                print(count)
                count+=1
            
            # Break the loop
            else: 
                break
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
    
    def face_recognition_image(self, image_path, save_path):
        frame = cv2.imread(image_path)
        print("COMPLETE LOAD IMAGE")
        start_face_detection = timeit.default_timer()
        self.face_detection(frame)
        print("COMPLETE FACE DETECTION")
        end_face_detection = timeit.default_timer()
        print("FACE DETECTION RUNTIME: "+str(end_face_detection-start_face_detection))
        # Nếu phát hiện đc khuôn mặt, ta mới thực hiện việc nhận diện khuôn mặt
        if len(self.bounding_box)!=0:
            frame=self.draw_bb_box(frame)
        print("COMPLETE FACE RECOGNITION")
        save_file_name = os.path.splitext(os.path.basename(image_path))[0]+"_"+self.type+"_"+self.clf_type+".jpeg"
        cv2.imwrite(os.path.join(save_path, save_file_name), frame)
        # Display the resulting frame
        cv2.imshow('Frame', frame)

if __name__ == '__main__':
    sample_path = "/Users/trananhvu/Documents/CV/CV_internship/face_recognition/step_by_step_face_recognition/sample"
    face_recognition = Face_Recognition("mtcnn_facenet", "svm")
    # face_recognition.face_recognition_image(os.path.join(sample_path, "test.webp"), 
    #                                          sample_path)
    face_recognition.face_recognition_video_save(os.path.join(sample_path, "Robert Downey Jr  Scarlett Johansson Mark Ruffalo Chris Hemsworth Interview.mp4"), 
                                                 sample_path)