from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
import cv2
import dlib
import openface
import json
import os
import timeit
import pandas as pd
import create_directory

preprocess_model_path = create_directory.detection_preprocess_model_dir

class Face_Recognition:
    def __init__(self, type, threshold=0.6):
        self.threshold = threshold
        self.type = type
        # Bộ phát hiện và xử lý khuôn mặt sử dụng HOG và aligned face của Openfacqe
        self.hog_detector = dlib.get_frontal_face_detector()
        self.face_aligner = openface.AlignDlib(os.path.join(preprocess_model_path, "shape_predictor_68_face_landmarks.dat"))
        # Mô hình lấy đặc trưng khuôn mặt của openface
        self.openface = cv2.dnn.readNetFromTorch(os.path.join(preprocess_model_path, "nn4.small2.v1.t7"))
        with open(os.path.join(create_directory.marvel_data_dir, 'distance_face_recognition/feature.json')) as json_file:
            self.feature = json.load(json_file)
     
    def face_detection(self, frame):
        self.preprocess_face = []
        self.bounding_box = []
        # start = timeit.default_timer()
        rects = self.hog_detector(frame, 0)
        # end = timeit.default_timer()
        # print("FACE BB DETECT RUNTIME: "+str(end-start))
        for idx, i in enumerate(rects):
            # start = timeit.default_timer()
            preprocess = self.face_aligner.align(imgDim = 96, rgbImg = frame, bb = i, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            # end = timeit.default_timer()
            # print("FACE "+str(idx)+" ALIGNMENT RUNTIME: "+str(end-start))
            self.bounding_box.append((i.left(), i.top(), i.right(), i.bottom()))
            self.preprocess_face.append(preprocess)
    
    def extract_feature(self):
        self.feature_list = []
        for image in self.preprocess_face:
            blob = cv2.dnn.blobFromImage(image, 1./255, (96, 96), (0,0,0))
            self.openface.setInput(blob)
            feature = self.openface.forward()
            self.feature_list.append(feature.reshape(128).tolist())
    
    def face_recognition(self):
        self.predict = []
        if self.type == 'nearest':
            for feat in self.feature_list:
                candidate = {}
                for label, vectors in self.feature.items():
                    if label not in candidate:
                        candidate[label]=[]
                    for vector in vectors:
                        candidate[label].append(euclidean_distances(np.array(feat).reshape(1, -1), np.array(vector).reshape(1, -1))[0][0])
                    candidate[label] = min(candidate[label])
                best_can = min(candidate, key=candidate.get)
                if candidate[best_can]>=self.threshold:
                    self.predict.append((best_can, candidate[best_can]))
                else:
                    self.predict.append(("Unknown", candidate[best_can]))
        elif self.type == 'neighbour':
            for feat in self.feature_list:
                candidate_label = []
                candidate_distance = []
                for label, vectors in self.feature.items():
                    for vector in vectors:
                        distance = euclidean_distances(np.array(feat).reshape(1, -1), np.array(vector).reshape(1, -1))[0][0]
                        if distance<self.threshold:
                            candidate_label.append(label)
                            candidate_distance.append(distance)
                if len(candidate_label)==0:
                    self.predict.append(("Unknown", 0))
                else:
                    df_candidate=pd.DataFrame({'Label': candidate_label})
                    df_candidate_count = pd.DataFrame(df_candidate["Label"].value_counts().to_dict().items(), columns=['Label','Count'])
                    list_candidate = list(df_candidate_count[df_candidate_count['Count']==max(df_candidate_count['Count'])]['Label'])
                    if len(list_candidate) == 1:
                        self.predict.append((max(candidate_label, key = candidate_label.count), 0))
                    else:
                        df_candidate_distance=pd.DataFrame({'Label': candidate_label, 'Distance': candidate_distance})
                        best_candidate = df_candidate_distance[df_candidate_distance['Label'].isin(list_candidate)].groupby('Label').mean().to_dict()['Distance']
                        self.predict.append((min(best_candidate, key = best_candidate.get),0))
    
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

            predict_name, predict_prob = self.predict[idx]
            # print(predict_name)
            cv2.rectangle(frame, (startX, startY), (startX+w, startY+h), (0, 255, 0), 2)
            cv2.putText(frame, predict_name+": "+"%.2f" % predict_prob, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (36,255,12), 2)

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
        save_file_name = os.path.splitext(os.path.basename(image_path))[0]+"_"+self.type+".jpeg"
        cv2.imwrite(os.path.join(save_path, save_file_name), frame)
        # Display the resulting frame
        cv2.imshow('Frame', frame)

if __name__ == '__main__':
    sample_path = create_directory.sample_dir
    save_path = create_directory.result_dir
    face_recognition = Face_Recognition(type = "nearest")
    face_recognition.face_recognition_image(os.path.join(sample_path, "test3.jpeg"), 
                                             save_path)
    # face_recognition.face_recognition_video(os.path.join(sample_path, "Robert Downey Jr  Scarlett Johansson Mark Ruffalo Chris Hemsworth Interview.mp4"))