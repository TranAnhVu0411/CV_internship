import os

# Sửa đổi đường dẫn data (data_đir) và đường dẫn thư mục lưu clone code (code_dir)
data_dir = "/Users/trananhvu/Documents/Data-Face-Recognition"
code_dir = "/Users/trananhvu/Documents/CV/CV_internship"

# Không được sửa
marvel_data_dir = os.path.join(data_dir, "Marvel")
detection_preprocess_model_dir = os.path.join(code_dir, "model")
recognition_model_dir = os.path.join(code_dir, "face_recognition/marvel_face_recognition/step_by_step_face_recognition/model")
result_dir = os.path.join(code_dir, "face_recognition/marvel_face_recognition/step_by_step_face_recognition/result")
sample_dir = os.path.join(code_dir, "face_recognition/marvel_face_recognition/sample")