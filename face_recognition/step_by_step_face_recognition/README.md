# Step by step face recognition

Tham khảo bài viết: https://medium0.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

## Tóm tắt quá trình nhận diện khuôn mặt
Về các bước trong quá trình huấn luyện bộ phát hiện khuôn mặt, bao gồm 3 bước chính:
- Bước 1: Tiền xử lý khuôn mặt: bao gồm việc lấy ảnh khuôn mặt, chỉnh khuôn mặt về vị trí thích hợp (Aligned face) và chỉnh sửa kích thước khuôn mặt phù hợp với các mô hình lấy đặc trưng (Thực hiện trong face_preprocess.ipynb)
- Bước 2: Lấy đặc trưng sử dụng các mô hình openface và facenet (Thực hiện trong extract_feature_without_finetune.ipynb)
- Bước 3: Huấn luyện các mô hình SVM và KNN để phân lớp khuôn mặt sử dụng các vector đặc trưng thu được ở bước trên, và lưu lại các mô hình huấn luyện (Thực hiện trong train_face_recognition.ipynb)

Về các bước nhận diện khuôn mặt trong ảnh và video, ta tổng hợp lại bước 1 và bước 2, và sử dụng các mô hình đã huấn luyện ở bước 3 để thực hiện nhận diện khuôn mặt (Thực hiện trong face_recognition.py)

## Công nghệ sử dụng trong quá trình nhận diện khuôn mặt
- Phương pháp 1: 
    + Nhận diện khuôn mặt bằng HOG + chỉnh sửa khuôn mặt bằng Aligned Face của Openface
    + Trích rút đặc trưng bằng Openface
    + Huấn luyện các mô hình SVM và HOG từ các vector đặc trưng thu được ở trên
- Phương pháp 2:
    + Nhận diện khuôn mặt bằng MTCNN + chỉnh sửa khuôn mặt (https://github.com/timesler/facenet-pytorch)
    + Trích rút đặc trưng bằng Facenet
    + Huấn luyện các mô hình SVM và HOG từ các vector đặc trưng thu được ở trên

## Yếu điểm của phần code này
- Phương pháp 2 chạy rất chậm đối với ảnh (mất khoảng hơn 3s cho một ảnh), vì vậy không thể chạy realtime. Phương pháp đầu tiên có phần tốt hơn
- Các mô hình huấn luyện cuối của cả hai phương pháp là các mô hình phân loại, đối với những đối tượng nằm trong các class xác định thì khả năng dự đoán khá tốt, nhưng đối với những đối tượng nằm trong class Unknown thì dự đoán khá là tệ
