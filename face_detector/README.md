# Face detector

Sử dụng 3 bộ phát hiện khuôn mặt chính là: 
- Bộ phát hiện khuôn mặt Viola Jones (Sử dụng đặc trưng Haar-like)
- Bộ phát hiện khuôn mặt sử dụng đặc trưng HOG
- Bộ phát hiện khuôn mặt sử dụng MTCNN

Chạy trong 2 tình huống:
- Ảnh tĩnh: stable_face_detector.ipynb
- Webcam:   
    + realtime_face_detector_haar.py (Viola Jones)
    + realtime_face_detector_hog.py (Đặc trưng HOG)
    + realtime_face_detector_mtcnn.py (MTCNN)

Kết quả chạy:
- Ảnh tĩnh: trong file stable_face_detector.ipynb
- Ảnh động: (Thử nghiệm trên 500 frame đầu)
    + viola jones: fps trung bình: 24
    + hog: fps trung bình: 33
    + mtcnn: fps trung bình: 17 

Báo cáo chi tiết có trong: https://docs.google.com/document/d/1ZJCEW3vqkmhvHuSK5rlx8Lw-Pj2ngZCu/edit?usp=sharing&ouid=113706365275486061312&rtpof=true&sd=true 

 
