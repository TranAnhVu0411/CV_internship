# Distance face recognition
Khắc phục được việc gán nhãn Unknown

## Cơ sở cho phương pháp
Bằng cách xem xét về quá trình giảm chiều (PCA) (face_clustering.py), ta nhận thấy vector đặc trưng có khả năng phân cụm khá tốt (các nhãn giống nhau sẽ tập trung lại về một cụm)

## Tóm tắt quá trình nhận diện khuôn mặt
Về các bước trong quá trình nhận diện khuôn mặt sử dụng phương pháp so sánh khoảng cách, bước 1 và bước 2 khá giống như trong Step by Step face recognition, bước 3 có một chút khác biệt: 
- Bước 3: Từ các vector đặc trưng cơ sơ, ta tiến hành so sánh khoảng cách của vector đang xét với các vector đặc trưng cơ sở

## Các phương pháp so sánh khoảng cách
- Phương pháp 1: Nearest candidate
    + Ta sẽ xét vector cơ sở gần nhất với vector đang xét, nếu khoảng cách giữa 2 vector nhỏ hơn threshold thì ta sẽ gán nhãn của vector đang xét là nhãn của vector cơ sở, ngược lại thì ta sẽ gán nhãn unknown
- Phương pháp 2: Naive Neighbour candidates
    + Ta xét khu vực xung quanh vector đang xét có bán kính là threshold, nếu khu vực đó không có vector cơ sở nào thì ta sẽ gán nhãn Unknown, nếu có tồn tại vector cơ sở thì ta sẽ gán nhãn là nhãn có số lượng vector cơ sở tối đa. Nếu có tồn tại 2 hoặc nhiều nhãn có số lượng vector cớ sở tối đa, ta sẽ xét khoảng cách trung bình của các vector với vector đang xét giữa các nhãn, nếu khoảng cách nào nhỏ hơn thì ta sẽ gán nhãn đó
- Phương pháp 3: Naive Neighbour candidates
    + Khá tương tự như phương pháp 2, nhưng ta sẽ không xét nhãn theo số lượng vector cơ sở nữa mà ta xét dựa theo trọng số của các nhãn

## Các metric so sánh
Trong bài này ta sử dụng 2 metric:
- Euclidian Distance:

$$ D_{eulclidean}(X,Y) = \sqrt{\sum_{i=1}^{n}(X_i-Y_i)^{2}} $$
- Cosine Similarity:

$$ similarity(\cos\theta) = \frac {X \cdotp Y}{\Vert X \Vert \Vert Y \Vert} = \frac {\sqrt{\sum_{i=1}^{n}X_i Y_i}}{\sqrt{\sum_{i=1}^{n}X_i^2}\sqrt{\sum_{i=1}^{n}Y_i^2}}$$

## Kết quả thử nghiệm
- Đối với ảnh: thử nghiệm trên 3 ảnh test, test1 và test2 trong phần sample (CV_internship/face_recognition/marvel_face_recognition/sample), kết quả ở trong phần result (CV_internship/face_recognition/marvel_face_recognition/step_by_step_face_recognition/result)
- Đối với video: thử nghiệm đối với một video trong phần sample

## Hướng dẫn chạy code
- Điều kiện: Phải có sẵn thư mục sample_feature từ drive (Xem kĩ hơn trong phần hướng dẫn chạy code của step_by_step_face_recognition)
- Thay đổi đường dẫn thư mục dataset và thư mục code git trong file create_directory.py
- Tạo sample data phù hợp với phần này: chạy code create_data.ipynb
- Để tìm threshold: chạy file face_recognition_stable_fixed_threshold.ipynb
- Để sử dụng bộ nhận diện khuôn mặt: chạy file face_recognition.py
- Để xem khả năng phân cụm: chạy file face_clustering.py