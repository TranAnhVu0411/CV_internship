# Distance face recognition
Khắc phục được việc gán nhãn Unknown

## Cơ sở cho phương pháp
Bằng cách xem xét về quá trình giảm chiều (PCA) và phân cụm (face_clustering.py), ta nhận thấy vector đặc trưng có khả năng phân cụm khá tốt (các nhãn giống nhau sẽ tập trung lại về một cụm)

## Tóm tắt quá trình nhận diện khuôn mặt
Về các bước trong quá trình nhận diện khuôn mặt sử dụng phương pháp so sánh khoảng cách, bước 1 và bước 2 khá giống như trong Step by Step face recognition, bước 3 có một chút khác biệt: 
- Bước 3: Từ các vector đặc trưng cơ sơ, ta tiến hành so sánh khoảng cách của vector đang xét với các vector đặc trưng cơ sở

## Các phương pháp so sánh khoảng cách
- Phương pháp 1: Nearest candidate
    + Ta sẽ xét vector cơ sở gần nhất với vector đang xét, nếu khoảng cách giữa 2 vector nhỏ hơn threshold thì ta sẽ gán nhãn của vector đang xét là nhãn của vector cơ sở, ngược lại thì ta sẽ gán nhãn unknown
- Phương pháp 2: Neighbour candidates
    + Ta xét khu vực xung quanh vector đang xét có bán kính là threshold, nếu khu vực đó không có vector cơ sở nào thì ta sẽ gán nhãn Unknown, nếu có tồn tại vector cơ sở thì ta sẽ gán nhãn là nhãn có số lượng vector cơ sở tối đa. Nếu có tồn tại 2 hoặc nhiều nhãn có số lượng vector cớ sở tối đa, ta sẽ xét khoảng cách trung bình của các vector với vector đang xét giữa các nhãn, nếu khoảng cách nào nhỏ hơn thì ta sẽ gán nhãn đó

## Kết quả thử nghiệm
- Đối với ảnh: thử nghiệm trên 3 ảnh test, test1 và test2 trong phần sample (CV_internship/face_recognition/marvel_face_recognition/sample), kết quả ở trong phần result (CV_internship/face_recognition/marvel_face_recognition/step_by_step_face_recognition/result)
- Đối với video: thử nghiệm đối với một video trong phần sample

## Hướng dẫn chạy code
- Điều kiện: Phải có sẵn thư mục dataset download từ drive, cũng như là các thư mục về các vector đặc trưng (Xem kĩ hơn trong phần hướng dẫn chạy code của step_by_step_face_recognition)
- Thay đổi đường dẫn thư mục dataset và thư mục code git trong file create_directory.py
- Để tìm threshold: chạy file face_recognition_stable.ipynb
- Để sử dụng bộ nhận diện khuôn mặt: chạy file face_recognition.py
- Để xem khả năng phân cụm: chạy file face_clustering.py