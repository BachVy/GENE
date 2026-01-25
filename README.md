# Hệ Thống Tối Ưu Hóa Chọn Lọc Gene Bằng Thuật Toán LAI

## Giới thiệu
Dự án này triển khai thuật toán tối ưu hóa bầy đàn cải tiến PA-5 cho bài toán chọn lọc đặc trưng (Feature Selection) trên dữ liệu biểu hiện Gene (Microarray).  
Mục tiêu của hệ thống là tìm ra tập Gene tối thiểu nhưng vẫn đạt độ chính xác phân loại cao, phục vụ các bài toán chẩn đoán bệnh như Ung thư máu ALL/AML.

---

## 1. Cấu trúc thư mục kết quả

Sau khi chương trình hoàn tất, toàn bộ kết quả sẽ được lưu tại:
```
results/[Tên_Bộ_Dữ_Liệu]/
```

### 1.1. Nhóm file báo cáo (.csv)

- **comprehensive_comparison.csv**  
  Bảng so sánh đối chứng giữa mô hình sử dụng **toàn bộ Gene (Baseline)** và mô hình sau khi tối ưu hóa bằng **PA-5**.

- **pa5_metrics_report.csv**  
  Báo cáo tổng hợp 7 chỉ số đánh giá quan trọng của PA-5:
  - Accuracy
  - Match Count
  - Fitness
  - Error Rate
  - Reduction Rate
  - Số lượng Gene được chọn
  - Thời gian thực thi

- **selected_genes_list.csv**  
  Danh sách tên các Gene tiêu biểu (“Gene vàng”) được thuật toán PA-5 chọn lọc.

- **detailed_predictions_pa5.csv**  
  Kết quả dự đoán chi tiết của mô hình tối ưu PA-5 trên từng mẫu bệnh nhân.

- **detailed_predictions_original.csv**  
  Kết quả dự đoán chi tiết của mô hình gốc (Baseline) sử dụng toàn bộ Gene.

---

### 1.2. Nhóm biểu đồ trực quan (.png)

- **accuracy_compare.png**  
  Biểu đồ cột so sánh trực quan độ chính xác giữa mô hình gốc và mô hình PA-5.

- **confusion_matrix_pa5.png**  
  Ma trận nhầm lẫn thể hiện tỉ lệ phân loại đúng/sai giữa các nhãn bệnh.

- **convergence.png**  
  Đồ thị hội tụ mô tả quá trình tối ưu hóa giá trị Fitness của thuật toán PA-5 qua các vòng lặp.

---

## 2. Hướng dẫn cài đặt và sử dụng

### 2.1. Yêu cầu hệ thống
- Python **3.8 trở lên**
- Các thư viện:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

---
### 2.2. Cài đặt thư viện
```
pip install pandas numpy scikit-learn matplotlib seaborn
```
### 2.3. Cách chạy chương trình
1. Đảm bảo dữ liệu đã được đặt đúng trong thư mục dataset/.

2. Chạy file chính từ terminal:
```
python src/main_console.py
```
3. Chọn chế độ chạy:
- Phím 1: Chạy bộ dữ liệu Golub
- Phím 2: Chạy bộ dữ liệu CuMiDa
- Phím 3: Chạy tất cả các bộ dữ liệu

## 3. Cấu hình thuật toán

Các tham số của thuật toán có thể được điều chỉnh trong file:
```
src/config.py
```
#### Các tham số chính:

- K_MAX: Số lượng Gene tối đa được giữ lại (ví dụ: 10, 20).
- MAX_ITER: Số vòng lặp của thuật toán tối ưu hóa.
- POP_SIZE: Số lượng cá thể trong bầy đàn.
- ALPHA_ERROR & BETA_REDUCTION: Trọng số cân bằng giữa:
    - Độ chính xác phân loại
    - Mức độ giảm số lượng Gene