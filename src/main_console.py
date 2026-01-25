import matplotlib
matplotlib.use('Agg') # Sửa lỗi "main thread is not in main loop"
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os, time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import config, data_loader, fitness_wrapper, pa5_runner, fs_mapping

def run_experiment(data_name, task_cfg):
    print(f"\n{'='*45}\n TIẾN TRÌNH: {data_name}\n{'='*45}")
    
    # 1. LOAD DATA (Cơ chế tự động nhận diện bộ Golub đã chia hoặc CuMiDa chưa chia)
    if task_cfg.get('is_split'):
        X_train, y_train, gene_names = data_loader.load_dataset(task_cfg['train_path'])
        X_test, y_test, _ = data_loader.load_dataset(task_cfg['test_path'])
    else:
        X_all, y_all, gene_names = data_loader.load_dataset(task_cfg['path'], task_cfg['label_col'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=config.TEST_SIZE, stratify=y_all, random_state=42
        )

    res_dir = os.path.join(config.BASE_DIR, "results", data_name)
    os.makedirs(res_dir, exist_ok=True)
    rf_params = config.RF_PARAMS.copy()
    rf_params['random_state'] = 42

    # 2. ĐÁNH GIÁ BASELINE (GỐC)
    rf_full = RandomForestClassifier(**rf_params)
    rf_full.fit(X_train, y_train)
    y_pred_full = rf_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)

    # 3. TỐI ƯU PA-5
    print(f"[*] Đang thực hiện tối ưu hóa PA-5...")
    start_t = time.time()
    best_z, best_fit, history = pa5_runner.run_pa5(X_train, y_train, fitness_wrapper.fitness_feature_selection)
    duration = time.time() - start_t
    
    # Lấy thông số chi tiết
    _, fit_details = fitness_wrapper.fitness_feature_selection(best_z, X_train, y_train, config)
    mask = fs_mapping.binarize_solution(best_z, config.THRESHOLD, k_max=config.K_MAX)
    sel_idx = np.where(mask == 1)[0]
    selected_genes = gene_names[sel_idx]

    # Đánh giá PA-5 trên Test
    rf_opt = RandomForestClassifier(**rf_params)
    rf_opt.fit(X_train[:, sel_idx], y_train)
    y_pred_opt = rf_opt.predict(X_test[:, sel_idx])
    acc_opt = accuracy_score(y_test, y_pred_opt)

    # 4. XUẤT HỒ SƠ KẾT QUẢ THEO YÊU CẦU
    
    # A. File pa5_metrics_report.csv (Báo cáo các chỉ số PA-5)
    pa5_report = pd.DataFrame({
        'Metric': [
            'Real Test Accuracy', 
            'Match Count (Test)', 
            'Total Fitness (Train)', 
            'Error Rate (Train)', 
            'Reduction Rate', 
            'Selected Genes Count', 
            'Execution Time (s)'
        ],
        'Value': [
            f"{acc_opt*100:.2f}%",
            f"{np.sum(y_pred_opt == y_test)}/{len(y_test)}",
            f"{best_fit:.6f}",
            f"{fit_details['error_rate']:.6f}",
            f"{fit_details['reduction_rate']:.6f}",
            len(sel_idx),
            f"{duration:.2f}"
        ]
    })
    pa5_report.to_csv(os.path.join(res_dir, 'pa5_metrics_report.csv'), index=False, encoding='utf-8-sig')

    # B. File selected_genes_list.csv (Danh sách Gene đã chọn)
    pd.DataFrame({
        'STT': range(1, len(selected_genes) + 1),
        'Gene_Index': sel_idx,
        'Gene_Name': selected_genes
    }).to_csv(os.path.join(res_dir, 'selected_genes_list.csv'), index=False, encoding='utf-8-sig')

    # C. File comprehensive_comparison.csv (Bảng so sánh Gốc vs PA-5)
    pd.DataFrame({
        'Thông số (Metric)': ['Accuracy', 'Số lượng Gene', 'Thời gian'],
        'Dữ liệu Gốc (Full)': [f"{acc_full*100:.2f}%", X_train.shape[1], "-"],
        'PA-5 Tối ưu': [f"{acc_opt*100:.2f}%", len(sel_idx), f"{duration:.2f}s"]
    }).to_csv(os.path.join(res_dir, 'comprehensive_comparison.csv'), index=False, encoding='utf-8-sig')

    # D. Hai file dự đoán chi tiết (Original vs PA-5)
    y_test_named = data_loader.label_encoder.inverse_transform(y_test)
    
    pd.DataFrame({
        'Actual': y_test_named,
        'Predicted_Original': data_loader.label_encoder.inverse_transform(y_pred_full),
        'Verdict': ['Correct' if a==p else 'Wrong' for a,p in zip(y_test, y_pred_full)]
    }).to_csv(os.path.join(res_dir, 'detailed_predictions_original.csv'), index=False, encoding='utf-8-sig')

    pd.DataFrame({
        'Actual': y_test_named,
        'Predicted_PA5': data_loader.label_encoder.inverse_transform(y_pred_opt),
        'Verdict': ['Correct' if a==p else 'Wrong' for a,p in zip(y_test, y_pred_opt)]
    }).to_csv(os.path.join(res_dir, 'detailed_predictions_pa5.csv'), index=False, encoding='utf-8-sig')

    # E. Các biểu đồ (Ma trận nhầm lẫn, Hội tụ, So sánh Accuracy)
    # Vẽ Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data_loader.label_encoder.classes_, 
                yticklabels=data_loader.label_encoder.classes_)
    plt.title(f'Confusion Matrix PA-5 - {data_name}')
    plt.savefig(os.path.join(res_dir, 'confusion_matrix_pa5.png'))
    plt.close()

    # Vẽ Convergence
    plt.figure()
    plt.plot(history, 'b-')
    plt.title(f'Convergence Curve - {data_name}'); plt.xlabel('Iterations'); plt.ylabel('Fitness')
    plt.savefig(os.path.join(res_dir, 'convergence.png')); plt.close()

    # Vẽ Accuracy Compare
    plt.figure()
    sns.barplot(x=['Original', 'PA-5'], y=[acc_full, acc_opt])
    plt.title(f'Accuracy Comparison - {data_name}')
    plt.savefig(os.path.join(res_dir, 'accuracy_compare.png')); plt.close()

    print(f"[+] Hoàn thành {data_name}. Đã lưu đầy đủ 6 file và 3 biểu đồ.")
    
def main():
    print("--- HỆ THỐNG TỐI ƯU HÓA CHỌN LỌC GENE PA-5 ---")
    print("1. Chạy bộ Golub | 2. Chạy bộ CuMiDa | 3. Chạy tất cả")
    c = input("Lựa chọn của bạn: ")
    
    if c == '1':
        run_experiment("Golub", config.DATASETS["Golub_Leukemia"])
    elif c == '2':
        run_experiment("CuMiDa", config.DATASETS["CuMiDa_GSE9476"])
    elif c == '3':
        for name, cfg in config.DATASETS.items():
            run_experiment(name, cfg)
    else:
        print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()