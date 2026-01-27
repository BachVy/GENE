import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os, time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import config, data_loader, fitness_wrapper, pa5_runner, fs_mapping

# def run_experiment(data_name, task_cfg):
#     print(f"\n{'='*45}\n TIẾN TRÌNH: {data_name} (R = {config.R_RUNS})\n{'='*45}")
    
#     # 1. LOAD DATA
#     if task_cfg.get('is_split'):
#         X_train, y_train, gene_names = data_loader.load_dataset(task_cfg['train_path'])
#         X_test, y_test, _ = data_loader.load_dataset(task_cfg['test_path'])
#     else:
#         X_all, y_all, gene_names = data_loader.load_dataset(task_cfg['path'], task_cfg['label_col'])
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_all, y_all, test_size=config.TEST_SIZE, stratify=y_all, random_state=42
#         )

#     res_dir = os.path.join(config.BASE_DIR, "results", data_name)
#     os.makedirs(res_dir, exist_ok=True)
    
#     # Biến tích lũy cho R lần chạy
#     all_best_masks = []
#     all_accuracies = []
#     all_durations = []
#     final_best_z = None
#     final_best_fit = np.inf
#     final_history = []

#     # 2. ĐÁNH GIÁ BASELINE
#     rf_full = RandomForestClassifier(**config.RF_PARAMS, random_state=42)
#     rf_full.fit(X_train, y_train)
#     y_pred_full = rf_full.predict(X_test)
#     acc_full = accuracy_score(y_test, y_pred_full)

#     # 3. TỐI ƯU HÓA R LẦN ĐỘC LẬP
#     for r in range(config.R_RUNS):
#         print(f"[*] Đang chạy lần thứ {r+1}/{config.R_RUNS}...")
#         start_t = time.time()
        
#         # Gọi pa5_runner với tham số reference_masks để tính Stability
#         current_z, current_fit, history = pa5_runner.run_pa5(
#             X_train, y_train, 
#             fitness_wrapper.fitness_feature_selection,
#             reference_masks=all_best_masks
#         )
        
#         duration = time.time() - start_t
#         all_durations.append(duration)
        
#         # Lưu mask tốt nhất của lần chạy này
#         m = fs_mapping.binarize_solution(current_z, config.THRESHOLD, k_max=config.K_MAX)
#         all_best_masks.append(m)
        
#         # Kiểm tra nếu đây là lần chạy tốt nhất tổng thể
#         if current_fit < final_best_fit:
#             final_best_fit = current_fit
#             final_best_z = current_z
#             final_history = history

#         # Tính Accuracy nhanh cho lần chạy này để vẽ Boxplot
#         sel_idx_r = np.where(m == 1)[0]
#         rf_r = RandomForestClassifier(**config.RF_PARAMS, random_state=42)
#         rf_r.fit(X_train[:, sel_idx_r], y_train)
#         acc_r = accuracy_score(y_test, rf_r.predict(X_test[:, sel_idx_r]))
#         all_accuracies.append(acc_r)

#     # 4. LẤY KẾT QUẢ TỐI ƯU CUỐI CÙNG
#     _, fit_details = fitness_wrapper.fitness_feature_selection(
#         final_best_z, X_train, y_train, config, reference_masks=all_best_masks[:-1]
#     )
#     mask = all_best_masks[np.argmin(all_accuracies)] # Lấy mask của lần chạy có Acc ổn định
#     sel_idx = np.where(fit_details['mask'] == 1)[0]
#     selected_genes = gene_names[sel_idx]

#     rf_opt = RandomForestClassifier(**config.RF_PARAMS, random_state=42)
#     rf_opt.fit(X_train[:, sel_idx], y_train)
#     y_pred_opt = rf_opt.predict(X_test[:, sel_idx])
#     acc_opt = accuracy_score(y_test, y_pred_opt)

#     # 5. XUẤT HỒ SƠ KẾT QUẢ
    
#     # A. pa5_metrics_report.csv
#     pd.DataFrame({
#         'Metric': ['Real Test Accuracy', 'Match Count (Test)', 'Total Fitness (Train)', 
#                    'Error Rate (Train)', 'Reduction Rate', 'Selected Genes Count', 'Execution Time (s)'],
#         'Value': [f"{acc_opt*100:.2f}%", f"{np.sum(y_pred_opt == y_test)}/{len(y_test)}", 
#                   f"{final_best_fit:.6f}", f"{fit_details['error_rate']:.6f}", 
#                   f"{fit_details['reduction_rate']:.6f}", len(sel_idx), f"{np.sum(all_durations):.2f}"]
#     }).to_csv(os.path.join(res_dir, 'pa5_metrics_report.csv'), index=False, encoding='utf-8-sig')

#     # B. selected_genes_list.csv
#     pd.DataFrame({'STT': range(1, len(selected_genes) + 1), 'Gene_Index': sel_idx, 'Gene_Name': selected_genes}).to_csv(os.path.join(res_dir, 'selected_genes_list.csv'), index=False, encoding='utf-8-sig')

#     # C. comprehensive_comparison.csv
#     pd.DataFrame({
#         'Thông số (Metric)': ['Accuracy', 'Số lượng Gene', 'Thời gian'],
#         'Dữ liệu Gốc (Full)': [f"{acc_full*100:.2f}%", X_train.shape[1], "-"],
#         'PA-5 Tối ưu': [f"{acc_opt*100:.2f}%", len(sel_idx), f"{np.mean(all_durations):.2f}s/run"]
#     }).to_csv(os.path.join(res_dir, 'comprehensive_comparison.csv'), index=False, encoding='utf-8-sig')

#     # D. Detailed Predictions
#     y_test_named = data_loader.label_encoder.inverse_transform(y_test)
#     pd.DataFrame({'Actual': y_test_named, 'Predicted_Original': data_loader.label_encoder.inverse_transform(y_pred_full), 'Verdict': ['Correct' if a==p else 'Wrong' for a,p in zip(y_test, y_pred_full)]}).to_csv(os.path.join(res_dir, 'detailed_predictions_original.csv'), index=False, encoding='utf-8-sig')
#     pd.DataFrame({'Actual': y_test_named, 'Predicted_PA5': data_loader.label_encoder.inverse_transform(y_pred_opt), 'Verdict': ['Correct' if a==p else 'Wrong' for a,p in zip(y_test, y_pred_opt)]}).to_csv(os.path.join(res_dir, 'detailed_predictions_pa5.csv'), index=False, encoding='utf-8-sig')

#     # E. BIỂU ĐỒ
#     # Ma trận nhầm lẫn
#     plt.figure(figsize=(8,6))
#     sns.heatmap(confusion_matrix(y_test, y_pred_opt), annot=True, fmt='d', cmap='Blues', xticklabels=data_loader.label_encoder.classes_, yticklabels=data_loader.label_encoder.classes_)
#     plt.title(f'Confusion Matrix PA-5 - {data_name}'); plt.savefig(os.path.join(res_dir, 'confusion_matrix_pa5.png')); plt.close()

#     # Hội tụ
#     plt.figure(); plt.plot(final_history, 'b-'); plt.title(f'Convergence Curve - {data_name}'); plt.xlabel('Iterations'); plt.ylabel('Fitness')
#     plt.savefig(os.path.join(res_dir, 'convergence.png')); plt.close()

#     # So sánh Accuracy
#     plt.figure(); sns.barplot(x=['Original', 'PA-5'], y=[acc_full, acc_opt]); plt.title(f'Accuracy Comparison - {data_name}')
#     plt.savefig(os.path.join(res_dir, 'accuracy_compare.png')); plt.close()

#     # Boxplot Accuracy (Độ biến động qua R vòng lặp)
#     plt.figure(); plt.boxplot(all_accuracies, labels=['PA-5 Accuracy Distribution'])
#     plt.title(f'Stability Analysis (R={config.R_RUNS}) - {data_name}'); plt.ylabel('Accuracy')
#     plt.savefig(os.path.join(res_dir, 'stability_boxplot.png')); plt.close()

#     print(f"[+] Hoàn thành {data_name}. Đã lưu 6 file CSV, 3 biểu đồ so sánh và 1 biểu đồ Boxplot ổn định.")

# import time
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# import config, data_loader, fitness_wrapper, pa5_runner, fs_mapping

# def run_experiment(data_name, task_cfg):
#     print(f"\n{'='*45}\n TIẾN TRÌNH: {data_name} (R = {config.R_RUNS})\n{'='*45}")
    
#     # 1. LOAD DATA
#     if task_cfg.get('is_split'):
#         X_train, y_train, gene_names = data_loader.load_dataset(task_cfg['train_path'])
#         X_test, y_test, _ = data_loader.load_dataset(task_cfg['test_path'])
#     else:
#         X_all, y_all, gene_names = data_loader.load_dataset(task_cfg['path'], task_cfg['label_col'])
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_all, y_all, test_size=config.TEST_SIZE, stratify=y_all, random_state=42
#         )

#     res_dir = os.path.join(config.BASE_DIR, "results", data_name)
#     os.makedirs(res_dir, exist_ok=True)
#     rf_params = config.RF_PARAMS.copy()
#     rf_params['random_state'] = 42

#     # 2. ĐÁNH GIÁ BASELINE (BỘ GỐC)
#     print(f"[*] Đang thực thi RF trên bộ dữ liệu gốc...")
#     rf_full = RandomForestClassifier(**rf_params)
    
#     # Đo thời gian thực thi (Huấn luyện + Dự đoán) của bộ Gốc
#     start_full = time.time()
#     rf_full.fit(X_train, y_train)
#     y_pred_full = rf_full.predict(X_test)
#     time_exec_full = time.time() - start_full # Thời gian thực thi bộ gốc
    
#     acc_full = accuracy_score(y_test, y_pred_full)

#     # 3. TỐI ƯU HÓA PA-5 (CHẠY R LẦN ĐỂ TÌM GENE)
#     all_best_masks = []
#     all_opt_durations = [] # Thời gian chạy thuật toán tìm kiếm
#     final_best_z = None
#     final_best_fit = np.inf

#     for r in range(config.R_RUNS):
#         start_opt = time.time()
#         current_z, current_fit, history = pa5_runner.run_pa5(
#             X_train, y_train, fitness_wrapper.fitness_feature_selection, reference_masks=all_best_masks
#         )
#         all_opt_durations.append(time.time() - start_opt)
        
#         m = fs_mapping.binarize_solution(current_z, config.THRESHOLD, k_max=config.K_MAX)
#         all_best_masks.append(m)
#         if current_fit < final_best_fit:
#             final_best_fit = current_fit
#             final_best_z = current_z

#     # 4. ĐÁNH GIÁ BỘ DATA ĐÃ CHỌN LỌC (LẦN CUỐI CÙNG)
#     _, fit_details = fitness_wrapper.fitness_feature_selection(final_best_z, X_train, y_train, config)
#     sel_idx = np.where(fit_details['mask'] == 1)[0]
#     selected_genes = gene_names[sel_idx]

#     print(f"[*] Đang thực thi RF trên bộ dữ liệu chọn lọc (PA-5)...")
#     rf_opt = RandomForestClassifier(**rf_params)
    
#     # Đo thời gian thực thi (Huấn luyện + Dự đoán) của bộ đã lọc
#     start_opt_final = time.time()
#     rf_opt.fit(X_train[:, sel_idx], y_train)
#     y_pred_opt = rf_opt.predict(X_test[:, sel_idx])
#     time_exec_opt = time.time() - start_opt_final # Thời gian thực thi bộ rút gọn
    
#     acc_opt = accuracy_score(y_test, y_pred_opt)

#     # 5. XUẤT HỒ SƠ KẾT QUẢ
    
#     # A. pa5_metrics_report.csv (Báo cáo nội bộ PA-5)
#     pd.DataFrame({
#         'Metric': ['Real Test Accuracy', 'Match Count (Test)', 'Total Fitness (Train)', 
#                    'Error Rate (Train)', 'Reduction Rate', 'Selected Genes Count', 'PA-5 Optimization Time (s)'],
#         'Value': [f"{acc_opt*100:.2f}%", f"{np.sum(y_pred_opt == y_test)}/{len(y_test)}", 
#                   f"{final_best_fit:.6f}", f"{fit_details['error_rate']:.6f}", 
#                   f"{fit_details['reduction_rate']:.6f}", len(sel_idx), f"{np.sum(all_opt_durations):.2f}"]
#     }).to_csv(os.path.join(res_dir, 'pa5_metrics_report.csv'), index=False, encoding='utf-8-sig')

#     # B. comprehensive_comparison.csv (So sánh thực thi RF)
#     pd.DataFrame({
#         'Thông số (Metric)': [
#             'Accuracy (%)', 
#             'Số lượng Gene', 
#             'Thời gian thực thi RF (s)' # Huấn luyện + Dự đoán
#         ],
#         'Dữ liệu Gốc (Full)': [
#             f"{acc_full*100:.2f}%", 
#             X_train.shape[1], 
#             f"{time_exec_full:.4f}"
#         ],
#         'PA-5 Tối ưu': [
#             f"{acc_opt*100:.2f}%", 
#             len(sel_idx), 
#             f"{time_exec_opt:.4f}"
#         ]
#     }).to_csv(os.path.join(res_dir, 'comprehensive_comparison.csv'), index=False, encoding='utf-8-sig')

#     # ... [Giữ nguyên các phần xuất Detailed Predictions và Biểu đồ] ...
#     # C. comprehensive_comparison.csv
#     # So sánh tốc độ phản hồi (Inference Speed) của mô hình
#     pd.DataFrame({
#         'Thông số (Metric)': [
#             'Accuracy (%)', 
#             'Số lượng Gene', 
#             'Thời gian đánh giá (s)', # Thời gian dự đoán trên tập Test
#             'Thời gian tối ưu TB (s/run)' # Thời gian thuật toán tìm kiếm gene
#         ],
#         'Dữ liệu Gốc (Full)': [
#             f"{acc_full*100:.2f}%", 
#             X_train.shape[1], 
#             f"{time_eval_full:.6f}", 
#             "-"
#         ],
#         'PA-5 Tối ưu': [
#             f"{acc_opt*100:.2f}%", 
#             len(sel_idx), 
#             f"{time_eval_opt:.6f}", 
#             f"{np.mean(all_opt_durations):.2f}"
#         ]
#     }).to_csv(os.path.join(res_dir, 'comprehensive_comparison.csv'), index=False, encoding='utf-8-sig')

#     # D. Detailed Predictions
#     y_test_named = data_loader.label_encoder.inverse_transform(y_test)
#     pd.DataFrame({'Actual': y_test_named, 'Predicted_Original': data_loader.label_encoder.inverse_transform(y_pred_full), 'Verdict': ['Correct' if a==p else 'Wrong' for a,p in zip(y_test, y_pred_full)]}).to_csv(os.path.join(res_dir, 'detailed_predictions_original.csv'), index=False, encoding='utf-8-sig')
#     pd.DataFrame({'Actual': y_test_named, 'Predicted_PA5': data_loader.label_encoder.inverse_transform(y_pred_opt), 'Verdict': ['Correct' if a==p else 'Wrong' for a,p in zip(y_test, y_pred_opt)]}).to_csv(os.path.join(res_dir, 'detailed_predictions_pa5.csv'), index=False, encoding='utf-8-sig')

#     # E. BIỂU ĐỒ
#     # Ma trận nhầm lẫn
#     plt.figure(figsize=(8,6))
#     sns.heatmap(confusion_matrix(y_test, y_pred_opt), annot=True, fmt='d', cmap='Blues', xticklabels=data_loader.label_encoder.classes_, yticklabels=data_loader.label_encoder.classes_)
#     plt.title(f'Confusion Matrix PA-5 - {data_name}'); plt.savefig(os.path.join(res_dir, 'confusion_matrix_pa5.png')); plt.close()

#     # Hội tụ
#     plt.figure(); plt.plot(final_history, 'b-'); plt.title(f'Convergence Curve - {data_name}'); plt.xlabel('Iterations'); plt.ylabel('Fitness')
#     plt.savefig(os.path.join(res_dir, 'convergence.png')); plt.close()

#     # So sánh Accuracy
#     plt.figure(); sns.barplot(x=['Original', 'PA-5'], y=[acc_full, acc_opt]); plt.title(f'Accuracy Comparison - {data_name}')
#     plt.savefig(os.path.join(res_dir, 'accuracy_compare.png')); plt.close()

#     # Boxplot Accuracy (Độ biến động qua R vòng lặp)
#     plt.figure(); plt.boxplot(all_accuracies, labels=['PA-5 Accuracy Distribution'])
#     plt.title(f'Stability Analysis (R={config.R_RUNS}) - {data_name}'); plt.ylabel('Accuracy')
#     plt.savefig(os.path.join(res_dir, 'stability_boxplot.png')); plt.close()
    
#     # Bổ sung Text hiển thị thời gian trung bình mỗi vòng chạy vào console
#     print(f"[+] Hoàn thành. Thời gian tối ưu trung bình: {np.mean(all_opt_durations):.2f}s/run")
    
#     print(f"[+] Hoàn thành {data_name}. Đã lưu 4 file CSV và 4 biểu đồ.")

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def calculate_jaccard(mask_a, mask_b):
    """Tính chỉ số Jaccard giữa hai tập gene (mask)"""
    set_a = set(np.where(mask_a == 1)[0])
    set_b = set(np.where(mask_b == 1)[0])
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0

def run_experiment(data_name, task_cfg):
    print(f"\n{'='*45}\n TIẾN TRÌNH: {data_name} (R = {config.R_RUNS})\n{'='*45}")
    
    # 1. LOAD DATA
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

    # 2. ĐÁNH GIÁ BASELINE (BỘ DỮ LIỆU GỐC)
    print(f"[*] Đang thực thi RF trên bộ dữ liệu gốc...")
    rf_full = RandomForestClassifier(**rf_params)
    
    start_full = time.time()
    rf_full.fit(X_train, y_train)
    y_pred_full = rf_full.predict(X_test)
    time_exec_full = time.time() - start_full # Thời gian thực thi bộ gốc
    
    acc_full = accuracy_score(y_test, y_pred_full)

    # 3. TỐI ƯU HÓA PA-5 QUA R LẦN CHẠY
    all_best_masks = []
    all_opt_durations = []
    final_best_z = None
    final_best_fit = np.inf
    final_history = []

    for r in range(config.R_RUNS):
        start_opt = time.time()
        # Chạy thuật toán PA-5
        current_z, current_fit, history = pa5_runner.run_pa5(
            X_train, y_train, 
            fitness_wrapper.fitness_feature_selection,
            reference_masks=all_best_masks
        )
        all_opt_durations.append(time.time() - start_opt)
        
        m = fs_mapping.binarize_solution(current_z, config.THRESHOLD, k_max=config.K_MAX)
        all_best_masks.append(m)
        
        if current_fit < final_best_fit:
            final_best_fit = current_fit
            final_best_z = current_z
            final_history = history

    # 4. TÍNH ĐỘ ỔN ĐỊNH (STAB) THEO CÔNG THỨC TRONG ẢNH
    # Stab = [1 / (R-1)] * Tổng J(S_r, S_r-1)
    if len(all_best_masks) > 1:
        jaccard_sums = 0
        for r in range(1, len(all_best_masks)):
            jaccard_sums += calculate_jaccard(all_best_masks[r], all_best_masks[r-1])
        stab_score = jaccard_sums / (len(all_best_masks) - 1)
    else:
        stab_score = 1.0

    # 5. ĐÁNH GIÁ BỘ DATA ĐÃ CHỌN LỌC (LẦN CUỐI CÙNG)
    _, fit_details = fitness_wrapper.fitness_feature_selection(final_best_z, X_train, y_train, config)
    sel_idx = np.where(fit_details['mask'] == 1)[0]
    selected_genes = gene_names[sel_idx]

    print(f"[*] Đang thực thi RF trên bộ dữ liệu chọn lọc (PA-5)...")
    rf_opt = RandomForestClassifier(**rf_params)
    
    start_opt_final = time.time()
    rf_opt.fit(X_train[:, sel_idx], y_train)
    y_pred_opt = rf_opt.predict(X_test[:, sel_idx])
    time_exec_opt = time.time() - start_opt_final # Thời gian thực thi bộ rút gọn
    
    acc_opt = accuracy_score(y_test, y_pred_opt)

    # 6. XUẤT HỒ SƠ KẾT QUẢ (Khớp chính xác image_9b8c55.png)
    
    # A. final_performance_report.csv
    pd.DataFrame({
        'Metric': ['Real Test Accuracy', 'Match Count', 'Total Fitness', 'Error Rate', 
                   'Reduction Rate', 'Stability (Stab)', 'Selected Genes', 'Execution Time (s)'],
        'Value': [f"{acc_opt*100:.2f}%", f"{np.sum(y_pred_opt==y_test)}/{len(y_test)}", 
                  f"{final_best_fit:.6f}", f"{fit_details['error_rate']:.6f}", 
                  f"{fit_details['reduction_rate']:.6f}", f"{stab_score:.6f}", 
                  len(sel_idx), f"{np.sum(all_opt_durations):.2f}"]
    }).to_csv(os.path.join(res_dir, 'final_performance_report.csv'), index=False, encoding='utf-8-sig')

    # B. comparison_report.csv
    pd.DataFrame({
        'Thông số': ['Accuracy', 'Số lượng Gene', 'Thời gian thực thi RF (s)'],
        'Dữ liệu Gốc': [f"{acc_full*100:.2f}%", X_train.shape[1], f"{time_exec_full:.4f}"],
        'PA-5 Tối ưu': [f"{acc_opt*100:.2f}%", len(sel_idx), f"{time_exec_opt:.4f}"]
    }).to_csv(os.path.join(res_dir, 'comparison_report.csv'), index=False, encoding='utf-8-sig')

    # C. selected_genes_list.csv
    pd.DataFrame({'STT': range(1, len(selected_genes) + 1), 'Gene_Name': selected_genes}).to_csv(os.path.join(res_dir, 'selected_genes_list.csv'), index=False, encoding='utf-8-sig')

    # D. detailed_predictions.csv & detailed_predictions_pa5.csv
    y_test_named = data_loader.label_encoder.inverse_transform(y_test)
    pd.DataFrame({'Actual': y_test_named, 'Predicted': data_loader.label_encoder.inverse_transform(y_pred_full), 'Verdict': ['Correct' if a==p else 'Wrong' for a,p in zip(y_test, y_pred_full)]}).to_csv(os.path.join(res_dir, 'detailed_predictions.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame({'Actual': y_test_named, 'Predicted': data_loader.label_encoder.inverse_transform(y_pred_opt), 'Verdict': ['Correct' if a==p else 'Wrong' for a,p in zip(y_test, y_pred_opt)]}).to_csv(os.path.join(res_dir, 'detailed_predictions_pa5.csv'), index=False, encoding='utf-8-sig')

    # 7. VẼ BIỂU ĐỒ
    # # confusion_matrix.png
    # plt.figure(figsize=(8,6))
    # sns.heatmap(confusion_matrix(y_test, y_pred_opt), annot=True, fmt='d', cmap='Blues', xticklabels=data_loader.label_encoder.classes_, yticklabels=data_loader.label_encoder.classes_)
    # plt.title(f'Confusion Matrix - {data_name}'); plt.savefig(os.path.join(res_dir, 'confusion_matrix.png')); plt.close()

    # # convergence_plot.png
    # plt.figure(); plt.plot(final_history); plt.title('Convergence Plot'); plt.xlabel('Iteration'); plt.ylabel('Fitness')
    # plt.savefig(os.path.join(res_dir, 'convergence_plot.png')); plt.close()

    # # accuracy_comparison.png
    # plt.figure(); sns.barplot(x=['Original', 'PA-5'], y=[acc_full, acc_opt]); plt.title('Accuracy Comparison')
    # plt.savefig(os.path.join(res_dir, 'accuracy_comparison.png')); plt.close()
    # --- Confusion Matrix ---
    plt.figure(figsize=(8,6))
    plt.clf() # Xóa trắng figure hiện tại
    sns.heatmap(confusion_matrix(y_test, y_pred_opt), annot=True, fmt='d', cmap='Blues', 
                xticklabels=data_loader.label_encoder.classes_, 
                yticklabels=data_loader.label_encoder.classes_)
    plt.title(f'Confusion Matrix - {data_name}')
    plt.tight_layout() # Đảm bảo không bị mất chữ ở rìa
    plt.savefig(os.path.join(res_dir, 'confusion_matrix.png'))
    plt.close() # Đóng figure để giải phóng RAM

    # --- Convergence Plot ---
    plt.figure(figsize=(8,5))
    plt.clf()
    plt.plot(final_history, color='red', linewidth=1.5, label='Best Fitness')
    plt.title(f'Convergence Plot - {data_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(res_dir, 'convergence_plot.png'))
    plt.close()

    # --- Accuracy Comparison ---
    plt.figure(figsize=(8,5))
    plt.clf()
    # Chuyển sang DataFrame để Seaborn vẽ chuẩn hơn và không bị nhầm lẫn dữ liệu cũ
    df_acc = pd.DataFrame({
        'Dataset': ['Dữ liệu Gốc', 'PA-5 Tối ưu'],
        'Accuracy': [acc_full, acc_opt]
    })
    sns.barplot(data=df_acc, x='Dataset', y='Accuracy', palette='viridis')
    plt.ylim(0, 1.1) # Cố định trục Y từ 0 đến 110% để dễ so sánh trực quan
    plt.title(f'Accuracy Comparison - {data_name}')
    # Ghi chú con số lên đầu cột
    for i, val in enumerate([acc_full, acc_opt]):
        plt.text(i, val + 0.02, f'{val*100:.2f}%', ha='center', fontweight='bold')
    
    plt.savefig(os.path.join(res_dir, 'accuracy_comparison.png'))
    plt.close('all') # Đóng tất cả các figure đang tồn tại

    print(f"[+] Hoàn thành {data_name}. Đã lưu đầy đủ hồ sơ kết quả.")
    
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