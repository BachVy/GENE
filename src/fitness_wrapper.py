import fs_mapping
import rf_evaluator
import numpy as np

def jaccard_similarity(mask_a, mask_b):
    """Tính độ tương đồng Jaccard giữa hai mặt nạ gene"""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union != 0 else 0

def fitness_feature_selection(z, X, y, config, reference_masks=None):
    # 1. Ánh xạ sang nhị phân (chọn tập gene)
    mask = fs_mapping.binarize_solution(z, config.THRESHOLD, k_max=config.K_MAX)
    
    # 2. Sai số phân loại E(x) - Đánh giá qua Cross-Validation trên tập Train
    error_rate, stats, num_sel = rf_evaluator.evaluate_rf(X, y, mask, config, is_cv=True)
    
    # 3. Tỷ lệ số gene được chọn R(x)
    # Lưu ý: Công thức trong ảnh thường dùng (Số gene chọn / Tổng số gene)
    reduction_rate = num_sel / X.shape[1]
    
    # 4. Độ mất ổn định U(x) = 1 - Stab
    # Theo công thức mới: Chỉ so sánh với kết quả lần chạy ngay trước đó (r-1)
    unstability = 0
    if reference_masks and len(reference_masks) > 0:
        # Lấy mask cuối cùng trong danh sách (tương ứng với lần chạy r-1)
        last_mask = reference_masks[-1]
        stab_r = jaccard_similarity(mask, last_mask)
        unstability = 1 - stab_r
    
    # 5. Hàm mục tiêu tổng hợp: F(x) = w1*E(x) + w2*R(x) + w3*U(x)
    # Mục tiêu là tối thiểu hóa Fit_val
    fit_val = (config.ALPHA_ERROR * error_rate) + \
              (config.BETA_REDUCTION * reduction_rate) + \
              (config.GAMMA_STABILITY * unstability)
    
    details = {
        'error_rate': error_rate,
        'reduction_rate': reduction_rate,
        'unstability': unstability,
        'num_selected': num_sel,
        'mask': mask
    }
    return fit_val, details