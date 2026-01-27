# import fs_mapping
# import rf_evaluator

# def fitness_feature_selection(z, X, y, config):
#     # 1. Ánh xạ sang nhị phân (có giới hạn K_MAX)
#     mask = fs_mapping.binarize_solution(z, config.THRESHOLD, k_max=config.K_MAX)
    
#     # 2. Tính sai số (Error Rate) và Accuracy trên tập huấn luyện (CV)
#     error_rate, stats, num_sel = rf_evaluator.evaluate_rf(X, y, mask, config, is_cv=True)
    
#     # 3. Tính tỷ lệ giảm gene
#     reduction_rate = num_sel / X.shape[1]
    
#     # 4. Tính Fitness tổng hợp
#     fit_val = (config.ALPHA_ERROR * error_rate) + (config.BETA_REDUCTION * reduction_rate)
    
#     # TRẢ VỀ: Fitness và thông tin chi tiết
#     details = {
#         'error_rate': error_rate,
#         'reduction_rate': reduction_rate,
#         'num_selected': num_sel,
#         'train_acc_cv': stats['acc']
#     }
    
#     return fit_val, details




import fs_mapping
import rf_evaluator
import numpy as np

def jaccard_similarity(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union != 0 else 0

def fitness_feature_selection(z, X, y, config, reference_masks=None):
    # 1. Ánh xạ sang nhị phân
    mask = fs_mapping.binarize_solution(z, config.THRESHOLD, k_max=config.K_MAX)
    
    # 2. Sai số phân loại E(x)
    error_rate, stats, num_sel = rf_evaluator.evaluate_rf(X, y, mask, config, is_cv=True)
    
    # 3. Tỷ lệ số gene được chọn R(x)
    reduction_rate = num_sel / X.shape[1]
    
    # 4. Độ mất ổn định U(x) = 1 - Stab
    unstability = 0
    if reference_masks and len(reference_masks) > 0:
        # Tính Jaccard trung bình với các tập gene từ các lần chạy trước
        stabilities = [jaccard_similarity(mask, m) for m in reference_masks]
        unstability = 1 - np.mean(stabilities)
    
    # 5. Hàm mục tiêu tổng hợp F(x) = w1*E + w2*R + w3*U
    # Đảm bảo trong config có ALPHA_ERROR, BETA_REDUCTION, GAMMA_STABILITY
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