import fs_mapping
import rf_evaluator

def fitness_feature_selection(z, X, y, config):
    # 1. Ánh xạ sang nhị phân (có giới hạn K_MAX)
    mask = fs_mapping.binarize_solution(z, config.THRESHOLD, k_max=config.K_MAX)
    
    # 2. Tính sai số (Error Rate) và Accuracy trên tập huấn luyện (CV)
    error_rate, stats, num_sel = rf_evaluator.evaluate_rf(X, y, mask, config, is_cv=True)
    
    # 3. Tính tỷ lệ giảm gene
    reduction_rate = num_sel / X.shape[1]
    
    # 4. Tính Fitness tổng hợp
    fit_val = (config.ALPHA_ERROR * error_rate) + (config.BETA_REDUCTION * reduction_rate)
    
    # TRẢ VỀ: Fitness và thông tin chi tiết
    details = {
        'error_rate': error_rate,
        'reduction_rate': reduction_rate,
        'num_selected': num_sel,
        'train_acc_cv': stats['acc']
    }
    
    return fit_val, details