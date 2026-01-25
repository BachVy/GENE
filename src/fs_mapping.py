import numpy as np

def binarize_solution(z, threshold=0.5, k_max=None):
    """
    Ánh xạ nghiệm liên tục sang nhị phân với cơ chế giới hạn K_MAX.
    """
    # 1. Ép về khoảng [0, 1] bằng Sigmoid
    mask_prob = 1 / (1 + np.exp(-z))
    
    # 2. Tạo mask dựa trên ngưỡng threshold
    mask = (mask_prob >= threshold).astype(int)
    
    # 3. Cơ chế giới hạn K_MAX (Repair Mechanism)
    # Nếu k_max được thiết lập và số lượng gene chọn > k_max
    if k_max is not None and np.sum(mask) > k_max:
        # Tìm chỉ số của k_max gene có xác suất cao nhất
        top_indices = np.argsort(mask_prob)[-k_max:] 
        
        # Reset mask về 0 và chỉ bật 1 cho top k_max gene
        new_mask = np.zeros_like(mask)
        new_mask[top_indices] = 1
        return new_mask
        
    return mask

