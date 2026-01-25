# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Khởi tạo encoder để dùng chung cho cả load và decode
# label_encoder = LabelEncoder()

# def load_dataset(path, label_col=None):
#     df = pd.read_csv(path)
    
#     # 1. LOẠI BỎ CÁC CỘT KHÔNG PHẢI BIỂU HIỆN GENE
#     # Loại bỏ cột 'samples', các cột chứa 'call', hoặc 'unnamed'
#     trash_cols = [c for c in df.columns if c.lower() in ['samples', 'sample', 'id', 'index'] 
#                   or 'call' in c.lower() or 'unnamed' in c.lower()]
#     df = df.drop(columns=trash_cols)

#     # 2. PHÂN TÁCH X, Y VÀ GENE_NAMES
#     if 'Gene Description' in df.columns:
#         # Dành cho bộ Golub (Gene là hàng)
#         gene_names = df.iloc[:, 1].values
#         X = df.iloc[:, 2:].values.T
#         y_raw = np.array(['ALL']*27 + ['AML']*(X.shape[0]-27)) if "train" in path.lower() else np.array(['ALL']*20 + ['AML']*(X.shape[0]-20))
#     else:
#         # Dành cho bộ CuMiDa (Gene là cột)
#         if label_col and label_col in df.columns:
#             y_raw = df[label_col].values
#             X_df = df.drop(columns=[label_col])
#             X = X_df.values
#             gene_names = X_df.columns.values # Đảm bảo lấy đúng tên Gene
#         else:
#             # Nếu không chỉ định, mặc định cột cuối là nhãn
#             X = df.iloc[:, :-1].values
#             y_raw = df.iloc[:, -1].values
#             gene_names = df.columns[:-1].values

#     # 3. CHUẨN HÓA VÀ MÃ HÓA
#     y = label_encoder.fit_transform(y_raw)
#     X_scaled = StandardScaler().fit_transform(X.astype(float))
    
#     return X_scaled, y, gene_names




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

label_encoder = LabelEncoder()

def load_dataset(path, label_col=None):
    df = pd.read_csv(path)
    
    # 1. Loại bỏ rác
    trash_cols = [c for c in df.columns if c.lower() in ['samples', 'sample', 'id', 'index'] 
                  or 'call' in c.lower() or 'unnamed' in c.lower()]
    df = df.drop(columns=trash_cols)

    # 2. Xử lý theo cấu trúc file
    if 'Gene Description' in df.columns or 'Gene Accession Number' in df.columns:
        # Golub: Gene là hàng, Mẫu là cột
        gene_names = df.iloc[:, 1].values # Lấy Gene Accession
        X = df.iloc[:, 2:].values.T # Chuyển vị để Mẫu thành hàng
        
        # Gán nhãn chuẩn Golub (Train 38 mẫu, Test 34 mẫu)
        if X.shape[0] == 38: # Tập Train
            y_raw = np.array(['ALL']*27 + ['AML']*11)
        elif X.shape[0] == 34: # Tập Test
            y_raw = np.array(['ALL']*20 + ['AML']*14)
        else:
            y_raw = np.array(['Unknown']*X.shape[0])
    else:
        # CuMiDa/General: Gene là cột
        if label_col and label_col in df.columns:
            y_raw = df[label_col].values
            X_df = df.drop(columns=[label_col])
            X = X_df.values
            gene_names = X_df.columns.values
        else:
            X = df.iloc[:, :-1].values
            y_raw = df.iloc[:, -1].values
            gene_names = df.columns[:-1].values

    # 3. Chuẩn hóa và Encode
    y = label_encoder.fit_transform(y_raw)
    X_scaled = StandardScaler().fit_transform(X.astype(float))
    
    return X_scaled, y, gene_names