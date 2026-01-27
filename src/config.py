import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "Golub_Leukemia": {
        "train_path": os.path.join(BASE_DIR, 'dataset', 'golub', 'data_set_ALL_AML_train.csv'),
        "test_path": os.path.join(BASE_DIR, 'dataset', 'golub', 'data_set_ALL_AML_independent.csv'), # File test chuẩn
        "label_col": None,
        "is_split": True # Đánh dấu bộ này đã chia sẵn
    },
    "CuMiDa_GSE9476": {
        "path": os.path.join(BASE_DIR, 'dataset', 'cumida', 'Leukemia_GSE9476.csv'),
        "label_col": "type",
        "is_split": False # Đánh dấu bộ này cần split
    }
}

# Cấu hình PA-5
POP_SIZE = 10              
MAX_ITER = 50            
EXCHANGE_INTERVAL = 10     
PRINT_PROGRESS = True    
THRESHOLD = 0.5
K_MAX = 50
R_RUNS = 3

# Trọng số Fitness
ALPHA_ERROR = 0.7
BETA_REDUCTION = 0.2
GAMMA_STABILITY = 0.1

# Cấu hình mô hình - BỎ random_state ở đây để main_console kiểm soát tập trung
RF_PARAMS = {'n_estimators': 50, 'n_jobs': -1}
N_FOLDS = 3
TEST_SIZE = 0.4 