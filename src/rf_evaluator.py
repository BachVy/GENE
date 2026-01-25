from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np

def evaluate_rf(X, y, mask, config, is_cv=True):
    selected_indices = np.where(mask == 1)[0]
    if len(selected_indices) == 0:
        return 1.0, {'acc': 0}, 0
    
    X_selected = X[:, selected_indices]
    rf = RandomForestClassifier(**config.RF_PARAMS)
    
    if is_cv:
        scores = cross_val_score(rf, X_selected, y, cv=config.N_FOLDS)
        acc = np.mean(scores)
        return (1.0 - acc), {'acc': acc}, len(selected_indices)
    else:
        rf.fit(X_selected, y)
        y_pred = rf.predict(X_selected)
        
        acc = accuracy_score(y, y_pred)
        # Macro giúp tính trung bình công bằng cho các bộ dữ liệu nhiều lớp (CuMiDa)
        sens = recall_score(y, y_pred, average='macro', zero_division=0)
        spec = precision_score(y, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y, y_pred)
        
        return (1.0 - acc), {'acc': acc, 'sens': sens, 'spec': spec, 'cm': cm}, len(selected_indices)