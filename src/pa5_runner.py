import numpy as np
from scipy.special import gamma as gamma_func
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
import config

# Giữ nguyên hàm Levy từ mã gốc 
def Levy(dim):
    beta = 1.5
    sigma = (gamma_func(1+beta) * np.sin(np.pi*beta/2) / (gamma_func((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u, v = np.random.normal(0, sigma, dim), np.random.normal(0, 1, dim)
    return u / np.abs(v)**(1/beta)

# Thuật toán SBOA gốc - Giữ nguyên các vòng lặp for 
def SBOA(N, T, dim, X, y, fitness_func, PopPos_in=None):
    lb, ub = -10, 10
    PopPos = PopPos_in.copy() if PopPos_in is not None else np.random.rand(N, dim) * (ub - lb) + lb
    pop_fit = np.array([fitness_func(p, X, y, config)[0] for p in PopPos])
    
    best_idx = np.argmin(pop_fit)
    best_f = pop_fit[best_idx]
    best_x = PopPos[best_idx, :].copy()

    for it in range(T):
        # Phase 1: Hunting snakes 
        for i in range(N):
            if (it + 1) <= T / 3:
                r1, r2 = np.random.choice([idx for idx in range(N) if idx != i], 2, replace=False)
                new_pos = PopPos[i, :] + (PopPos[r1, :] - PopPos[r2, :]) * np.random.rand(dim)
            elif (it + 1) <= (2 * T / 3):
                new_pos = best_x + np.exp((it/T)**4) * (np.random.randn(dim) - 0.5) * (best_x - PopPos[i, :])
            else:
                new_pos = best_x + (1 - it/T)**(2*it/T) * PopPos[i, :] * (0.5 * Levy(dim))
            
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config)
            if new_fit < pop_fit[i]: 
                PopPos[i, :], pop_fit[i] = new_pos, new_fit

        # Phase 2: Escaping predators 
        for i in range(N):
            if np.random.rand() < 0.5:
                r2_val = np.random.rand(dim) 
                new_pos = np.random.uniform(0, 2, dim) * best_x + (2 * r2_val - 1) * ((1 - it/T)**2) * PopPos[i, :]
            else:
                rand_idx = np.random.randint(0, N)
                new_pos = np.random.uniform(0, 2, dim) * PopPos[i, :] + np.random.randn(dim) * (PopPos[rand_idx, :] - np.random.choice([1, 2]) * PopPos[i, :])
            
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config)
            if new_fit < pop_fit[i]: 
                PopPos[i, :], pop_fit[i] = new_pos, new_fit

        if np.min(pop_fit) < best_f:
            best_idx = np.argmin(pop_fit)
            best_f, best_x = pop_fit[best_idx], PopPos[best_idx, :].copy()
            
    return best_f, PopPos

# Thuật toán RBMO gốc - Giữ nguyên các vòng lặp for 
def RBMO(N, T, dim, X, y, fitness_func, PopPos_in=None, epsilon=0.5):
    lb, ub = -10, 10
    PopPos = PopPos_in.copy() if PopPos_in is not None else np.random.rand(N, dim) * (ub - lb) + lb
    pop_fit = np.array([fitness_func(p, X, y, config)[0] for p in PopPos])
    
    best_idx = np.argmin(pop_fit)
    best_f = pop_fit[best_idx]
    best_x = PopPos[best_idx, :].copy()

    for it in range(T):
        # 1. Searching Phase 
        for i in range(N):
            # p_val = np.random.randint(2, 6) if np.random.rand() < epsilon else np.random.randint(10, N + 1)
            
            if np.random.rand() < epsilon:
                q_val = np.random.randint(2, 6)
            else:
                # SỬA LỖI: Đảm bảo giới hạn trên luôn lớn hơn giới hạn dưới (10)
                upper_limit = max(11, N + 1)
                q_val = np.random.randint(10, upper_limit)
            
            m_idx = np.random.choice(range(N), min(q_val, N), replace=False)
            sum_Xm = np.mean(PopPos[m_idx], axis=0)
            new_pos = PopPos[i,:] + (sum_Xm - PopPos[np.random.randint(0, N), :]) * np.random.rand()
            
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config)
            if new_fit < pop_fit[i]:
                PopPos[i, :], pop_fit[i] = new_pos, new_fit
                if new_fit < best_f: best_f, best_x = new_fit, new_pos.copy()
        
        # 2. Chasing and Attacking 
        for i in range(N):
            # q_val = np.random.randint(2, 6) if np.random.rand() < epsilon else np.random.randint(10, N + 1)
            
            if np.random.rand() < epsilon:
                q_val = np.random.randint(2, 6)
            else:
                # SỬA LỖI: Đảm bảo giới hạn trên luôn lớn hơn giới hạn dưới (10)
                upper_limit = max(11, N + 1)
                q_val = np.random.randint(10, upper_limit)
            
            m_idx = np.random.choice(range(N), min(q_val, N), replace=False)
            sum_Xm = np.mean(PopPos[m_idx], axis=0)
            CF = (1 - it/T)**(2*it/T)
            new_pos = best_x + CF * (sum_Xm - PopPos[i, :]) * np.random.randn(dim)
            
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config)
            if new_fit < pop_fit[i]:
                PopPos[i, :], pop_fit[i] = new_pos, new_fit
                if new_fit < best_f: best_f, best_x = new_fit, new_pos.copy()

        # 3. Food Storage 
        for i in range(N):
            if np.random.rand() < 0.5:
                new_pos = best_x + np.random.rand() * (best_x - PopPos[i, :]) + \
                          np.random.randn() * (PopPos[np.random.randint(0, N), :] - PopPos[i, :])
            else:
                new_pos = PopPos[i, :] * (1 + np.random.randn(dim) * (1 - it/T))
            
            new_pos = np.clip(new_pos, lb, ub)
            new_fit, _ = fitness_func(new_pos, X, y, config)
            if new_fit < pop_fit[i]:
                PopPos[i, :], pop_fit[i] = new_pos, new_fit
                if new_fit < best_f: best_f, best_x = new_fit, new_pos.copy()
        
    return best_f, PopPos

def run_pa5(X, y, fitness_func):
    dim = X.shape[1]
    N, T, k = config.POP_SIZE // 2, config.MAX_ITER, config.EXCHANGE_INTERVAL
    pop_r, pop_s = None, None
    best_overall_f, best_overall_z = np.inf, None
    fitness_history = []

    for curr_it in range(0, T, k):
        f_r, pop_r = RBMO(N, k, dim, X, y, fitness_func, PopPos_in=pop_r)
        f_s, pop_s = SBOA(N, k, dim, X, y, fitness_func, PopPos_in=pop_s)

        fit_r = np.array([fitness_func(p, X, y, config)[0] for p in pop_r])
        fit_s = np.array([fitness_func(p, X, y, config)[0] for p in pop_s])
        
        # Sharing logic
        pop_r[np.random.randint(0, N)] = pop_s[np.argmin(fit_s)].copy()
        pop_s[np.random.randint(0, N)] = pop_r[np.argmin(fit_r)].copy()

        current_min = min(f_r, f_s)
        if current_min < best_overall_f:
            best_overall_f = current_min
            best_overall_z = pop_r[np.argmin(fit_r)].copy() if f_r < f_s else pop_s[np.argmin(fit_s)].copy()
        
        for _ in range(k): fitness_history.append(best_overall_f)
        if config.PRINT_PROGRESS: print(f"> Iter {curr_it+k}/{T} | Best Fit: {best_overall_f:.5f}")

    return best_overall_z, best_overall_f, fitness_history