### 第三版初步估计参数的方法，mu,gamma 随时间变化， 引入RMSLE估计误差

from scipy.integrate import solve_ivp   
from scipy.optimize import curve_fit, least_squares
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pathlib

# λ(t) 候选函数
def lambda_logistic(a, t):   # a = [L, k, t0]
    return a[0] / (1.0 + np.exp(-a[1]*(t - a[2])))

def lambda_exp_tail(a, t):   # a = [b0, k, c]
    return a[0] + np.exp(-a[1]*(t + a[2]))

# κ(t) 候选函数
def kappa_sym_logit(a, t):   # a = [K, k, t0], 形状 ~ 1/(exp(k(t-t0))+exp(-k(t-t0)))
    return a[0] / (np.exp(a[1]*(t - a[2])) + np.exp(-a[1]*(t - a[2])))

def kappa_gauss(a, t):       # a = [A, k, t0]
    return a[0] * np.exp(-(a[1]*(t - a[2]))**2)

def kappa_exp_tail(a, t):    # a = [b0, k, c]
    return a[0] + np.exp(-a[1]*(t + a[2]))

def _finite_rate(series, t, denom, clip_upper=None, drop_zeros=True):
    dt = np.median(np.diff(t))
    rate = np.diff(series) / dt
    rate = rate / denom[1:]
    x = t[1:].astype(float)

    rate = rate.astype(float)
    rate[np.isinf(rate) | np.isnan(rate)] = np.nan
    if clip_upper is not None:
        rate[np.abs(rate) > clip_upper] = np.nan
    if drop_zeros:
        if np.sum(rate == 0) / rate.size < 0.5:
            rate[rate == 0] = np.nan
    mask = ~np.isnan(rate)
    return x[mask], rate[mask]

def _increments(x):
    x = np.asarray(x, dtype=float)
    return np.clip(np.diff(np.r_[x[0], x]), 0.0, None)

def prefit_lambda(tTarget, Q, R, guess_3):
    # 若 R 很少，不做预拟合，返回 logistic 作为默认
    if (np.max(R) if len(R) else 0) < 20:
        return np.array(guess_3, float), lambda_logistic
    x, y = _finite_rate(R, tTarget, Q, clip_upper=1.0)
    if len(y) < 5:
        return np.array(guess_3, float), lambda_logistic

    # 试两种形式，择优
    cands = [(lambda_logistic, [0,0,0], [1, 5, 100]),
             (lambda_exp_tail, [0,0,0], [1, 5, 100])]
    best = None
    for fun, lb, ub in cands:
        def res(a): return fun(a, x) - y
        try:
            out = least_squares(res, x0=guess_3, bounds=(lb, ub), xtol=1e-6, ftol=1e-6, verbose=0)
            if best is None or out.cost < best[0]:
                best = (out.cost, out.x, fun)
        except Exception:
            pass
    if best is None:
        return np.array(guess_3, float), lambda_logistic
    return best[1], best[2]

def prefit_kappa(tTarget, Q, D, guess_3):
    if (np.max(D) if len(D) else 0) < 10:
        return np.array(guess_3, float), kappa_sym_logit
    x, y = _finite_rate(D, tTarget, Q, clip_upper=3.0)
    if len(y) < 5:
        return np.array(guess_3, float), kappa_sym_logit

    cands = [(kappa_sym_logit, [0,0,0], [1, 5, 100]),
             (kappa_gauss,     [0,0,0], [1, 5, 100]),
             (kappa_exp_tail,  [0,0,0], [1, 5, 100])]
    best = None
    for fun, lb, ub in cands:
        def res(a): return fun(a, x) - y
        try:
            out = least_squares(res, x0=guess_3, bounds=(lb, ub), xtol=1e-6, ftol=1e-6, verbose=0)
            if best is None or out.cost < best[0]:
                best = (out.cost, out.x, fun)
        except Exception:
            pass
    if best is None:
        return np.array(guess_3, float), kappa_sym_logit
    return best[1], best[2]                    

def seird_ode_timevary(t, y, alpha, beta, delta, lambda_coef, kappa_coef, Npop,
                       lambda_fun, kappa_fun):
    S, E, I, Q, R, D = y
    lam = float(lambda_fun(lambda_coef, t))
    kap = float(kappa_fun(kappa_coef, t))
    dS = -(beta * S * I + beta * 0.5 * S * Q) / Npop
    dE = -dS - alpha * E
    dI = alpha * E - delta * I
    dQ = delta * I - (lam + kap) * Q
    dR = lam * Q
    dD = kap * Q
    return [dS, dE, dI, dQ, dR, dD]


def euler_seird_solver(alpha, beta, delta, lambda_params, kappa_params,
                                   Npop, y0, t_full, lambda_fun = lambda_logistic, kappa_fun = kappa_gauss):
    """
    使用欧拉法(dt=1)求解SEIRD模型
    """
    # 初始化
    n_steps = len(t_full)
    dt = 1  # 固定步长为1
    # 创建结果数组
    sol_y = np.zeros((len(y0), n_steps))
    sol_y[:, 0] = y0  # 初始条件
    # 欧拉法迭代
    for i in range(1, n_steps):
        t_prev = t_full[i-1]
        y_prev = sol_y[:, i-1]
        # 计算当前点的导数
        dydt = seird_ode_timevary(t_prev, y_prev, alpha, beta, delta, lambda_params, kappa_params, Npop, lambda_fun, kappa_fun)
        # 欧拉公式: y_new = y_old + dt * dy/dt
        sol_y[:, i] = y_prev + dt * np.array(dydt)
    return sol_y

def fit_SEIRD(Q_data, R_data, D_data, Npop, E0, I0, time, guess, lambda_fun = lambda_logistic, kappa_fun = kappa_gauss, **kwargs):
    tolX = kwargs.get('tolX', 1e-5)
    tolFun = kwargs.get('tolFun', 1e-5)
    dt = 1

    # Data preprocessing
    Q_data = np.maximum(Q_data, 0)
    R_data = np.maximum(R_data, 0)
    D_data = np.maximum(D_data, 0)

    has_R = not np.all(R_data == 0) and not np.any(np.isnan(R_data))

    t_target = np.array(time)
    t_full = np.arange(t_target[0], t_target[-1] + dt, dt)

    guess = list(guess) 
    initial_params = np.array(guess)

    def model_for_fitting(t_target, *params):
        alpha, beta, delta = params[0:3]
        lambda_params = params[3:6]
        kappa_params = params[6:9]
        #print(f"Testing parameters: alpha={alpha}, beta={beta}, delta={delta}, lambda0={lambda0}, kappa={kappa}")
        # Initial conditions
        S0 = Npop - Q_data[0] - D_data[0] - E0 - I0
        if has_R:
            S0 -= R_data[0]
        y0 = [S0, E0, I0, Q_data[0], R_data[0] if has_R else 0, D_data[0]]
        sol = euler_seird_solver(alpha, beta, delta,
                                                    lambda_params, kappa_params,
                                                    Npop, y0, t_full, lambda_fun = lambda_fun, kappa_fun = kappa_fun)
        Q_model = sol[3, :]
        R_model = sol[4, :]
        D_model = sol[5, :]

        Q_interp = np.interp(t_target, t_full, Q_model)
        R_interp = np.interp(t_target, t_full, R_model)
        D_interp = np.interp(t_target, t_full, D_model)
        
        if has_R:
            return np.concatenate((Q_interp, R_interp, D_interp))
        else:
            return np.concatenate((Q_interp, D_interp))
    
    # Set parameter bounds based on MATLAB logic
    # lb = [1/10, 0.02, 0.1,   # alpha, beta, delta
    #       0.0, 0.0, 0.0,      # λ参数
    #       0.0, 0.0, 0.1]      # κ参数
    # ub = [1/2, 5.0, 2.0,
    #       1.0, 5.0, 200.0,
    #       1.0, 200.0, 500.0]
    
    lb = [1/10, 0.1, 0.1,   # alpha, beta, delta
          0.0, 0.0, 0.0,      # λ参数
          0.0, 0.0, 0.1]      # κ参数
    ub = [1/2, 2.0, 1.5,
          1.0, 5.0, 200.0,
          1.0, 200.0, 500.0]

    if has_R:
        target_data = np.concatenate((Q_data, R_data, D_data))
    else:
        target_data = np.concatenate((Q_data, D_data))


    def residuals(p, t, data):
        wQ = 1.0 / (np.max(Q_data) + 1e-6)
        wR = 1.0 / (np.max(R_data) + 1e-6) if has_R else 1.0
        wD = 1.0 / (np.max(D_data) + 1e-6)
        weights = np.concatenate((
        np.full_like(Q_data, wQ, dtype=float),
        np.full_like(R_data, wR, dtype=float) if has_R else [],
        np.full_like(D_data, wD, dtype=float)))
        pred = model_for_fitting(t, *p)
        return (pred - data) * weights


    def residuals_v2(p, t, data_concat, has_R=True, eps=1e-8):
        """
        data_concat: 按顺序拼接的观测数据 [Q_gt, (R_gt), D_gt]
                    与 model_for_fitting 的输出 pred_concat 一致
        """
        # 预测的拼接向量（需与 data_concat 同长度、同顺序）
        wQ = 2
        wR = 1
        wD = 1
        pred_concat = model_for_fitting(t, *p)
        nQ = len(Q_data)
        nR = len(R_data) if has_R else 0
        nD = len(D_data)
        Q_gt = data_concat[0:nQ]
        idx = nQ
        R_gt = data_concat[idx:idx+nR]; idx += nR
        D_gt = data_concat[idx:idx+nD]
        Q_pred = pred_concat[0:nQ]
        idx = nQ
        R_pred = pred_concat[idx:idx+nR]; idx += nR
        D_pred = pred_concat[idx:idx+nD]
        Q_gt   = np.clip(Q_gt,   0.0, None)
        Q_pred = np.clip(Q_pred, 0.0, None)
        R_gt   = np.clip(R_gt,   0.0, None)
        R_pred = np.clip(R_pred, 0.0, None)
        D_gt   = np.clip(D_gt,   0.0, None)
        D_pred = np.clip(D_pred, 0.0, None)
        res_Q = np.log1p(Q_pred + eps) - np.log1p(Q_gt + eps)
        res_Q *= np.sqrt(wQ)  # 系数以 sqrt 形式进入，以便总损失≈加权均方
        # --- R、D：先转增量，再做 RMSLE 残差 ---
        dR_gt   = _increments(R_gt)
        dR_pred = _increments(R_pred)
        res_R = np.log1p(dR_pred + eps) - np.log1p(dR_gt + eps)
        res_R *= np.sqrt(wR)
        dD_gt   = _increments(D_gt)
        dD_pred = _increments(D_pred)
        res_D = np.log1p(dD_pred + eps) - np.log1p(dD_gt + eps)
        res_D *= np.sqrt(wD)
        return np.concatenate([res_Q, res_R, res_D])
    
    result = least_squares(
        residuals_v2,
        x0=initial_params,
        args=(t_target, target_data),
        bounds=(lb, ub),
        ftol=tolFun,
        xtol=tolX,
        verbose=1
    )

    coeffs = result.x
    final_cost = result.cost
    #print("Final cost:", final_cost)
    
    # Extract fitted parameters
    alpha1 = coeffs[0]
    beta1 = coeffs[1]
    delta1 = coeffs[2]
    lambda_param = coeffs[3:6]
    kappa_param = coeffs[6:9]

    return alpha1, beta1, delta1, lambda_param, kappa_param, final_cost


def fit_function(df, initial_conditions=None):
    if initial_conditions is None:
        E = np.arange(1000,10000,500)
        I = np.arange(1000,10000,500)
        min_cost = float('inf')
        best_E0 = None
        best_I0 = None
        for E0 in E:
            for I0 in I:
                try:
                    alpha1, beta1,  delta1, lambda_hat, kappa_hat, final_cost  = fit_SEIRD(
                        Q_data=df['Active'].values,
                        R_data=df['Recovered'].values,
                        D_data=df['Deaths'].values,
                        Npop=df['Population'].values[0],
                        E0=E0,
                        I0=I0,
                        time=np.arange(len(df)),
                        guess = [0.2, 0.5, 0.2,   0.5, 0.05, 40.0,   0.03, 0.05, 90.0],
                        lambda_fun=lambda_logistic, kappa_fun=kappa_gauss
                    )
                    print("E0:", E0, "I0:", I0, "Final cost:", final_cost)
                    if final_cost < min_cost:
                        min_cost = final_cost
                        best_E0 = E0
                        best_I0 = I0
                except Exception as e:
                    print("E0:", E0, "I0:", I0, "Error:", e)
                    continue
        print("最小final cost:", min_cost, "对应E0:", best_E0, "对应I0:", best_I0)
            # 使用最佳的E0和I0进行最终拟合
        E0 = best_E0
        I0 = best_I0
    else:
        E0, I0 = initial_conditions
    alpha1, beta1,  delta1, lambda_hat, kappa_hat, final_cost = fit_SEIRD(
        Q_data=df['Active'].values,
        R_data=df['Recovered'].values,
        D_data=df['Deaths'].values,
        Npop=df['Population'].values[0],
        E0= E0,
        I0 = I0,
        time = np.arange(len(df)),
        guess = [0.2, 0.5, 0.2,   0.5, 0.05, 40.0,   0.03, 0.05, 90.0],
        lambda_fun=lambda_logistic,
        kappa_fun=kappa_gauss
    )
    print("alpha1:", alpha1, "beta1:", beta1,  "delta1:", delta1, "Lambda1:", lambda_hat, "Kappa1:", kappa_hat)
    #print("R0:", beta1 / delta1 + beta1 * 0.5 / (lambda_hat + kappa_hat))
    print("Final cost:", final_cost)
    return (alpha1, beta1, delta1, lambda_hat, kappa_hat), E0, I0

def plot_comparea(df, parameters, E0, I0,
                  lambda_fun=lambda_logistic, kappa_fun=kappa_gauss):
    """
    对比真实数据与模型预测 (支持时间变 λ(t), κ(t) 的 SEIRD)
    parameters: (alpha, beta, delta, lambda_params, kappa_params)
    """
    Npop = df['Population'].values[0]
    Q0 = df['Active'].values[0]
    R0 = df['Recovered'].values[0]
    D0 = df['Deaths'].values[0]
    S0 = Npop - Q0 - D0 - E0 - I0 - R0
    S0 = max(S0, 0.0)  # 防止负值
    y0 = [S0, E0, I0, Q0, R0, D0]

    t_eval = np.arange(len(df))
    alpha1, beta1, delta1, Lambda1, Kappa1 = parameters
    # 调用欧拉法解
    sol = euler_seird_solver(alpha1, beta1, delta1,
                             Lambda1, Kappa1,
                             Npop, y0, t_eval,
                             lambda_fun=lambda_fun,
                             kappa_fun=kappa_fun)
    # 模型预测值
    D_pred = sol[5]
    Q_pred = sol[3]
    R_pred = sol[4]
    # 真实值
    Q_real = df['Active'].values
    R_real = df['Recovered'].values
    D_real = df['Deaths'].values
    # 绘图
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    # D
    axs[0].plot(t_eval, D_real, 'b:', label='Real Dead')
    axs[0].plot(t_eval, D_pred, 'b-', label='Model Dead')
    axs[0].set_ylabel('Population')
    axs[0].set_title('Dead: Real vs Model')
    axs[0].legend()
    axs[0].grid()
    # Q
    axs[1].plot(t_eval, Q_real, 'orange', linestyle='dotted', label='Real Active')
    axs[1].plot(t_eval, Q_pred, 'orange', label='Model Active')
    axs[1].set_ylabel('Population')
    axs[1].set_title('Active (Quarantined): Real vs Model')
    axs[1].legend()
    axs[1].grid()
    # R
    axs[2].plot(t_eval, R_real, 'g:', label='Real Recovered')
    axs[2].plot(t_eval, R_pred, 'g-', label='Model Recovered')
    axs[2].set_ylabel('Population')
    axs[2].set_title('Recovered: Real vs Model')
    axs[2].legend()
    axs[2].grid()
    plt.xlabel('Days')
    plt.tight_layout()
    plt.show()


def main():
    State_name = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
    path = pathlib.Path().resolve()  #path = "D:\MyDownload\Code\OD-COVID\datasets"
    para_summary = []
    for i in range(len(State_name)):
        name = State_name[i]
        df = pd.read_csv(os.path.join(path, 'datasets', name + '_cases_0412_1231.csv'))
        print(len(df))
        #initial_conditions = [(9500,9500),(1000,9500),(1000,1000),(9500,9500),(1000, 1000)]  # Example initial conditions for E0 and I0
        #parameters, E0, I0 = fit_function(df, initial_conditions[i])
        parameters, E0, I0 = fit_function(df)
        plot_comparea(df, parameters, E0, I0)
        # 记录参数结果
        para_summary.append((name, *parameters, E0, I0))

    # 生成DataFrame
    para_df = pd.DataFrame(para_summary, columns=['State', 'alpha', 'beta', 'delta', 'Lambda', 'Kappa', 'E0', 'I0'])
    print(para_df)
    para_df.rename(columns={'Lambda': 'lambda', 'Kappa': 'kappa'}, inplace=True)
    #para_df.to_csv('D:\MyDownload\Code\OD-COVID\datasets\seird_parameters_v1.csv', index=False)
    para_df.to_csv(os.path.join(path, 'seird_parameters_step1_v4.csv'), index=False)
if __name__ == "__main__":
    main()