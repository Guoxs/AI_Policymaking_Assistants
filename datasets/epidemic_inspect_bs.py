from scipy.integrate import solve_ivp   
from scipy.optimize import curve_fit, least_squares
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
import pandas as pd
import os
import pathlib

def seird_ode(t, y, alpha, beta, delta, lambda0, kappa,  Npop):
    S, E, I, Q, R, D = y

    # Differential equations:
    dS = -(beta * S * I + beta * 0.5 * S * Q)/ Npop
    dE = (beta * S * I + beta * 0.5 * S * Q)/ Npop - alpha * E
    dI = alpha * E - delta * I
    dQ = delta * I - (lambda0 + kappa) * Q
    dR = lambda0 * Q
    dD = kappa * Q

    return [dS, dE, dI, dQ, dR, dD]

def euler_seird_solver(alpha, beta,  delta, lambda0, kappa, Npop, y0, t_full):
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
        dydt = seird_ode(t_prev, y_prev, alpha, beta, delta, lambda0, kappa, Npop)
        # 欧拉公式: y_new = y_old + dt * dy/dt
        sol_y[:, i] = y_prev + dt * np.array(dydt)
    return sol_y

def fit_SEIRD(Q_data, R_data, D_data, Npop, E0, I0, alpha, time, guess, **kwargs):

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

    def model_for_fitting(t_target, *para):
        beta, delta, lambda0, kappa = para[:4]
        #print(f"Testing parameters: alpha={alpha}, beta={beta}, delta={delta}, lambda0={lambda0}, kappa={kappa}")
        # Initial conditions
        S0 = Npop - Q_data[0] - D_data[0] - E0 - I0
        if has_R:
            S0 -= R_data[0]
        y0 = [S0, E0, I0, Q_data[0], R_data[0] if has_R else 0, D_data[0]]
        ### use euler
        sol = euler_seird_solver(alpha, beta,  delta, lambda0, kappa, Npop, y0, t_full)
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
    ub = [5.0, 2.0,  0.2, 0.05]
    lb = [0.02, 0.1, 0.01, 0.0]

    if has_R:
        target_data = np.concatenate((Q_data, R_data, D_data))
    else:
        target_data = np.concatenate((Q_data, D_data))

    result = least_squares(
        lambda p, t, data: model_for_fitting(t, *p) - data,
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
    # alpha1 = coeffs[0]
    beta1 = coeffs[0]
    delta1 = coeffs[1]
    Lambda1 = coeffs[2]
    Kappa1 = coeffs[3]

    return beta1, delta1, Lambda1, Kappa1, final_cost

def fit_function(df, initial_conditions=None, alpha1=None):
    if initial_conditions is None:
        E = np.arange(1000,10000,500)
        I = np.arange(1000,10000,500)
        min_cost = float('inf')
        best_E0 = None
        best_I0 = None
        for E0 in E:
            for I0 in I:
                try:
                    beta1, delta1, Lambda1, Kappa1, final_cost = fit_SEIRD(
                        Q_data=df['Active'].values,
                        R_data=df['Recovered'].values,
                        D_data=df['Deaths'].values,
                        Npop=df['Population'].values[0],
                        E0=E0,
                        I0=I0,
                        alpha=alpha1,
                        time=np.arange(len(df)),
                        guess=[0.5, 0.1, 0.1, 0.001]
                    )
                    print("E0:", E0, "I0:", I0, "Final cost:", final_cost)
                    if final_cost < min_cost:
                        min_cost = final_cost
                        best_E0 = E0
                        best_I0 = I0
                except Exception as e:
                    print("E0:", E0, "I0:", I0, "Error")
                    continue
        print("最小final cost:", min_cost, "对应E0:", best_E0, "对应I0:", best_I0)
            # 使用最佳的E0和I0进行最终拟合
        E0 = best_E0
        I0 = best_I0
    else:
        E0, I0 = initial_conditions
    beta1, delta1, Lambda1, Kappa1, final_cost = fit_SEIRD(
        Q_data=df['Active'].values,
        R_data=df['Recovered'].values,
        D_data=df['Deaths'].values,
        Npop=df['Population'].values[0],
        E0= E0,
        I0 = I0,
        alpha=alpha1,
        time=np.arange(len(df)),
        guess=[0.5,  0.1, 0.1, 0.001]
    )
    print("beta1:", beta1,  "delta1:", delta1, "Lambda1:", Lambda1, "Kappa1:", Kappa1)
    print("R0:", beta1 / delta1 + beta1 * 0.5 / (Lambda1 + Kappa1))
    print("Final cost:", final_cost)
    return (beta1, delta1, Lambda1, Kappa1), E0, I0

def predict_SEIRD(df, parameters, initial_conditions):  
    Npop = df['Population'].values[0]
    Q0 = df['Active'].values[0]
    R0 = df['Recovered'].values[0]
    D0 = df['Deaths'].values[0]
    S0 = Npop - Q0 - D0 - E0 - I0 - R0
    y0 = [S0, E0, I0, Q0, R0, D0]
    t_span = (0, len(df)-1)
    t_eval = np.arange(len(df))
    alpha1, beta1,  delta1, Lambda1, Kappa1 = parameters

    sol = solve_ivp(
        fun=lambda t, y: seird_ode(t, y, alpha1, beta1, delta1, Lambda1, Kappa1, Npop),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45'
    )
    # 模型预测值
    S_pred = sol.y[0]
    E_pred = sol.y[1]
    I_pred = sol.y[2]
    Q_pred = sol.y[3]
    R_pred = sol.y[4]
    D_pred = sol.y[5]
    return S_pred, E_pred, I_pred, Q_pred, R_pred, D_pred

if __name__ == "__main__":
    State_name = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
    for i in range(len(State_name)):
        name = State_name[i]
        path = pathlib.Path().resolve()  #path = "D:\MyDownload\Code\OD-COVID\datasets"
        path = os.path.join(path, "datasets")
        total_para = pd.read_csv(os.path.join(path, "seird_parameters_4197.csv"))
        alpha1 = total_para.loc[total_para['State']==name, 'alpha'].values[0]
        print(f"Using alpha: {alpha1} for {name}")
        para_summary = []
        df = pd.read_csv(os.path.join(path,  name + '_cases_processed.csv'))
        inspect_window = 21
        simulation_steps = 7
        iteation = (len(df) - inspect_window)/simulation_steps + 1
        for it in range(int(iteation)):
            df_fit = df.iloc[it*simulation_steps:it*simulation_steps+inspect_window]
            print(f"Fitting {name}, window {it*simulation_steps} to {it*simulation_steps+inspect_window}")
            if it == 0:
                (beta1, delta1, Lambda1, Kappa1), E0, I0 = fit_function(df_fit, alpha1=alpha1)
            else:
                (beta1, delta1, Lambda1, Kappa1), E0, I0 = fit_function(df_fit, initial_conditions=(E0, I0), alpha1=alpha1)

            para_summary.append({
                'date': df_fit['date'].values[-1],
                'beta': beta1,
                'delta': delta1,
                'gamma': Lambda1,
                'mu': Kappa1
            })
            # 使用拟合参数进行预测
            S_pred, E_pred, I_pred, Q_pred, R_pred, D_pred = predict_SEIRD(df_fit, (alpha1, beta1, delta1, Lambda1, Kappa1), (E0, I0))
            E0 = E_pred[simulation_steps-1]
            I0 = I_pred[simulation_steps-1]
        # 保存拟合参数
        para_df = pd.DataFrame(para_summary)
        path = os.path.join(path, "bs_epidemic")
        para_df.to_csv(os.path.join(path, name + '_bs_epidemic_inspect.csv'), index=False)