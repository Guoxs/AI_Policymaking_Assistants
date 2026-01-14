### fit the model with mobility data (the second stage)

import os
import pandas as pd
from datetime import datetime
import time
import numpy as np
from scipy.optimize import least_squares
from agents.workflow import ODWorkflow
from agents.configs import CONFIG
import pathlib
import optuna
from optuna.samplers import TPESampler,CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from agents.state import WorkflowState, RegionState
from agents.state import PopulationState, EpidemicParameters, EpidemicParameters_v2
from agents.utils import setup_logger, load_mobility_data, load_population_data, load_epidemic_params
from fit_param_v import compute_loss, objective, early_stop

def optuna_objective(trial):
    num_regions = len(CONFIG['region_names'])
    root_path = pathlib.Path(__file__).parent.resolve()
    path = os.path.join(root_path, 'datasets', 'seird_parameters_step1_v3.csv')
    obtain_param = pd.read_csv(path)
    # 可调范围建议：围绕 step1 的 0.5–2 或更窄（根据你对 step1 信心）
    # 用 log=True 让搜索更稳（等价于在 log-space 采样）
    # 下面是比较通用的一组范围
    # 可选：对特别敏感的州收紧范围
    # if i in {1,2}: ...
    params = []
    for i in range(num_regions):
        ALPHA_RANGE = (0.5, 3.0)
        BETA_RANGE  = (1, 5.0)
        DELTA_RANGE = (0.2, 1)
        G_RANGE     = (0.3, 3.0)   # gamma_params overall scale
        MU_RANGE    = (0.3, 3.0)   # mu_params overall scale
        E0_RANGE    = (0.2, 5.0)   # 初值通常不确定性更大
        I0_RANGE    = (0.2, 5.0)
        if i == 3:
            ALPHA_RANGE = (0.2, 1.2)
        row = obtain_param[obtain_param['State'] == CONFIG['region_names'][i]].iloc[0]

        alpha0 = float(row['alpha'])
        beta0  = float(row['beta'])
        delta0 = float(row['delta'])
        gamma0 = [float(x) for x in row['lambda'][1:-1].split()]
        mu0    = [float(x) for x in row['kappa'][1:-1].split()]
        E0_0 = int(row['E0'])
        I0_0 = int(row['I0'])

        # ---- scales (all) ----
        alpha_scale = trial.suggest_float(f"alpha_scale_{i}", *ALPHA_RANGE, log=True)
        beta_scale  = trial.suggest_float(f"beta_scale_{i}",  *BETA_RANGE,  log=True)
        delta_scale = trial.suggest_float(f"delta_scale_{i}", *DELTA_RANGE, log=True)

        gamma_scale = trial.suggest_float(f"gamma_scale_{i}", *G_RANGE, log=True)
        mu_scale    = trial.suggest_float(f"mu_scale_{i}",    *MU_RANGE, log=True)

        E0_scale = trial.suggest_float(f"E0_scale_{i}", *E0_RANGE, log=True)
        I0_scale = trial.suggest_float(f"I0_scale_{i}", *I0_RANGE, log=True)

        # ---- apply scales ----
        alpha = alpha_scale * alpha0
        beta  = beta_scale  * beta0
        delta = delta_scale * delta0

        gamma_para = [gamma_scale * v for v in gamma0]
        kappa_para = [mu_scale    * v for v in mu0]

        # 初值取整 + 简单裁剪（避免出现 0 或极端大导致数值不稳）
        E0 = int(round(E0_scale * E0_0))
        I0 = int(round(I0_scale * I0_0))
        E0 = max(1, E0)
        I0 = max(1, I0)

        # ---- special: i == 1 needs beta_2/delta_2 ----
        if i == 1:
            beta_scale_2  = trial.suggest_float(f"beta_scale_2_{i}",  *BETA_RANGE,  log=True)
            delta_scale_2 = trial.suggest_float(f"delta_scale_2_{i}", *DELTA_RANGE, log=True)
            beta_2  = beta_scale_2  * beta0
            delta_2 = delta_scale_2 * delta0

            params.append([alpha, beta, delta, gamma_para, kappa_para, E0, I0, beta_2, delta_2])
        else:
            params.append([alpha, beta, delta, gamma_para, kappa_para, E0, I0])

    list_df = objective(params)
    loss = compute_loss(list_df)

    return loss



if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction="minimize",
    sampler=TPESampler(n_startup_trials = 200, multivariate=False),  # 使用TPE算法; n startup trail = 200 costs 90000s
    pruner=HyperbandPruner(min_resource=1,   
        reduction_factor=3))  # 使用MedianPruner提前终止)
    study.optimize(optuna_objective, n_trials=10000, show_progress_bar=True, n_jobs=2, callbacks=[early_stop(2000)])

    best_params = study.best_trial.params
    print("\n 最优目标函数值 (loss):", study.best_value)

    num_regions = len(CONFIG['region_names'])
    flat_best = [best_params[k] for k in best_params]

    # fit all parameters for each region
    # optimal_params = np.array(flat_best).reshape(num_regions, 5)
    # columns = ['alpha', 'beta', 'delta', 'gamma', 'kappa']
    # df_params = pd.DataFrame(optimal_params, columns=columns, index=CONFIG['region_names'])
    # print("拟合完成，最优参数：")
    # print(df_params)
    # df_params.rename(columns={'gamma': 'lambda'}, inplace=True)
    # df_params.insert(0, 'State', CONFIG['region_names'])

    # only fit beta for each region 
    #df_params = pd.read_csv("D:\MyDownload\Code\OD-COVID\datasets\seird_parameters-v1.csv")
    def vec_to_str(v):
        return "[" + " ".join([f"{x:.8g}" for x in v]) + "]"

    # ... in __main__ after best_params obtained:
    root_path = pathlib.Path(__file__).parent.resolve()
    path = os.path.join(root_path, 'datasets', 'seird_parameters_step1_v3.csv')
    df_params = pd.read_csv(path)

    for i, state in enumerate(CONFIG['region_names']):
        # base values
        alpha0 = float(df_params.loc[df_params['State'] == state, 'alpha'].values[0])
        beta0  = float(df_params.loc[df_params['State'] == state, 'beta'].values[0])
        delta0 = float(df_params.loc[df_params['State'] == state, 'delta'].values[0])

        gamma0 = [float(x) for x in df_params.loc[df_params['State'] == state, 'lambda'].values[0][1:-1].split()]
        mu0    = [float(x) for x in df_params.loc[df_params['State'] == state, 'kappa'].values[0][1:-1].split()]

        E0_0 = int(df_params.loc[df_params['State'] == state, 'E0'].values[0])
        I0_0 = int(df_params.loc[df_params['State'] == state, 'I0'].values[0])

        # scales
        a_s  = best_params[f"alpha_scale_{i}"]
        b_s  = best_params[f"beta_scale_{i}"]
        d_s  = best_params[f"delta_scale_{i}"]
        g_s  = best_params[f"gamma_scale_{i}"]
        mu_s = best_params[f"mu_scale_{i}"]
        e_s  = best_params[f"E0_scale_{i}"]
        i_s  = best_params[f"I0_scale_{i}"]

        # apply
        df_params.loc[df_params['State'] == state, 'alpha'] = a_s * alpha0
        df_params.loc[df_params['State'] == state, 'beta']  = b_s * beta0
        df_params.loc[df_params['State'] == state, 'delta'] = d_s * delta0

        gamma_new = [g_s * v for v in gamma0]
        mu_new    = [mu_s * v for v in mu0]
        df_params.loc[df_params['State'] == state, 'lambda'] = vec_to_str(gamma_new)
        df_params.loc[df_params['State'] == state, 'kappa']  = vec_to_str(mu_new)

        df_params.loc[df_params['State'] == state, 'E0'] = max(1, int(round(e_s * E0_0)))
        df_params.loc[df_params['State'] == state, 'I0'] = max(1, int(round(i_s * I0_0)))

        if i == 1:
            b2_s = best_params[f"beta_scale_2_{i}"]
            d2_s = best_params[f"delta_scale_2_{i}"]
            df_params.loc[df_params['State'] == state, 'beta_2']  = b2_s * beta0
            df_params.loc[df_params['State'] == state, 'delta_2'] = d2_s * delta0

    df_params.to_csv(os.path.join(root_path, 'datasets', 'seird_parameters_step2_v3.csv'), index=False)
    elapsed_seconds = time.time() - start_time
    print(f"time used: {elapsed_seconds:.2f} seconds")


