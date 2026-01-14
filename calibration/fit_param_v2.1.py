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


def objective(params):

    init_state = WorkflowState()
    init_state['start_date'] = CONFIG['start_date']
    init_state['region_ids'] = CONFIG['region_names']
    init_state['simulation_steps'] = CONFIG['simulation_config']['simulation_steps']
    init_state['region_infos'] = {}
    
    # init region_infos
    for region_id in init_state['region_ids']:
        region_state = RegionState()
        region_state.region_id = region_id
        region_state.neighboring_region_ids = [r for r in init_state['region_ids'] if r != region_id]
        init_state['region_infos'][region_id] = region_state
    
    # load mobility data and ground truth populations
    init_state = load_mobility_data(init_state, CONFIG['mobility_path'])
    
    # load population data
    init_state = load_population_data(init_state, CONFIG['dataset_path'], CONFIG['path_suffex'])
    
    # load epidemic parameters from file
    region_num = len(CONFIG['region_names'])
    #params_shaped = params.reshape(region_num, -1)
    for i, region in enumerate(CONFIG['region_names']):
        region_state = init_state['region_infos'][region]
        params_row = params[i]
        if len(params_row) > 7:
            epidemic_params = EpidemicParameters_v2(
            alpha = float(params_row[0]),
            beta = float(params_row[1]),
            delta = float(params_row[2]),
            # gamma = float(params_row[3]),
            # kappa = float(params_row[4]))
            gamma_params= params_row[3],
            mu_params= params_row[4],
            beta_2 = float(params_row[7]))
        else:
            epidemic_params = EpidemicParameters_v2(
            alpha = float(params_row[0]),
            beta = float(params_row[1]),
            delta = float(params_row[2]),
            # gamma = float(params_row[3]),
            # kappa = float(params_row[4]))
            gamma_params= params_row[3],
            mu_params= params_row[4])
        region_state.epidemic_params = epidemic_params
        # region_state.current_population.infected = int(init_state['region_infos'][region].gt_populations[0].Npop *  0.0001)
        # region_state.current_population.exposed  = region_state.current_population.infected * 3
        region_state.current_population.exposed  = int(params_row[5])
        region_state.current_population.infected  = int(params_row[6])
        region_state.current_population.susceptible = region_state.current_population.Npop - region_state.current_population.exposed - region_state.current_population.infected - region_state.current_population.confirmed - region_state.current_population.recovered - region_state.current_population.deaths

    mobility_length = min([
        len(init_state['region_infos'][r].gt_mobilities) 
        for r in init_state['region_ids']
    ])
    
    population_length = min([
        len(init_state['region_infos'][r].gt_populations) 
        for r in init_state['region_ids']
    ])
    
    init_state['max_iterations'] = min(mobility_length, population_length) // init_state['simulation_steps']
    
    workflow = ODWorkflow(CONFIG, logger_path='no_path')
    
    # Run workflow with disturbance capabilities
    list_df = workflow.run(init_state)
    return list_df

# def compute_loss(list_df):
#     loss = 0.0
#     for i, df in enumerate(list_df):
#         confirmed_error = (df['Q_pred'] - df['Q_gt'])**2
#         deaths_error = (df['D_pred'] - df['D_gt'])**2
#         recovered_error = (df['R_pred'] - df['R_gt'])**2

#         region_loss = (
#             (confirmed_error / (df['Q_gt'].max() + 1e-6)).mean() +
#             (deaths_error / (df['D_gt'].max() + 1e-6)).mean() +
#             (recovered_error / (df['R_gt'].max() + 1e-6)).mean()
#             )
#         loss += (region_loss)
#     return loss/len(list_df)

def compute_loss(list_df):
    loss = 0.0
    for i, df in enumerate(list_df):
        Q_pred = df['Q_pred'].astype(float)
        Q_gt   = df['Q_gt'].astype(float)
        D_pred = df['D_pred'].astype(float)
        D_gt   = df['D_gt'].astype(float)
        R_pred = df['R_pred'].astype(float)
        R_gt   = df['R_gt'].astype(float)
        confirmed_error = (Q_pred - Q_gt)**2
        deaths_error    = (D_pred - D_gt)**2
        recovered_error = (R_pred - R_gt)**2
        eps = 1e-6
        region_loss = (
            (confirmed_error / (Q_gt.max() + eps)).mean() +
            (deaths_error    / (D_gt.max() + eps)).mean() +
            (recovered_error / (R_gt.max() + eps)).mean()
        )
        if i==1:
            region_loss += 4 *(confirmed_error / (Q_gt.max() + eps)).mean()
        loss += region_loss
    return loss / len(list_df)

def compute_loss_3(list_df):
    loss = 0.0
    state_loss = {}
    for i, df in enumerate(list_df):
        last_100 = df.tail(100).copy()
        float_cols = ['Q_pred', 'Q_gt', 'D_pred', 'D_gt', 'R_pred', 'R_gt']
        # Convert in a single pass, safely handling errors
        for col in float_cols:
            last_100[col] = last_100[col].astype(float)
        confirmed_error = (last_100['Q_pred'] - last_100['Q_gt'])**2
        deaths_error = (last_100['D_pred'] - last_100['D_gt'])**2
        has_R =  not np.all(df['R_gt'] == 0) and not np.any(np.isnan(df['R_gt']))
        Q_min = df['Q_gt'].quantile(0.50)
        D_min = df['D_gt'].quantile(0.50)
        denom_Q = np.maximum(last_100['Q_gt'], Q_min)**2 + 1e-6
        denom_D = np.maximum(last_100['D_gt'], D_min)**2 + 1e-6
        confirmed_loss = (confirmed_error / denom_Q).mean()
        deaths_loss = (deaths_error / denom_D).mean()
        if has_R:
            recovered_error = (last_100['R_pred'] - last_100['R_gt'])**2
            R_min = df['R_gt'].quantile(0.50)
            denom_R = np.maximum(last_100['R_gt'], R_min)**2 + 1e-6
            recovered_loss = (recovered_error / denom_R).mean()
            region_loss = confirmed_loss * 1.5 + deaths_loss + recovered_loss    #考虑给confirm loss 乘一个系数
            region_loss = region_loss/3
        else:
            region_loss = confirmed_loss * 1.5 + deaths_loss
            region_loss = region_loss/2
        loss += (region_loss)
        state_loss[CONFIG['region_names'][i]] = region_loss
    return loss/len(list_df)

def optuna_objective(trial):
    num_regions = len(CONFIG['region_names'])
    #obtain_param = pd.read_csv("D:\MyDownload\Code\OD-COVID\datasets\seird_parameters-v1.csv")
    root_path = pathlib.Path(__file__).parent.resolve()
    # path =  os.path.join(root_path, 'datasets','seird_parameters_step2_6340.csv')
    path =  os.path.join(root_path, 'datasets','seird_parameters_step1_v2.csv')
    obtain_param = pd.read_csv(path)
    params = []
    for i in range(num_regions):
        row = obtain_param[obtain_param['State'] == CONFIG['region_names'][i]].iloc[0]
        alpha = row['alpha']
        # delta = row['delta']
        # beta = row['beta']
        # kappa = row['kappa']
        para = row['lambda'][1:-1].split() 
        gamma_para = [float(x) for x in para]
        para = row['kappa'][1:-1].split()
        kappa_para = [float(x) for x in para]
        E0 = row['E0']
        I0 = row['I0']
        # alpha = trial.suggest_float(f"alpha_{i}", 0.1, 0.5)
 
        beta_scale = trial.suggest_float(
            f"beta_scale_{i}",
            0.2, 1.5,                 # 或 0.5–1.5，看你对 step1 的信心
            log=False
        )

        delta_scale = trial.suggest_float(
            f"delta_scale_{i}",
            0.2, 1.5,
            log=False
        )
        beta = row['beta']   * beta_scale
        delta = row['delta'] * delta_scale
        if i == 1 :
            beta_scale_2 = trial.suggest_float(
                f"beta_scale_2_{i}", 0.2, 1.5, log=False
            )
            beta_2 = row['beta'] * beta_scale_2
            # delta_scale_2 = trial.suggest_float(
            #     f"delta_scale_2_{i}", 0.2, 1.5, log=False
            # )
            # delta_2 = row['delta'] * delta_scale_2
            params.append([alpha, beta, delta, gamma_para, kappa_para, E0, I0, beta_2])
        else:
            params.append([alpha, beta, delta, gamma_para, kappa_para, E0, I0])
        # delta = trial.suggest_float(f"delta_{i}", 0.1, 1.0)
        # gamma = trial.suggest_float(f"gamma_{i}", 0.01, 0.2)
        # kappa = trial.suggest_float(f"kappa_{i}", 0.0, 0.05)
        # E0 = trial.suggest_int(f"E0_{i}", 100, 10000)
        # I0 = trial.suggest_int(f"I0_{i}", 100, 10000)
        # params.extend([alpha, beta, delta, gamma, kappa])
        params.append([alpha, beta, delta, gamma_para, kappa_para, E0, I0])

    #params = np.array(params)
    list_df = objective(params)
    loss = compute_loss(list_df)
    return loss


def early_stop(patience=2000):
    best, stale = [float('inf')], [0]
    def cb(study, trial):
        v = trial.value
        if v is None:
            return
        if v + 1e-12 < best[0]:
            best[0], stale[0] = v, 0
        else:
            stale[0] += 1
            if stale[0] >= patience:
                study.stop()
    return cb



if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction="minimize",
    sampler=TPESampler(n_startup_trials = 200, multivariate=False),  # 使用TPE算法; n startup trail = 200 costs 90000s
    pruner=HyperbandPruner(min_resource=1,   
        reduction_factor=3))  # 使用MedianPruner提前终止)
    study.optimize(optuna_objective, n_trials=10000, show_progress_bar=True, n_jobs=1, callbacks=[early_stop(2000)])

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
    root_path = pathlib.Path(__file__).parent.resolve()
    path =  os.path.join(root_path, 'datasets','seird_parameters_step1_v2.csv')
    df_params = pd.read_csv(path)
    flat_best = list(best_params.values())
    for i, state in enumerate(CONFIG['region_names']):
        beta_scale = best_params[f"beta_scale_{i}"]
        df_params.loc[df_params['State'] == state, 'beta'] *= beta_scale
        delta_scale = best_params[f"delta_scale_{i}"]
        df_params.loc[df_params['State'] == state, 'delta'] *= delta_scale
        if i==1:
            beta_scale_2 = best_params[f"beta_scale_2_{i}"]
            df_params.loc[df_params['State'] == state, 'beta_2'] = df_params.loc[df_params['State'] == state, 'beta'] * beta_scale_2
    # print("拟合完成，最优参数：")
    print(df_params)

    #df_params.to_csv('D:\MyDownload\Code\OD-COVID\datasets\seird_parameters.csv', index=False)
    df_params.to_csv(os.path.join(root_path, 'datasets','seird_parameters_step2_v2.csv'), index=False)
    elapsed_seconds = time.time() - start_time
    print(f"time used: {elapsed_seconds:.2f} seconds")


