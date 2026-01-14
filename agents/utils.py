import os
import logging
from typing import List, Dict
import colorlog
import pandas as pd
import json
import numpy as np
from logging.handlers import TimedRotatingFileHandler
import re
import ast
import json
import sys
from pathlib import Path
from agents.state import WorkflowState, TransportationState, PopulationState, EpidemicParameters, EpidemicParameters_v2



def setup_logger(task_name, output_path=None, level=logging.INFO):
    
    logger = logging.getLogger(task_name)
    logger.setLevel(level) 

    if not logger.hasHandlers():
        # Create a console handler
        c_handler = colorlog.StreamHandler()
        color_fmt = (
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(reset)s"
            "%(message)s"
        )
        colors = {
            'DEBUG': 'blue',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
        c_handler.setFormatter(colorlog.ColoredFormatter(color_fmt, log_colors=colors))
        logger.addHandler(c_handler)

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            # Create a file handler that logs to a file named with the current date
            log_filename = os.path.join(output_path, f"{task_name}.log")
            f_handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1, backupCount=7)
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            f_handler.setFormatter(f_format)
            f_handler.suffix = "%Y-%m-%d"
            logger.addHandler(f_handler)
    
    return logger


def load_mobility_data(init_state: WorkflowState, mobility_path: str, default_state_num = 5):
    """Load mobility data in date from CSV file."""
    
    def convert_string_to_matrix(s):
        arr = np.fromstring(s.replace('[', '').replace(']', '').replace('\n', ' '), sep=' ')
        return arr.reshape(default_state_num, default_state_num)
    
    if not os.path.exists(mobility_path):
        raise FileNotFoundError(f"Mobility data file not found at {mobility_path}")
    
    if mobility_path.endswith('.csv'):
        mobility_df = pd.read_csv(mobility_path)
    elif mobility_path.endswith('.pkl'):
        print("Loading mobility data from pickle file...")
        mobility_df = pd.read_pickle(mobility_path)
        mobility_df = mobility_df.reset_index().rename(columns={'index': 'date'})
    mobility_df.columns = ['date'] + ['mobility_matrix']
    mobility_df['date'] = mobility_df['date'].str.replace('_', '-')
    mobility_df['date'] = pd.to_datetime(mobility_df['date'])
    
    # convert mobility_matrix from string to numpy array
    if isinstance(mobility_df['mobility_matrix'][0], str):
        print("Converting mobility matrix from string to numpy array...")
        mobility_df['mobility_matrix'] = mobility_df['mobility_matrix'].apply(convert_string_to_matrix)
    
    # sort by date
    mobility_df = mobility_df.sort_values(by='date').reset_index(drop=True)
    
    window_size = init_state['simulation_steps']
    region_names = init_state['region_ids']
    
    length = len(mobility_df) - len(mobility_df) % window_size
    length = max(length, 252)
    mobility_df = mobility_df.iloc[:length].reset_index(drop=True)
    
    for i, region in enumerate(region_names):
        region_state = init_state['region_infos'][region]
        region_mobilities = []
        for step in range(len(mobility_df)):
            matrix = mobility_df.loc[step, 'mobility_matrix']
            inflow = matrix[i, :].tolist()
            outflow = matrix[:, i].tolist()
            
            inflow_map = {region_names[j]: int(inflow[j]) for j in range(len(region_names))}
            outflow_map = {region_names[j]: int(outflow[j]) for j in range(len(region_names))}
            
            trans_state = TransportationState(inflow=inflow_map, outflow=outflow_map)
            region_mobilities.append(trans_state)
        
        # update current_mobility
        #region_state.current_mobility = region_mobilities[window_size-1]
        # update mobility_history
        region_state.mobility_history = []
        #region_state.mobility_history = region_mobilities[:window_size]
        # update gt_mobilities
        region_state.gt_mobilities = region_mobilities
        region_state.current_mobility = region_mobilities[0]
        
        print(f"Loaded mobility data for region {region}, total records: {len(region_state.gt_mobilities)}")
    
    return init_state


def load_population_data(init_state: WorkflowState, population_path: str, path_suffex: str):
    """Load population data in date from CSV file."""
    
    window_size = init_state['simulation_steps']
    
    region_names = init_state['region_ids']
    for region in region_names:
        region_state = init_state['region_infos'][region]
        
        region_file = os.path.join(population_path, f"{region}{path_suffex}")
    
        if not os.path.exists(region_file):
            raise FileNotFoundError(f"Population data file not found at {region_file}")
        
        population_df = pd.read_csv(region_file)
        population_df['date'] = pd.to_datetime(population_df['date'])
        
        length = len(population_df) - len(population_df) % window_size   ##252
        length = max(length, 252)  ##252
        population_df = population_df.iloc[:length].reset_index(drop=True)  #0412-1219 
    
        gt_populations = []
        for step in range(len(population_df)):
            row = population_df.iloc[step]
            pop_state = PopulationState(
                confirmed=int(row['Active']),
                recovered=int(row['Recovered']),
                deaths=int(row['Deaths']),
                Npop=int(row['Population'])
            )
            
            gt_populations.append(pop_state)
        
        # update current_population
        region_state.current_population = gt_populations[0]
        # update population_history
        region_state.population_history = []
        #region_state.population_history = gt_populations[:window_size]
        # update gt_populations
        region_state.gt_populations = gt_populations
        
        print(f"Loaded population data for region {region}, total records: {len(region_state.gt_populations)}")
    
    return init_state


def load_epidemic_params(init_state: WorkflowState, config: Dict):
    """Load or calculate epidemic parameters for each region."""
    epidemic_params_path = config['epidemic_parameters_path']
    if not os.path.exists(epidemic_params_path):
        raise FileNotFoundError(f"Epidemic parameters file not found at {epidemic_params_path}")
    
    epidemic_params_df = pd.read_csv(epidemic_params_path)
    
    region_names = init_state['region_ids']
    for region in region_names:
        region_state = init_state['region_infos'][region]
        params_row = epidemic_params_df[epidemic_params_df['State'].str.lower() == region.lower()]
        if params_row.empty:
            raise ValueError(f"Epidemic parameters for region {region} not found in the file.")
        
        params_row = params_row.iloc[0]
        # epidemic_params = EpidemicParameters(
        #     beta=float(params_row['beta']),
        #     alpha=float(params_row['alpha']),
        #     gamma=float(params_row['gamma']),
        #     mu=float(params_row['mu']),
        # )
        if 'beta_2' in params_row.index and not pd.isna(params_row['beta_2']):
            epidemic_params = EpidemicParameters_v2(
                beta = float(params_row['beta']),
                #beta_knots = json.loads(params_row['beta']),
                alpha = float(params_row['alpha']),
                #amma = float(params_row['lambda']),
                gamma_params = [float(x) for x in params_row['lambda'][1:-1].split()],
                #mu = float(params_row['kappa']),
                mu_params= [float(x) for x in params_row['kappa'][1:-1].split()],
                delta = float(params_row['delta']),
                beta_2 = float(params_row['beta_2']))
        else:
            epidemic_params = EpidemicParameters_v2(
                beta = float(params_row['beta']),
                #beta_knots = json.loads(params_row['beta']),
                alpha = float(params_row['alpha']),
                #amma = float(params_row['lambda']),
                gamma_params = [float(x) for x in params_row['lambda'][1:-1].split()],
                #mu = float(params_row['kappa']),
                mu_params= [float(x) for x in params_row['kappa'][1:-1].split()],
                delta = float(params_row['delta']),
            )
        region_state.epidemic_params = epidemic_params
        # load initial population value
        region_state.current_population.exposed  = int(params_row['E0'])
        region_state.current_population.infected  = int(params_row['I0'])
        region_state.current_population.susceptible = region_state.current_population.Npop - region_state.current_population.exposed - region_state.current_population.infected - region_state.current_population.confirmed - region_state.current_population.recovered - region_state.current_population.deaths

        region_state.epidemic_inspect = []
        print(f"Loaded epidemic parameters for region {region}: {epidemic_params}")
    
    return init_state

def compute_loss(list_df):
    loss = 0.0
    for i, df in enumerate(list_df):
        confirmed_error = (df['Q_pred'] - df['Q_gt'])**2
        deaths_error = (df['D_pred'] - df['D_gt'])**2
        recovered_error = (df['R_pred'] - df['R_gt'])**2

        region_loss = (
            (confirmed_error / (df['Q_gt'].max() + 1e-6)).mean() +
            (deaths_error / (df['D_gt'].max() + 1e-6)).mean() +
            (recovered_error / (df['R_gt'].max() + 1e-6)).mean()
            )
        loss += (region_loss)
    return loss/len(list_df)

def lambda_logistic(a, t):   # a = [L, k, t0]
    return a[0] / (1.0 + np.exp(-a[1]*(t - a[2])))

def kappa_gauss(a, t):       # a = [A, k, t0]
    return a[0] * np.exp(-(a[1]*(t - a[2]))**2)



def extract_policy_responses(log_path: str):
    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    # 匹配整行的 Parsed PolicyResponse 段
    line_pattern = re.compile(r"Parsed PolicyResponse:\s*(.*)")
    results = []
    with log_path.open("r", encoding="utf-8",errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m = line_pattern.search(line)
            if not m:
                continue
            # 拿到 'policy=... explanation=...' 这一整段
            payload = m.group(1).strip()
            # 再用一个正则，把 policy 和 explanation 拆开
            # 注意这里 policy 用贪婪匹配，到最后一个 " explanation=" 为止
            m2 = re.match(r"policy=(\{.*\})\s+explanation=(.*)", payload)
            if not m2:
                # 如果格式跟预期不符，可以打印出来调试
                print(f"[WARN] Cannot parse payload: {payload}", file=sys.stderr)
                continue
            policy_str = m2.group(1)
            explanation_str = m2.group(2)
            # 组装成一个完整的 Python 字面量，再用 ast.literal_eval 解析
            dict_source = "{'policy': " + policy_str + ", 'explanation': " + explanation_str + "}"
            try:
                obj = ast.literal_eval(dict_source)
                results.append(obj)
            except Exception as e:
                print(f"[ERROR] Failed to eval: {e}", file=sys.stderr)
                print(f"         source: {dict_source[:200]}...", file=sys.stderr)

    return results