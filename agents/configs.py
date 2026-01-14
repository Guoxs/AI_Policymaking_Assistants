import os
import pathlib

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
import pandas as pd

## 20 state 
state_name = ['alabama','arizona', 'arkansas', 'idaho','indiana', 'iowa', 'kentucky', 'michigan','minnesota', 'mississippi', 'nebraska','new mexico','ohio','oklahoma','south carolina', 'tennessee', 'texas', 'utah','virginia','wisconsin']
## 5 state
#state_name = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']

if len(state_name) == 5:
    CONFIG = {
        'output_path': os.path.join(ROOT_PATH, 'outputs', '5 states', 'detection'),
        'dataset_path': os.path.join(ROOT_PATH, 'datasets', '5 states'),
        'benchmark_path': os.path.join(ROOT_PATH, 'outputs','baseline-5states','results'),
        'region_names': state_name,
        'path_suffex': '_cases_0412_1231.csv',
        'mobility_path': os.path.join(ROOT_PATH, 'datasets', '5 states', 'all_mobility_0412_1231.csv'),
        'epidemic_parameters_path': os.path.join(ROOT_PATH, 'datasets','5 states', 'seird_parameters_step2_final.csv'),  ###5756, 3241
        'start_date': '2020-04-12',
        'init_epidemic_params_from_file': True,
        'decision_type': 'no_action', ##'agent','rule','load_policty','no_action','random'
        'simulation_config': {
            'simulation_steps': 14,  # number of simulation steps in each iteration
            'fit_window_size': 21,  # number of days to use for fitting the model
            'dt': 1.0,              # time step for each simulation step (in days)
            'stochastic': False,
            'detection_rate': 0.9,  # for detection policy
            
            # for model fitting
            'param_bounds': [
                (0.02, 5),       # beta
                (1/14, 1/5),    # gamma
                (1/200, 1/33),  # mu
                (0.1, 0.5),    # alpha
            ],
            
            'initial_guess': [0.5, 0.2, 0.01, 0.2],
            'regulation_freq': 'weekly',  # 'daily', 'weekly' 
            'policy_type': 'detection',  ## 'ori_restriction'  or 'reallocation', 'detection
        },
            
        'llm_config': {
            'model': 'gpt-3.5-turbo-0125', # gpt-4o-2024-11-20，  
            #'model': 'llama-3-8b-instruct',  #"deekseek-r1" "gpt-4.1-mini"， "gemini-2.5-flash-thinking"， "llama-3-8b-instruct"
            'base_url': "https://api.nuwaapi.com/v1", 
            'api_key': "sk-LV9lxYyADUWvCrj6UK59WFR3anyn40nDyBJpK8iZv5Cb2bDa",
            'temperature': 0.7,
        },
    #     'llm_config': {
    #     'model': 'qwen2.5-72b-instruct',
    #     'base_url': "https://vip.yi-zhan.top/v1", 
    #     'api_key': "sk-ZDAw2OabbTueKBqu372aEaBeE14e46A7B8Fc655438C45683",
    #     'model_provider': 'openai',
    #     'temperature': 0.8,
    #     'max_tokens': 4096,
    # },

    }
else:
    CONFIG = {
        
        'output_path': os.path.join(ROOT_PATH, 'outputs','US', 'detection'),
        'dataset_path_us': os.path.join(ROOT_PATH, 'datasets', 'US', 'epedimic'),
        'benchmark_path': os.path.join(ROOT_PATH, 'outputs','baseline-us','results'),
        'region_names': state_name,

        'path_suffex': '_cases.csv',
        'dataset_path': os.path.join(ROOT_PATH, 'datasets', 'US', 'epedimic'),
        'mobility_path': os.path.join(ROOT_PATH, 'datasets', 'US', 'all_mobility_us_0412_1231_20states.pkl'),
        'epidemic_parameters_path': os.path.join(ROOT_PATH, 'datasets', 'US', 'seird_parameters_step2_v3_0.18.csv'),
        
        'start_date': '2020-04-12',
        'init_epidemic_params_from_file': True,
        'decision_type': 'agent', ##'agent','rule','load_policty','no_action'

        'simulation_config': {
            'simulation_steps': 14,  # number of simulation steps in each iteration
            'fit_window_size': 21,  # number of days to use for fitting the model
            'dt': 1.0,              # time step for each simulation step (in days)
            'stochastic': False,
            'policy_type': 'detection',  ## 'ori_restriction'  or 'reallocation', 'detection'
            'detection_rate': 0.9,  # for detection policy
            
            # for model fitting
            'param_bounds': [
                (0.02, 5),       # beta
                (1/14, 1/5),    # gamma
                (1/200, 1/33),  # mu
                (0.1, 0.5),    # alpha
            ],
            
            'initial_guess': [0.5, 0.2, 0.01, 0.2],
            'regulation_freq': 'weekly',  # 'daily', 'weekly' 
        },
            
        'llm_config': {
            'model': 'gpt-4.1-mini', # gpt-4o-2024-11-20 
            #'model': 'GPT-4.1-2025-04-14',
            'base_url': "https://api.nuwaapi.com/v1", 
            'api_key': "sk-LV9lxYyADUWvCrj6UK59WFR3anyn40nDyBJpK8iZv5Cb2bDa",
            'temperature': 0.7,
        },

    }