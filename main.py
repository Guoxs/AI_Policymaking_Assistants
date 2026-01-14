import os
import pandas as pd
from datetime import datetime

from agents.workflow import ODWorkflow
from agents.configs import CONFIG

from agents.state import WorkflowState, RegionState
from calibration.fit_param_v import compute_loss
from agents.utils import setup_logger, load_mobility_data, load_population_data,load_epidemic_params

def main():
    model_name = CONFIG['llm_config']['model']
    if CONFIG['decision_type'] == 'rule':
        exp_name = f"rule_based_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    elif CONFIG['decision_type'] == 'no_action':
        exp_name = f"no_action_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    elif CONFIG['decision_type'] == 'agent':
        exp_name = f"{model_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    elif CONFIG['decision_type'] == 'random':
        exp_name = f"random_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    elif CONFIG['decision_type'] == 'load_policy':
        exp_name = f"load_policy_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    logger_path = CONFIG['output_path'] + f'/{exp_name}'
    os.makedirs(logger_path, exist_ok=True)
    
    logger = setup_logger('od_agent', logger_path)
    
    logger.info("=== Initializing WorkflowState ===")
    
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
    logger.info("Loading mobility data...")
    init_state = load_mobility_data(init_state, CONFIG['mobility_path'], len(init_state['region_ids']))
    
    # load population data
    logger.info("Loading population data...")
    init_state = load_population_data(init_state, CONFIG['dataset_path'], CONFIG['path_suffex'])
    
    # load epidemic parameters from file
    logger.info(f"--- Loading Epidemic Parameter from File ---")
    init_state = load_epidemic_params(init_state, CONFIG)

    mobility_length = min([
        len(init_state['region_infos'][r].gt_mobilities) 
        for r in init_state['region_ids']
    ])
    
    population_length = min([
        len(init_state['region_infos'][r].gt_populations) 
        for r in init_state['region_ids']
    ])
    
    init_state['max_iterations'] = min(mobility_length, population_length) // init_state['simulation_steps'] 
    if init_state['max_iterations'] * init_state['simulation_steps'] < 252:
        init_state['max_iterations'] += 1

    logger.info(f"Initialized WorkflowState with {len(init_state['region_ids'])} regions, "
                f"max_iterations: {init_state['max_iterations']}, "
                f"simulation_steps: {init_state['simulation_steps']}, "
                f"mobility data length: {mobility_length}, "
                f"population data length: {population_length}.")
    
    logger.info("=== Initializing Multi-Agent Workflow ===")
    workflow = ODWorkflow(CONFIG, logger, logger_path)
    
    # Run workflow with disturbance capabilities
    logger.info("=== Running Multi-Agent Workflow ===")
    result = workflow.run(init_state)
    loss = compute_loss(result)
    print(f"Final computed loss: {loss}")
    # Display results
    logger.info("=== Finished ===")
        

if __name__ == "__main__":
    for i in range(1):
        main()
        
        

        
        
        
    