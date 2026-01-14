from typing import List, Dict, Any

from agents.state import WorkflowState, PopulationState, EpidemicParameters
from agents.epidemic_model import SEIRModel
from agents.epidemic_model_fit import EpidemicModelFit


class CoordinateAgent:
    """Agent for coordinating multiple region agents."""
    def __init__(self, region_ids, simulation_config: Dict[str, Any], logger=None):
        
        self.region_ids = region_ids
        self.simulation_config = simulation_config
        self.logger = logger
        
        self.epidemic_model = SEIRModel(logger=self.logger)
        self.model_fitter = EpidemicModelFit(
            param_bounds=self.simulation_config['param_bounds'], 
            initial_guess=self.simulation_config['initial_guess'])
        
    
    def run(self, state: WorkflowState) -> WorkflowState:
        """Execute the coordinate agent's main workflow."""
        
        # epidemic simulation     (with initial parameters)
        new_state = self.epidemic_model.simulate(
            state, 
            simulation_steps=self.simulation_config['simulation_steps'],
            dt=self.simulation_config['dt'],
            stochastic=self.simulation_config['stochastic'],
            regulation_freq=self.simulation_config['regulation_freq'],
            policy_type=self.simulation_config['policy_type']
        )
        
        # update epidemic state in each region, NO NOT updating parameters here
        # fit_window_size = self.simulation_config['fit_window_size']
        # for region_id in self.region_ids:
        #     region_state = new_state['region_infos'][region_id]
        #     fit_data = region_state.population_history
        #     if len(fit_data) >= fit_window_size:
        #         fit_data = region_state.population_history[-fit_window_size:]
        #         fitted_params = self.model_fitter.fit_v2(fit_data, old_params=region_state.epidemic_params)
        #         #region_state.epidemic_params = fitted_params
        #         region_state.epidemic_inspect.append(fitted_params)
        #         if self.logger:
        #             self.logger.info(f"Updated epidemic parameters for region {region_id}: {fitted_params}")
    
        
        if self.logger:
            self.logger.info("Coordinate Agent completed epidemic simulation and updated states.")
        
        return new_state

