from typing import List, Dict, Any, Optional

import copy
import os
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from agents.state import WorkflowState, RegionState

from agents.region_agent import RegionAgent
from agents.coordinate_agent import CoordinateAgent

from agents.utils import load_epidemic_params, compute_loss


class ODWorkflow:
    """Workflow to manage region agents and coordinate agent."""
    def __init__(self, configs: Dict[str, Any], logger=None, logger_path: str=""):
        
        self.configs = configs
        self.logger = logger
        self.logger_path = logger_path
        self.benchmark_path = self.configs['benchmark_path']
        self.region_names = self.configs['region_names']
        self.llm_config = self.configs['llm_config']
        self.simulation_config = self.configs['simulation_config']
        self.decision_type = self.configs['decision_type']
        
        try:
            assert len(self.region_names) > 0, "Region profiles cannot be empty."
            
            self.region_agents = {}
            for name in self.region_names:
                region_agent = RegionAgent(name, self.llm_config, self.simulation_config, self.logger_path, self.decision_type)
                self.region_agents[name] = region_agent
            
            self.coordinate_agent = CoordinateAgent(self.region_names, self.simulation_config, self.logger)
            
            if self.logger:
                self.logger.info("Workflow initialized successfully.")
        
        except AssertionError as e:
            if self.logger:
                self.logger.error(f"Workflow initialization error: {e}")
            raise


    def run(self, state: WorkflowState, trial = None) -> Dict[str, Any]:
        """Run the complete workflow."""
        max_iterations = state['max_iterations']
                
        for iteration in range(max_iterations):
            if self.logger:
                self.logger.info(f"=== Workflow Iteration {iteration+1}/{max_iterations} ===")
            
            state['current_iteration_step'] = iteration
                        
            if iteration >= 0:
                # given historical state, run each region agent to get policies
                for region_id, agent in self.region_agents.items():
                    if self.logger:
                        self.logger.info(f"--- Running Region Agent {region_id} for Policy Making ---")
                    if self.decision_type == 'agent':
                        policy_response = agent.run(state)    ##agent
                    elif self.decision_type == 'rule':
                        policy_response = agent.rule_based_policy(state, self.simulation_config) ###rule based
                    elif self.decision_type == 'load_policy':
                        policy_response = agent.get_loaded_policy()       ###load from log
                    elif self.decision_type == 'no_action':
                        policy_response = None    #no action for now      ## no action
                    elif self.decision_type == 'random':
                        policy_response = agent.random_policy(state)  ## random action
                    else:
                        raise ValueError(f"Unknown decision type: {self.decision_type}")
                    state['region_infos'][region_id].policy_response = policy_response
                    
                    if self.logger:
                        self.logger.info(f"Region {region_id} policy response:\n {policy_response}")
            
            # Run coordinate agent to simulate epidemic spread
            if self.logger:
                self.logger.info(f"--- Running Coordinate Agent for Epidemic Parameter Update ---")
            state = self.coordinate_agent.run(state)

            if trial is not None:
                if (iteration + 1) % 2 == 0 or (iteration + 1) == max_iterations:
                    df = self.draw_results(state)
                    loss = compute_loss(df)
                    trial.report(loss, step=iteration + 1)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        
        # save predicted population states in files and draw figures
        if self.logger:
            self.logger.info("Workflow completed all iterations.")
        
        if self.logger_path == 'no_path':
            df = self.get_analysis(state)
        else:
            result_dir = os.path.join(self.logger_path, 'results')
            os.makedirs(result_dir, exist_ok=True)
            df = self.draw_results(state, result_dir)
            # self.draw_inspect(state, result_dir)
        
        for region_id in state['region_infos']:
            mobility_history = state['region_infos'][region_id].mobility_history
            result_dir = os.path.join(self.logger_path, 'mobility')
            os.makedirs(result_dir, exist_ok=True)
            mobility_path = os.path.join(result_dir, f"{region_id}_mobility.pkl")
            with open(mobility_path, 'wb') as f:
                pickle.dump(mobility_history, f)

        if self.logger:
            self.logger.info(f"Results saved in {result_dir}")
        return df
        
    
    def draw_results(self, state: WorkflowState, output_dir = None):
        """Draw and save results."""
        list_df = []
        for region_id, region_state in state['region_infos'].items():
            gt_populations = region_state.gt_populations
            pred_populations = region_state.population_history
            
            start_date = pd.to_datetime(state['start_date'])
            min_length = min(len(gt_populations), len(pred_populations))
            dates = [start_date + pd.Timedelta(days=i) for i in range(min_length)]
            
            pd_data = {
                'date': dates,
                'Q_gt': [p.confirmed for p in gt_populations[:min_length]],
                'Q_pred': [p.confirmed for p in pred_populations[:min_length]],
                'R_gt': [p.recovered for p in gt_populations[:min_length]],
                'R_pred': [p.recovered for p in pred_populations[:min_length]],
                'D_gt': [p.deaths for p in gt_populations[:min_length]],
                'D_pred': [p.deaths for p in pred_populations[:min_length]],
            }
            
            df = pd.DataFrame(pd_data)
            list_df.append(df)
            if  output_dir:
                csv_path = os.path.join(output_dir, f"{region_id}_results.csv")
                df.to_csv(csv_path, index=False)
            
                #obtain the benchmark
                benchmark_path = os.path.join(self.benchmark_path, f"{region_id}_results.csv")
                if not os.path.exists(benchmark_path):
                    benchmark = False
                else:
                    benchmark = True
                    df_no_policy = pd.read_csv(benchmark_path)
                    min_length = min(len(df), len(df_no_policy))
                    df = df.iloc[:min_length]
                    df_no_policy = df_no_policy.iloc[:min_length]
                # draw subplots
                fig, axs = plt.subplots(3, 1, figsize=(16, 16))
                compartments = ['Q', 'R', 'D']
                for i, comp in enumerate(compartments):
                    ax = axs[i]
                    ax.plot(df['date'], df[f'{comp}_gt'], label=f'{comp} (GT)', linestyle='--')
                    ax.plot(df['date'], df[f'{comp}_pred'], label=f'{comp} (Policy)', color='red', linestyle='-')
                    if benchmark:
                        ax.plot(df['date'], df_no_policy[f'{comp}_pred'], label=f'{comp} (No Policy)', linestyle=':')
                    ax.set_xlabel('Date')
                    # make x label vertical and rotate 45 degrees
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    ax.set_ylabel('Population')
                    ax.set_title(f'{comp} over Time in {region_id}')
                    ax.legend()
                    ax.grid()
                
                plt.tight_layout()
                plt_path = os.path.join(output_dir, f"{region_id}_results_subplots.png")
                plt.savefig(plt_path)
                plt.close()
        return list_df
    
    
    def draw_inspect(self, state: WorkflowState, output_dir = None):
        """Draw and save epidemic parameter inspection results."""
        for region_id, region_state in state['region_infos'].items():
            inspect_params = region_state.epidemic_inspect
            if len(inspect_params) == 0:
                continue
            # alphas = [p.alpha for p in inspect_params]
            betas = [p.beta for p in inspect_params]
            # deltas = [p.delta for p in inspect_params]
            # gammas = [p.gamma for p in inspect_params]
            mus = [p.mu for p in inspect_params]
            
            iterations = list(range(1, len(inspect_params)+1))
            df = pd.DataFrame({
                'iteration': iterations,
                # 'alpha': alphas,
                'beta': betas,
                # 'delta': deltas,
                # 'gamma': gammas,
                'mu': mus
            })
            fig, axs = plt.subplots(2, 1, figsize=(12, 12))     
            benchmark_path = os.path.join(self.benchmark_path, f"{region_id}_inspects.csv")
            df_no_policy = pd.read_csv(benchmark_path)
            beta_bs = df_no_policy['beta'].tolist()
            mu_bs = df_no_policy['mu'].tolist()
            # beta_bs = betas
            # mu_bs = mus
            param_list = [
                # (alphas, 'Alpha (Incubation Rate)'),
                (betas, beta_bs, 'Beta (Infection Rate)'),
                # (deltas, 'Delta (Confirmation Rate)'),
                # (gammas, 'Gamma (Recovery Rate)'),
                (mus, mu_bs, 'Mu (Mortality Rate)')
            ]
            for i, (param_values, param_baseline, param_name) in enumerate(param_list):
                ax = axs[i]
                ax.plot(iterations, param_values, marker='o', label='Policy', color='red')
                ax.plot(iterations, param_baseline, marker='x', label='GT')
                ax.set_xlabel('Iteration')
                ax.set_ylabel(param_name)
                ax.set_title(f'{param_name} over Iterations in {region_id}')
                ax.grid()
                ax.legend()
                plt.tight_layout()
            
            if  output_dir:
                csv_path = os.path.join(output_dir, f"{region_id}_inspects.csv")
                df.to_csv(csv_path, index=False)
                plt_path = os.path.join(output_dir, f"{region_id}_epidemic_params_inspect.png")
                plt.savefig(plt_path)
            plt.close()

    def get_analysis(self, state: WorkflowState):
        list_df = []
        for region_id, region_state in state['region_infos'].items():
            gt_populations = region_state.gt_populations
            pred_populations = region_state.population_history
            
            start_date = pd.to_datetime(state['start_date'])
            min_length = min(len(gt_populations), len(pred_populations))
            dates = [start_date + pd.Timedelta(days=i) for i in range(min_length)]
            
            pd_data = {
                'date': dates,
                'Q_gt': [p.confirmed for p in gt_populations[:min_length]],
                'Q_pred': [p.confirmed for p in pred_populations[:min_length]],
                'R_gt': [p.recovered for p in gt_populations[:min_length]],
                'R_pred': [p.recovered for p in pred_populations[:min_length]],
                'D_gt': [p.deaths for p in gt_populations[:min_length]],
                'D_pred': [p.deaths for p in pred_populations[:min_length]],
            }
            
            df = pd.DataFrame(pd_data)
            list_df.append(df)
        return list_df