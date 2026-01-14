from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from  agents.configs import CONFIG
from agents.utils import *
from scipy.stats import poisson


from agents.state import PopulationState, TransportationState, WorkflowState




class EpidemicModel(ABC):
    """Abstract base class for epidemic models."""  
    
    def __init__(self, logger=None, max_iterations=12):
        super().__init__()
        self.T = CONFIG['simulation_config']['simulation_steps'] * max_iterations # 全时长天数
        self.decision_type = CONFIG['decision_type']
        self.logger = logger
        
    @abstractmethod
    def step(self, state: WorkflowState, **kwargs) -> WorkflowState:
        """Perform one simulation step."""
        pass
    
    @abstractmethod
    def simulate(self, state: WorkflowState, silumation_steps: int, dt: float, **kwargs) -> WorkflowState:
        """Run full simulation."""
        pass



class SEIRModel(EpidemicModel):
    """
    Standard SEIR epidemic model with optional mobility.
    
    Supports both deterministic and stochastic simulation modes.
    """
    
    def _seir_derivatives(self, 
                          state: WorkflowState, 
                          region_id: str,
                          curr_trans_state: TransportationState,
                          t: int,
                          detection_states: dict) -> List[float]:
        """Calculate SEIR derivatives for a single region."""
        
        region_state = state['region_infos'].get(region_id)
        num_state = len(state['region_ids'])
        pop_state = region_state.current_population
        total_pop = pop_state.total_population()
        S, E, I, Q, R, D = pop_state.susceptible, pop_state.exposed, pop_state.infected, pop_state.confirmed, pop_state.recovered, pop_state.deaths

        epidemic_params = region_state.epidemic_params
        # beta_t = self.piecewise_const(epidemic_params.beta_knots,  int(t))
        # delta_t = epidemic_params.delta
        # delta_t = self.piecewise_const(epidemic_params.delta_knots, int(t))
        # beta_t = epidemic_params.beta
        # delta_t = epidemic_params.delta
        if (t>=160) and epidemic_params.beta_2>0:
            beta_t = epidemic_params.beta_2
        else:
            beta_t = epidemic_params.beta
        delta_t = epidemic_params.delta
        gamma_t = lambda_logistic(epidemic_params.gamma_params, t)
        mu_t = kappa_gauss(epidemic_params.mu_params, t)
        epidemic_params.gamma = gamma_t
        epidemic_params.mu = mu_t
        region_state.epidemic_params = epidemic_params
        # Basic SEIR dynamics
        new_infection = beta_t * S * (I + 0.5 * Q) / total_pop if total_pop > 0 else 0

        # Add mobility effects if available
        mobility_infection = 0.0
        new_S = 0.0
        new_I = 0.0
        new_Q = 0.0
        new_E = 0.0
        new_R = 0.0
        inflow = curr_trans_state.inflow
        ##### S R parameters update for 5 states
        target_state = None
        if region_id in detection_states and detection_states[region_id] is not None:
            target_state = detection_states[region_id]
            detect_rate = CONFIG['simulation_config']['detection_rate']

        for neighbor_id, flow in inflow.items():
            if neighbor_id == region_id:
                continue
            neighbor_state = state['region_infos'].get(neighbor_id)
            if neighbor_state:  
                neighbor_pop = neighbor_state.current_population
                neighbor_total = neighbor_pop.total_population()
                if neighbor_total > 0 and total_pop > 0:
                    infect_traveler_I = (flow) * (neighbor_pop.infected) / (neighbor_total-neighbor_pop.deaths + 1e-5)
                    infect_traveler_Q = (flow) * (neighbor_pop.confirmed) / (neighbor_total-neighbor_pop.deaths + 1e-5)
                    infect_traveler_E = (flow) * (neighbor_pop.exposed) / (neighbor_total-neighbor_pop.deaths + 1e-5)
                    infect_traveler_R = (flow) * (neighbor_pop.recovered) / (neighbor_total-neighbor_pop.deaths + 1e-5)
                    mobility_infection += beta_t *  (infect_traveler_I + 0.5 * infect_traveler_Q) * S / total_pop
                    if target_state is not None and neighbor_id in target_state:
                        infect_traveler_Q = infect_traveler_Q + infect_traveler_I * detect_rate + infect_traveler_E * detect_rate
                        infect_traveler_I = infect_traveler_I * (1 - detect_rate)
                        infect_traveler_E = infect_traveler_E * (1 - detect_rate)
                    new_I += infect_traveler_I
                    new_Q += infect_traveler_Q
                    new_E += infect_traveler_E
                    new_S += (flow) * neighbor_pop.susceptible / (neighbor_total-neighbor_pop.deaths + 1e-5)
                    new_R += (infect_traveler_R)
            
        outflow = curr_trans_state.outflow
        sum_outflow = sum(flow for neighbor_id, flow in outflow.items() if neighbor_id != region_id)
        out_I = (sum_outflow) * I / (total_pop - D) if (total_pop - D) > 0 else 0
        out_Q = (sum_outflow) * Q / (total_pop - D) if (total_pop - D) > 0 else 0
        out_E = (sum_outflow) * E / (total_pop - D) if (total_pop - D) > 0 else 0
        out_S = (sum_outflow) * S / (total_pop - D) if (total_pop - D) > 0 else 0
        out_R = (sum_outflow) * R / (total_pop - D) if (total_pop - D) > 0 else 0
        # new_infection = new_infection * (1 - (sum_outflow/num_step) / total_pop) 

        #total_infection = (new_infection + mobility_infection)
        total_infection = new_infection 
        # Derivatives
        if num_state == 5:
            dS = -total_infection + new_S - out_S   
            dE = total_infection - epidemic_params.alpha * E + new_E - out_E
            dQ =  delta_t * I - gamma_t * Q - mu_t * Q 
        else:
            ##this term (new_S - out_S) not in US all to reduce complexity
            dS = -total_infection - mobility_infection
            dE = total_infection + mobility_infection - epidemic_params.alpha * E + new_E - out_E
            dQ = delta_t * I - gamma_t * Q - mu_t * Q + new_Q - out_Q
        dI = epidemic_params.alpha * E - delta_t * I + new_I - out_I
        dR = gamma_t * Q 
        dD = mu_t * Q

        return [dS, dE, dI, dQ, dR, dD]

    def piecewise_const(self, values_at_knots, t) -> float:

        # values_at_knots: len = len(KNOTS)
        # 返回时刻 t 的值
        import bisect
        idx = min(len(self.KNOTS)-1, max(0, bisect.bisect_right(self.KNOTS, t)-1))
        return float(values_at_knots[idx])

    def step(self, 
             state: WorkflowState, 
             region_id: str,
             curr_trans_state: TransportationState,
             dt: float,
             t: int,
             stochastic: bool = False,
             detection_states: dict = None) -> WorkflowState:
        """
        Perform one simulation step.
        
        Args:
            state: Current workflow state
            region_id: Region name to simulate
            curr_trans_state: Current transportation state
            dt: Time step
            stochastic: Whether to use stochastic updates
            
        Returns:
            Updated population state
        """
        dS, dE, dI, dQ, dR, dD = self._seir_derivatives(state, region_id, curr_trans_state, t, detection_states)

        # Get current population state for the region
        region_state = state['region_infos'].get(region_id)
        current_pop = region_state.current_population
        
        # Create new population state by copying current values
        new_pop_state = PopulationState(
            susceptible=current_pop.susceptible,
            exposed=current_pop.exposed,
            infected=current_pop.infected,
            confirmed = current_pop.confirmed,
            recovered=current_pop.recovered,
            deaths=current_pop.deaths
        )
        
        if stochastic:
            # Stochastic updates using Poisson process
            infections = poisson.rvs(max(0, -dS * dt))
            exposures = poisson.rvs(max(0, dE * dt + infections))
            recoveries = poisson.rvs(max(0, dR * dt))
            deaths = poisson.rvs(max(0, dD * dt))
            
            new_pop_state.susceptible = max(0, new_pop_state.susceptible - infections)
            new_pop_state.exposed = max(0, new_pop_state.exposed + infections - exposures)
            new_pop_state.infected = max(0, new_pop_state.infected + exposures - recoveries - deaths)
            new_pop_state.recovered = new_pop_state.recovered + recoveries
            new_pop_state.deaths = new_pop_state.deaths + deaths
        else:
            # Deterministic updates
            new_pop_state.susceptible = max(0, new_pop_state.susceptible + dS * dt)
            new_pop_state.exposed = max(0, new_pop_state.exposed + dE * dt)
            new_pop_state.infected = max(0, new_pop_state.infected + dI * dt)
            new_pop_state.confirmed = max(0, new_pop_state.confirmed + dQ * dt)
            new_pop_state.recovered = new_pop_state.recovered + dR * dt
            new_pop_state.deaths = new_pop_state.deaths + dD * dt
        
        # make population counts integers
        new_pop_state.susceptible = int(round(new_pop_state.susceptible))
        new_pop_state.exposed = int(round(new_pop_state.exposed))
        new_pop_state.infected = int(round(new_pop_state.infected))
        new_pop_state.confirmed = int(round(new_pop_state.confirmed))
        new_pop_state.recovered = int(round(new_pop_state.recovered))
        new_pop_state.deaths = int(round(new_pop_state.deaths))
        
        return new_pop_state

    
    def simulate(self, 
                 state: WorkflowState,
                 simulation_steps: int,
                 dt: float = 1.0,
                 stochastic: bool = False,
                 regulation_freq: str = 'daily',
                 policy_type: str = 'restriction') -> WorkflowState:
        """
        Run full simulation.
        
        Args:
            state: Initial workflow state
            simulation_steps: Number of simulation steps
            dt: Time step for each simulation step
            stochastic: Whether to use stochastic updates
            
        Returns:
            workflow state
        """                

        for step in range(simulation_steps):
            
            global_step = (state['current_iteration_step']) * simulation_steps + step

            if self.logger:
                self.logger.info(f"SEIR Model Simulation Step: {step}, using data at global step {global_step}")
            
            adjusted_trans_states = {}
            detection_states = {}
            if self.decision_type == 'agent':
                if policy_type == 'ori_restriction' or policy_type == 'des_restriction':
                    od_baseline = {}
                    for region_id, region_state in state['region_infos'].items():
                        curr_trans_state = deepcopy(region_state.gt_mobilities[global_step])
                        od_baseline.update({(nb_id, region_id): flow for nb_id, flow in curr_trans_state.inflow.items()})
                    OD_multiplier, OD_new_flow, banned_records = self.compute_od_multipliers_with_redistribution(od_baseline, state['region_infos'], default_cut=0.5, policy_type=policy_type)
                    if self.logger:
                        self.logger.info(f"Computed OD multipliers with redistribution: {OD_multiplier}")
                        self.logger.info(f"Banned records: {banned_records}")

            for region_id, region_state in state['region_infos'].items():

                # Check bounds to prevent index errors
                if global_step >= len(region_state.gt_mobilities):
                    if self.logger:
                        self.logger.warning(f"Global step {global_step} exceeds available mobility data length {len(region_state.gt_mobilities)}. Skipping.")
                    continue

                if self.logger:
                    self.logger.info(f"=== Simulating region: {region_id} ===")
                    
                mob_future = region_state.gt_mobilities[state['current_iteration_step']*simulation_steps:(state['current_iteration_step']+1)*simulation_steps]
                if len(mob_future) < (simulation_steps):
                    delta = simulation_steps/len(mob_future)
                else:
                    delta = 1
                # Get current transportation state
                curr_trans_state = deepcopy(region_state.gt_mobilities[global_step])

                if self.logger:
                    self.logger.info(f"Current transportation state: Inflow & Outflow {curr_trans_state}")
                
                if region_state.policy_response:
                    if policy_type == 'reallocation':
                        for nb_id, reduction in region_state.policy_response.policy.items():
                            if sum(reduction) ==0:
                                continue
                            inflow_bs = [m.inflow.get(nb_id, 0) for m in mob_future]
                            total = sum(inflow_bs)
                            if regulation_freq == 'weekly':
                                week_index = step // 7
                                week_expected_flow = reduction[week_index] * total * delta
                                mob_next_week = region_state.gt_mobilities[state['current_iteration_step']*simulation_steps + week_index*7:state['current_iteration_step']*simulation_steps+ (week_index+1)*7]
                                week_actual_flow = sum([m.inflow.get(nb_id, 0) for m in mob_next_week])
                                updated_inflow = curr_trans_state.inflow.get(nb_id, 0) * (week_expected_flow / week_actual_flow) if week_actual_flow > 0 else 0
                            elif regulation_freq == 'daily':
                                day_expected_flow = reduction[step] * total * delta
                                mob_next_day = region_state.gt_mobilities[state['current_iteration_step']*simulation_steps + step]
                                day_actual_flow = mob_next_day.inflow.get(nb_id, 0)
                                updated_inflow = curr_trans_state.inflow.get(nb_id, 0) * (day_expected_flow / day_actual_flow) if day_actual_flow > 0 else 0
                            curr_trans_state.inflow[nb_id] = int(updated_inflow)
                    elif  policy_type == 'ori_restriction' or policy_type == 'des_restriction':
                        for neighbor_id, _ in state['region_infos'].items():
                            if neighbor_id == region_id:
                                continue
                            else:
                                curr_trans_state.inflow[neighbor_id] = int(OD_new_flow.get((neighbor_id, region_id), 0))
                    elif policy_type == 'detection':
                        detection_states[region_id] = region_state.policy_response.policy.keys()
                adjusted_trans_states[region_id] = curr_trans_state
                    
            ###update outflow based on inflow of other regions
            for region_id, region_state in state['region_infos'].items():
                if global_step >= len(region_state.gt_mobilities):
                    continue
                if self.logger:
                    self.logger.info(f"=== Simulating region, apply policy: {region_id} ===")
                curr_trans_state = adjusted_trans_states[region_id]
                for nb_id, nb_state in state['region_infos'].items():
                    if nb_id == region_id:
                        continue
                    nb_trans_state = adjusted_trans_states[nb_id]
                    outflow_to_nb = nb_trans_state.inflow.get(region_id, 0)
                    curr_trans_state.outflow[nb_id] = outflow_to_nb
                if self.logger:
                    if region_state.policy_response:
                        self.logger.info(f"Applied policy reductions: {region_state.policy_response.policy}")
                        self.logger.info(f"Adjusted transportation state after policy: Inflow & outflow {curr_trans_state}")
            
                new_pop_state = self.step(state, 
                                        region_id, 
                                        curr_trans_state, 
                                        dt, 
                                        global_step,
                                        stochastic=stochastic,
                                        detection_states=detection_states)

                if self.logger:
                    self.logger.info(f"Updated population state for region {region_id}: {new_pop_state}")
                
                region_state.current_population = new_pop_state
                region_state.population_history.append(new_pop_state)
                region_state.mobility_history.append(curr_trans_state)
                region_state.current_mobility = curr_trans_state
                
        return state    
    


    def compute_od_multipliers_with_redistribution(self, baseline_od, region_infos, default_cut=0.5, fallback_if_no_receiver="drop", policy_type = 'ori_restriction'):
        regions = list(region_infos.keys())
        OD_new_flow = {(o, d): float(baseline_od.get((o, d), 0.0))
                    for o in regions for d in regions if o != d}
        removed_by_origin = {o: 0.0 for o in regions}
        banned_dests_by_origin = {o: set() for o in regions}
        # 1) 应用限制，并记录 removed flow 以及 banned dest
        if policy_type == 'ori_restriction':
            for dest_id, dest_state in region_infos.items():
                origin_id, restricted_ratio =  dest_state.policy_response.policy.keys(), dest_state.policy_response.policy.values()
                origin_id = next(iter(origin_id), None)
                restricted_ratio = next(iter(restricted_ratio), None)
                if origin_id is None:
                    continue
                elif origin_id == dest_id:
                    continue
                key = (origin_id, dest_id)
                if key not in OD_new_flow:
                    continue
                # 记录：origin_id 的被限制 destination
                banned_dests_by_origin.setdefault(origin_id, set()).add(dest_id)
                old = OD_new_flow[key]
                new = int(old * float(default_cut))
                OD_new_flow[key] = new
                removed_by_origin[origin_id] += (old - new)
        elif policy_type == 'des_restriction':
            for origin_id, origin_state in region_infos.items():
                dest_id, restricted_ratio =  origin_state.policy_response.policy.keys(), origin_state.policy_response.policy.values()
                dest_id = next(iter(dest_id), None)
                restricted_ratio = next(iter(restricted_ratio), None)
                if dest_id is None:
                    continue
                elif dest_id == origin_id:
                    continue
                key = (origin_id, dest_id)
                if key not in OD_new_flow:
                    continue
                banned_dests_by_origin.setdefault(origin_id, set()).add(dest_id)
                old = OD_new_flow[key]
                new = int(old * float(default_cut))
                OD_new_flow[key] = new
                removed_by_origin[origin_id] += (old - new)
        # 2) 对每个 origin，把 removed flow 重分配到其它 destination（排除该 origin 的 banned dest， 此处代码设计是否考虑出行量的转移）
        for o, removed in removed_by_origin.items():
            if removed <= 0:
                continue
            banned = banned_dests_by_origin.get(o, set())
            # receivers：排除 self + 排除该 origin 对应 banned destinations
            receivers = [d for d in regions if d not in banned and d != o]
            if not receivers:
                if fallback_if_no_receiver == "drop":
                    continue
            weights = [float(baseline_od.get((o, d), 0.0)) for d in receivers]
            total_w = sum(weights)
            if total_w <= 0:
                if fallback_if_no_receiver == "equal":
                    share = removed / len(receivers)
                    for d in receivers:
                        OD_new_flow[(o, d)] += share
                else:
                    # 默认 drop
                    continue
            else:
                for d, w in zip(receivers, weights):
                    if w <= 0:
                        continue
                    if d == o:
                        continue
                    OD_new_flow[(o, d)] += int(removed * (w / total_w))
        # 3) 输出所有OD变化比例
        OD_multiplier = {}
        for (o, d), new_f in OD_new_flow.items():
            base_f = float(baseline_od.get((o, d), 0.0))
            OD_multiplier[(o, d)] = (new_f / base_f) if base_f > 0 else (0.0 if new_f == 0 else float("inf"))
        return OD_multiplier, OD_new_flow, banned_dests_by_origin