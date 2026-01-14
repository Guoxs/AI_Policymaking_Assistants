
import re
import json
import ast
import os
from typing import TypedDict, List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
#from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import WorkflowState, RegionState, PolicyResponse, PolicyResponse_v2

from agents.utils import setup_logger,extract_policy_responses
from agents.epidemic_model import SEIRModel


class RegionAgent:
    """Agent for managing region-specific operations."""
    def __init__(self, 
                 region_id: str, 
                 llm_config: Dict[str, Any],
                 simulation_config: Dict[str, Any], 
                 log_path,
                 decision_type: str):
        
        self.region_id = region_id
        self.logger = setup_logger(self.region_id, log_path)
        
        self.llm_config = llm_config
        self.simulation_config = simulation_config

        if self.simulation_config['regulation_freq'] == 'weekly':
            if self.simulation_config['simulation_steps'] == 14:
                if self.simulation_config['policy_type'] == 'ori_restriction':
                    from agents.restriction_policy.prompts_v2_ori import REGION_AGENT_SYSTEM_PROMPT
                elif self.simulation_config['policy_type'] == 'des_restriction':
                    from agents.restriction_policy.prompts_v2_des import REGION_AGENT_SYSTEM_PROMPT
                elif self.simulation_config['policy_type'] == 'reallocation':
                    from agents.reallocation_policy.prompts_v2 import REGION_AGENT_SYSTEM_PROMPT
                elif self.simulation_config['policy_type'] == 'detection':
                    from agents.detection_policy.prompt import REGION_AGENT_SYSTEM_PROMPT
            elif self.simulation_config['simulation_steps'] == 21:
                from agents.reallocation_policy.prompts_v3 import REGION_AGENT_SYSTEM_PROMPT
            elif self.simulation_config['simulation_steps'] == 28:
                from agents.reallocation_policy.prompts_v4 import REGION_AGENT_SYSTEM_PROMPT
            elif self.simulation_config['simulation_steps'] == 42:
                from agents.reallocation_policy.prompts_v6_1 import REGION_AGENT_SYSTEM_PROMPT
            elif self.simulation_config['simulation_steps'] == 56:
                from agents.reallocation_policy.prompts_v8 import REGION_AGENT_SYSTEM_PROMPT
            elif self.simulation_config['simulation_steps'] == 70:
                from agents.reallocation_policy.prompts_v10 import REGION_AGENT_SYSTEM_PROMPT
        elif self.simulation_config['regulation_freq'] == 'daily':
            if self.simulation_config['simulation_steps'] == 7:
                from agents.reallocation_policy.prompts_d7 import REGION_AGENT_SYSTEM_PROMPT
            elif self.simulation_config['simulation_steps'] == 14:
                from agents.reallocation_policy.prompts_d14 import REGION_AGENT_SYSTEM_PROMPT
            elif self.simulation_config['simulation_steps'] == 21:
                from agents.reallocation_policy.prompts_d21 import REGION_AGENT_SYSTEM_PROMPT
            elif self.simulation_config['simulation_steps'] == 28:
                from agents.reallocation_policy.prompts_d28 import REGION_AGENT_SYSTEM_PROMPT

                
        self.sys_prompt = REGION_AGENT_SYSTEM_PROMPT.format(
            region_id=self.region_id,
            N=self.simulation_config['simulation_steps'],
            M=self.simulation_config['simulation_steps']
        )

        self.llm = init_chat_model(
            model=self.llm_config['model'], 
            base_url=self.llm_config['base_url'],
            api_key=self.llm_config['api_key'],
            temperature=self.llm_config['temperature'],
            model_provider='openai' 
        )
        
        if decision_type == 'load_policy':
            self.load_policy()
        # self.load_policy()
        
    def run(self, state: WorkflowState) -> PolicyResponse_v2:
        """Execute the region agent's main workflow."""
        max_iterations = state['max_iterations']
        iteration_number = state['current_iteration_step'] + 1
        try:
            if self.logger:
                self.logger.info(f"Region Agent {self.region_id} - Iteration {iteration_number} / {max_iterations}: Starting policy decision process.")
            
            # make policy for transportation control based on state
            user_prompt = self._construct_policy_prompt(state)
            if self.logger:
                self.logger.info(f"Region user prompt: {user_prompt}")
        
            # Run the agent
            messages = [
                SystemMessage(content=self.sys_prompt),
                HumanMessage(content=user_prompt)
            ]
            result = self.llm.invoke(messages)
            raw_output = result.content.strip() if hasattr(result, 'content') else str(result)
            
            if self.logger:
                self.logger.info(f"Raw agent output:\n{raw_output}")
            
            #policy_response = self._parse_response(raw_output)
            if self.simulation_config['policy_type'] == 'reallocation':
                policy_response = self._parse_adjust_policy_response(raw_output)
            elif self.simulation_config['policy_type'] == 'ori_restriction' or self.simulation_config['policy_type'] == 'des_restriction' or self.simulation_config['policy_type'] == 'detection':
                policy_response = self._parse_restriction_policy_response(raw_output)
            if self.logger:
                self.logger.info(f"Parsed PolicyResponse: {policy_response}")
            
            return policy_response
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in Region Agent {self.region_id}: {str(e)}")
            
            # Return empty policy on error
            policy_response = PolicyResponse(
                policy={},
                explanation=f"Error occurred: {str(e)}"
            )
                
            return policy_response
    
    
    def _construct_policy_prompt(self, state: WorkflowState) -> str:
        """Construct the prompt for policy decision-making."""
                
        region_state = state['region_infos'].get(self.region_id)
       
        if region_state is None:
            raise ValueError(f"Region {self.region_id} not found in WorkflowState.")
    
        pop_state = region_state.current_population
        seir_params = region_state.epidemic_params

        # SEIRD parameters
        user_prompt = f"For {self.region_id}, you will be given the following information:\n\n"
        user_prompt += "Epidemic and mobility information of this state are as follows:\n\n"
        user_prompt += "SEIRD parameters:\n"
        user_prompt += f"- beta = {seir_params.beta}\n"
        user_prompt += f"- alpha = {seir_params.alpha}\n"
        user_prompt += f"- gamma = {seir_params.gamma}\n"
        user_prompt += f"- mu = {seir_params.mu}\n"
        user_prompt += f"- delta = {seir_params.delta}\n\n"
        
        # Current region population
        user_prompt += "Current population composition:\n"
        for k, v in pop_state.to_dict().items():
            user_prompt += f"- {k} = {v}\n"
        
         # Other regions
        #history_days = self.simulation_config['simulation_steps']
        history_days = 21

        user_prompt += "\nOther origin states:\n"
        for idx, nb_id in enumerate(region_state.neighboring_region_ids, 1):
            if nb_id not in list(state["region_infos"].keys()):
                continue
            
            nb_state = state["region_infos"][nb_id]

            # For population history
            if nb_state.population_history and len(nb_state.population_history) >= history_days:
                hist = nb_state.population_history[-history_days:]
            else:
                hist = nb_state.population_history or [nb_state.current_population]

            avg_comp = {
                k: int(sum(p.to_dict()[k] for p in hist) / len(hist))
                for k in ["S", "E", "I", "Q", "R", "D"]
            }
            comp_str = ", ".join([f"{k}={v}" for k, v in avg_comp.items()])

            # For mobility history
            if nb_state.mobility_history and len(nb_state.gt_mobilities) >= history_days:
                mob_hist = nb_state.mobility_history[-history_days:]
            else:
                mob_hist = nb_state.mobility_history or [nb_state.current_mobility]
            
            avg_inflow = {}
            for m in mob_hist:
                for k, v in m.inflow.items():
                    avg_inflow[k] = avg_inflow.get(k, 0) + v   
            avg_inflow = {k: int(v / len(mob_hist)) for k, v in avg_inflow.items()}
            
            inflow_str = ", ".join([f"{k}={v}" for k, v in avg_inflow.items()])
            
            user_prompt += (
                f"{idx}. {nb_id}\n"
                f"   - Avg past {history_days}-day population composition: {comp_str}\n"
                f"   - Avg past {history_days}-day inflow: {inflow_str}\n\n"
            )
        ###demonstrate future mobility
        future_days = self.simulation_config['simulation_steps']
        mob_future = region_state.gt_mobilities[state['current_iteration_step']*future_days:(state['current_iteration_step']+1)*future_days]
        future_lines = []
        if self.simulation_config['policy_type'] == 'ori_restriction' or self.simulation_config['policy_type'] == 'rellocation':
            # 计算未来21天各州流入目标州的日均流量
            inflow_fraction_dict = {}
            for from_region in region_state.neighboring_region_ids:
                inflow_list = [m.inflow.get(from_region, 0) for m in mob_future]
                total = sum(inflow_list)/len(inflow_list)
                # if total > 0:
                #     fraction_list = [round(v / total, 4) for v in inflow_list]
                # else:
                #     fraction_list = [0 for _ in inflow_list]
                inflow_fraction_dict[from_region] = {
                "daily_inflow": float(round(total, 2)),
                # "daily_fraction": fraction_list
                }
            for from_region in sorted(inflow_fraction_dict.keys()):
                future_lines.append(f"From {from_region}: Inflow ={inflow_fraction_dict[from_region]['daily_inflow']};")
            user_prompt += "\n"
            user_prompt += f" Ground truth daily average for next {self.simulation_config['simulation_steps']}-day inbound by origin states; \n"
            user_prompt += "\n".join(future_lines) + "\n\n"
        elif self.simulation_config['policy_type'] == 'des_restriction':
            outflow_fraction_dict = {}
            for des_region in region_state.neighboring_region_ids:
                outflow_list = [m.outflow.get(des_region, 0) for m in mob_future]
                total = sum(outflow_list)/len(outflow_list)
                outflow_fraction_dict[des_region] = {
                "daily_outflow": float(round(total, 2)),
                }
            for des_region in sorted(outflow_fraction_dict.keys()):
                future_lines.append(f"To {des_region}: Outflow ={outflow_fraction_dict[des_region]['daily_outflow']};")
            user_prompt += "\n"
            user_prompt += f" Ground truth daily average for next {self.simulation_config['simulation_steps']}-day outbound to destination states; \n"
            user_prompt += "\n".join(future_lines) + "\n\n"
        # day_labels = []
        # for d, m in enumerate(mob_future):
        #     date_label = getattr(m, "date", None)
        #     day_labels.append(str(date_label) if date_label else f"Day{d+1}")

        # 2) 为每个来源州构建一行：total + 每日占比（确保sum≈1；total=0时全0）
        # future_lines = []
        # for from_region in sorted(inflow_fraction_dict.keys()):
        #     info = inflow_fraction_dict[from_region]
        #     total = int(info["total"])
        #     fracs = info["daily_fraction"]
        #     fracs = [0.0 if abs(x) < 1e-12 else float(f"{x:.4f}") for x in fracs]
        #     if total > 0:
        #         s = sum(fracs)
        #         if s > 0 and abs(s - 1.0) > 1e-6:
        #             fracs = [float(f"{x/s:.4f}") for x in fracs]
        #     else:
        #         fracs = [0.0 for _ in fracs]
        #     daily_pairs = ", ".join([f"{dlabel}:{frac:.4f}" for dlabel, frac in zip(day_labels, fracs)])
        #     future_lines.append(f"From {from_region}: total={total}; daily_fraction=[{daily_pairs}]")
        # user_prompt += (
        #     f"   - Ground truth next {self.simulation_config['simulation_steps']}--day inbound by origin (total & daily fractions); "
        #     "fractions sum to 1 per origin when total>0):\n" +
        #     "\n".join(future_lines) + "\n\n")        

        user_prompt += f"Please output the recommended traffic control policy in the required JSON format."
        
        return user_prompt

        
    def _parse_response(self, response: str) -> PolicyResponse:
        """Parse the response from the agent into a PolicyResponse object"""
        try:
            # Try to find JSON block first
            json_matches = re.findall(r"```json\s*([\s\S]*?)\s*```", response, re.DOTALL | re.IGNORECASE)
            if json_matches:
                json_block = json_matches[0]
            else:
                # 2. Otherwise assume whole response is JSON
                json_block = response.strip()
                
                # Clean up invalid line breaks inside strings
                json_block = re.sub(r"\n\s+", " ", json_block)
                
                # replace %
                json_block = json_block.replace('%', 'percent')
            
                try:
                    data = json.loads(json_block)
                except json.JSONDecodeError:
                    data = ast.literal_eval(json_block)
                return PolicyResponse(**data)
            
        except Exception as e:
            # fallback: empty response
            return PolicyResponse(
                policy={},
                explanation=f"Parse error: {e}. Raw response: {response[:200]}..."
            )
        
    

    def _normalize(self, fracs: List[float]) -> List[float]:
        policy_frequency = self.simulation_config['simulation_steps']
        if self.simulation_config['regulation_freq'] == 'weekly':
            num = int(policy_frequency/7)
        elif self.simulation_config['regulation_freq'] == 'daily':
            num = policy_frequency
        fracs = [float(x) for x in fracs[:num]] + [0.0] * max(0, num - len(fracs))
        s = sum(fracs)
        if s <= 0:
            return [0.0] * num
        fracs = [x / s for x in fracs]
        fracs[-1] += 1.0 - sum(fracs)  # 修正尾差，确保和=1
        return fracs

    def _pick_json_block(self, text: str) -> str:
        m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.I)
        return (m.group(1) if m else text).strip().replace("%", "percent")

    def _parse_adjust_policy_response(self, response: str) -> PolicyResponse_v2:
        txt = self._pick_json_block(response)

        # 1) 标准反序列化
        data = None
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(txt)
                break
            except Exception:
                pass

        policy: Dict[str, List[float]] = {}
        explanation = ""

        if isinstance(data, dict):
            explanation = str(data.get("global_guidance", ""))
            for k, v in data.items():
                if k == "global_guidance":
                    continue
                if isinstance(v, dict) and "fractions" in v:
                    if 'chosen_strategy' in v and v['chosen_strategy'] == "S2_no_control":
                        policy[k] = [0, 0, 0, 0]
                    else:
                        policy[k] = self._normalize(v["fractions"])
                elif isinstance(v, list):
                    if 'chosen_strategy' in v and v['chosen_strategy'] == "S2_no_control":
                        policy[k] = [0, 0, 0, 0]
                    policy[k] = self._normalize(v)
        else:
            # 2) 极简正则兜底（处理坏 JSON）
            for m in re.finditer(r'"([^"]+)"\s*:\s*\{[^{}]*?"fractions"\s*:\s*\[([^\]]+)\]', txt):
                state = m.group(1)
                nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', m.group(2))]
                policy[state] = self._normalize(nums)
            mg = re.search(r'"global_guidance"\s*:\s*"([^"]*)"', txt)
            if mg:
                explanation = mg.group(1)

        return PolicyResponse_v2(policy=policy, explanation=explanation)
    
    def _parse_restriction_policy_response(self, response: str) -> PolicyResponse_v2:
        txt = self._pick_json_block(response)
        # 1) 标准反序列化
        data = None
        for loader in (json.loads, ast.literal_eval):
            try:
                if not txt.endswith('}'):
                    txt = txt + '}'
                data = json.loads(txt)
                break
            except Exception:
                pass
        policy: Dict[str, List[float]] = {}
        explanation = ""
        if isinstance(data, dict):
            explanation = str(data.get("global_guidance", ""))
            for k, v in data.items():
                if k == "global_guidance":
                    continue
                if isinstance(v, dict) and "restricted_state" in v:
                    origin = v["restricted_state"]
                    if origin is None or origin.lower() == "null":
                        policy[origin] = [0.0, 0.0]
                    else:
                        policy[origin] = [0.5, 0.5]  # 固定50%削减
        return PolicyResponse_v2(policy=policy, explanation=explanation)

    def rule_based_policy(self, state, simulation_config) -> Dict[str, List[float]]:
        """
        rule-based 策略：
        - 只考虑本州 {region_id} 的疫情参数和当前状态
        - 对所有 origin 州采用相同的 weekly pattern
        - 不考虑跨州协同，体现“各自为政”
        返回：
            { from_region_id: [f1, f2, f3, f4], ... }
        """
        decision_freq = simulation_config['regulation_freq']
        simulation_steps = simulation_config['simulation_steps']
        if decision_freq == 'weekly':
            decision_num = int(simulation_steps / 7)
        elif decision_freq == 'daily':
            decision_num = simulation_steps
        region_state = state["region_infos"].get(self.region_id)
        if region_state is None:
            raise ValueError(f"Region {self.region_id} not found in WorkflowState.")

        # --- 1. 读取当前人口构成 ---
        pop = region_state.current_population.to_dict()
        S = float(pop.get("S", 0.0))
        E = float(pop.get("E", 0.0))
        I = float(pop.get("I", 0.0))
        Q = float(pop.get("Q", 0.0))
        R = float(pop.get("R", 0.0))
        D = float(pop.get("D", 0.0))

        N = max(S + E + I + Q + R + D, 1.0)  # 防止除 0

        # 活跃病例比例（只看 I + Q）
        active_ratio = (I + Q) / N

        # --- 2. 读取 SEIRD 参数，构造一个粗糙 R_eff proxy ---
        params = region_state.epidemic_params
        beta = float(getattr(params, "beta", 0.0))
        gamma = float(getattr(params, "gamma", 0.0))
        mu    = float(getattr(params, "mu", 0.0))
        delta = float(getattr(params, "delta", 0.0))

        denom = gamma + mu + delta
        if denom > 0:
            Reff = (beta * S / N) / denom
        else:
            Reff = 0.0

        # --- 3. 根据 active_ratio 和 Reff 选择一种 pattern ---
        # 严重：先严后松（strict-first）
        if active_ratio > 0.01 or Reff > 1.2:
            if decision_num ==4:
            #fractions = [0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15]
                fractions = [0.2,0.2,0.3,0.3]
            elif decision_num ==6:
                fractions = [0.10, 0.10, 0.15, 0.15, 0.25, 0.25]
            elif decision_num ==7:
                fractions = [0.07, 0.10, 0.14, 0.18, 0.20, 0.15, 0.16]
        # 中等：近似均匀
        elif active_ratio > 0.003 or Reff >= 1.0:
            if decision_num ==4:
            #fractions = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
                fractions = [0.25, 0.25, 0.25, 0.25]
            elif decision_num ==6:
                fractions = [1/6] * 6
            elif decision_num ==7:
                fractions = [1/7] * 7
        # 轻微：偏宽松（relaxed-first）
        else:
            fractions = [0]
        # --- 4. 对所有 origin 州采用相同 fractions（体现各自为政） ---
        policy_apply: Dict[str, List[float]] = {}
        for from_region in region_state.neighboring_region_ids:
            if from_region not in state["region_infos"]:
                continue
            policy_apply[from_region] = fractions
        return PolicyResponse_v2(policy=policy_apply, explanation="Rule-based policy applied.")
    

    def load_policy(self):
        results_root = "outputs\\5 states\\6 weeks\\gpt-3.5"
        #use_folder = sorted([f for f in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, f))])[0]
        use_folder = 'gpt-3.5-turbo-0125_2025-12-16-13-39-47'
        print("Last folder:", use_folder)
        agent_folder = os.path.join(results_root, use_folder)
        log_file = self.region_id +'.log'
        log_path = os.path.join(agent_folder, log_file)
        self.responses = extract_policy_responses(log_path)
        self.count = 0
    
    def get_loaded_policy(self) -> PolicyResponse_v2:
        if self.count < len(self.responses):
            response = self.responses[self.count]
            self.count += 1
            policy_response = PolicyResponse_v2(policy=response['policy'], explanation=response['explanation'])
            if self.logger:
                self.logger.info(f"Parsed PolicyResponse: {policy_response}")
            return policy_response
        else:
            policy_response = None
            return policy_response 
            #raise IndexError("No more loaded policy responses available.")
        
    def random_policy(self, state) -> PolicyResponse_v2:
        region_ids = self.region_id
        decision_freq = self.simulation_config['regulation_freq']
        simulation_steps = self.simulation_config['simulation_steps']
        if decision_freq == 'weekly':
            decision_num = int(simulation_steps / 7)
        elif decision_freq == 'daily':
            decision_num = simulation_steps
        policy: Dict[str, List[float]] = {}
        import random
        region_state = state["region_infos"].get(self.region_id)
        for from_region in region_state.neighboring_region_ids:
            if from_region == region_ids:
                continue
            fracs = [random.random() for _ in range(decision_num)]
            s = sum(fracs)
            fracs = [x / s for x in fracs]
            policy[from_region] = fracs
        return PolicyResponse_v2(policy=policy, explanation="Random policy applied.")