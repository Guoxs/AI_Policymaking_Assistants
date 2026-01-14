import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.integrate import solve_ivp   
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import odeint
from scipy.optimize import minimize

from agents.state import PopulationState, EpidemicParameters, EpidemicParameters_v2

class EpidemicModelFit:
    
    def __init__(self, param_bounds: List[tuple], initial_guess: List[float]):        
        self.param_bounds = param_bounds
        self.initial_guess = initial_guess  #beta, delta, lambda0, kapp
        
    
    def seird_ode(self, t, y,  beta, delta, lambda0, kappa, Npop):
        S, E, I, Q, R, D = y
        # Differential equations:
        dS = -(beta * S * I + beta * 0.5 * S * Q)/ Npop
        dE = (beta * S * I + beta * 0.5 * S * Q)/ Npop - self.alpha * E
        dI = self.alpha * E - delta * I
        dQ = delta * I - (lambda0 +kappa) * Q
        dR = lambda0 * Q
        dD = kappa * Q

        return [dS, dE, dI, dQ, dR, dD]

    def euler_seird_solver(self, beta,  delta, lambda0, kappa, Npop, y0, t_full):
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
            dydt = self.seird_ode(t_prev, y_prev, beta, delta,  lambda0, kappa, Npop)
            # 欧拉公式: y_new = y_old + dt * dy/dt
            sol_y[:, i] = y_prev + dt * np.array(dydt)
        return sol_y
    
    def rk45_solve_seird(self, beta, kappa, Npop, y0, t_eval):
        def f(t, y):
            y = np.maximum(y, 0.0)
            S,E,I,Q,R,D = y
            dS = -(beta*S*I + beta*0.5*S*Q)/Npop
            dE = (beta*S*I + beta*0.5*S*Q)/Npop - self.alpha*E
            dI = self.alpha*E - self.delta*I
            dQ = self.delta*I - (self.lambda0+kappa)*Q
            dR = self.lambda0*Q
            dD = kappa*Q
            return [dS,dE,dI,dQ,dR,dD]
        sol = solve_ivp(f, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-8)
        return sol.y  # shape: (6, T)

    def fit_v2(self, pop_data: List[PopulationState], old_params: EpidemicParameters, **kwargs) -> EpidemicParameters:
        """Fit the SEIRD model to the data."""
        
        fit_data = self.state_to_frame(pop_data)
        tolX = kwargs.get('tolX', 1e-5)
        tolFun = kwargs.get('tolFun', 1e-5)
        dt = 1       
        S_data = fit_data['S']
        E_data = fit_data['E']
        I_data = fit_data['I']
        Q_data = fit_data['Q']
        R_data = fit_data['R']
        D_data = fit_data['D']

        t_full = np.linspace(0, len(fit_data)-1, len(fit_data))
        Npop = fit_data.iloc[0][['S', 'E', 'I', 'Q', 'R', 'D']].sum()

        self.alpha = old_params.alpha
        beta = old_params.beta
        self.delta = old_params.delta
        self.lambda0 = old_params.gamma
        kappa = old_params.mu
        initial_params = np.array([beta, kappa])
        ub = [5.0, 0.05]
        lb = [0.01, 0.0]
        wQ = 1.0 / (np.max(Q_data) + 1e-6)
        wR = 1.0 / (np.max(R_data) + 1e-6) 
        wD = 1.0 / (np.max(D_data) + 1e-6)
        weights = [wQ, wR, wD]
        target_data = np.concatenate((S_data, E_data, I_data, Q_data, R_data, D_data))
        y0 = [S_data[0], E_data[0], I_data[0], Q_data[0], R_data[0], D_data[0]]
        def residuals(p):
            beta, kap = p[:2]
            sol = self.rk45_solve_seird(beta, kap, Npop, y0, t_full)
            Qm = sol[3, :]
            Rm = sol[4, :]
            Dm = sol[5, :]
            return self._pack_residuals_log(Qm, Rm, Dm, Q_data, R_data, D_data, weights, eps=1e-6)

        result = least_squares(
            # lambda p, t, data: model_for_fitting(t, *p) - data,
            residuals,
            x0=initial_params,
            # args=(t_full, target_data),
            bounds=(lb, ub),
            ftol=tolFun,
            xtol=tolX,
            method =   'trf',
            verbose=0
        )

        coeffs = result.x
        beta1 = coeffs[0]
        # delta1 = coeffs[1]
        # Lambda1 = coeffs[2]
        Kappa1 = coeffs[1]
        return EpidemicParameters_v2(alpha = self.alpha, beta=beta1, delta = self.delta, gamma=self.lambda0, mu=Kappa1)


    def mse_loss(self, fit_data: pd.DataFrame, params: List[float]) -> float:
        
        beta, gamma, mu, alpha = params
        
        y0 = [fit_data['S'].iloc[0], 
              fit_data['E'].iloc[0], 
              fit_data['I'].iloc[0], 
              fit_data['R'].iloc[0], 
              fit_data['D'].iloc[0]
            ]
        
        t = np.linspace(0, len(fit_data)-1, len(fit_data))
        
        solution = odeint(self.SEIRD_model, y0, t, args=(beta, alpha, gamma, mu))
        
        pred_S, pred_E, pred_I, pred_R, pred_D = solution.T
        

        mse = np.mean((pred_E - fit_data['E']) ** 2 + 
                      (pred_I - fit_data['I']) ** 2 + 
                      (pred_R - fit_data['R']) ** 2 +
                      (pred_D - fit_data['D']) ** 2
                    )
        return mse
    
    def fit(self, pop_data: PopulationState) -> EpidemicParameters:
        """Fit the SEIRD model to the data."""
        
        fit_data = self.state_to_frame(pop_data)
        
        result = minimize(
            lambda params: self.mse_loss(fit_data, params),
            self.initial_guess,
            bounds=self.param_bounds,
            method='L-BFGS-B'
        )
        

        # Ensure fitted parameters are within bounds
        fitted_params = np.clip(result.x, [b[0] for b in self.param_bounds], [b[1] for b in self.param_bounds])
        beta_fit, gamma_fit, mu_fit, alpha_fit = fitted_params
        # beta_fit, gamma_fit, mu_fit, alpha_fit = result.x
        
        return EpidemicParameters(beta=beta_fit, alpha=alpha_fit, gamma=gamma_fit, mu=mu_fit)
    
    
    def state_to_frame(self, state: PopulationState) -> pd.DataFrame:
        """Convert a PopulationState list to a DataFrame."""
        
        data = {
            'S': [s.susceptible for s in state],
            'E': [s.exposed for s in state],
            'I': [s.infected for s in state],
            'Q': [s.confirmed for s in state],
            'R': [s.recovered for s in state],
            'D': [s.deaths for s in state],
        }
        return pd.DataFrame(data)

    @staticmethod
    def _pack_residuals_log(Qm, Rm, Dm, Qd, Rd, Dd, w=[1.0, 1.0, 1.2], eps=1e-6):
        """
        构造加权对数残差：
        - Q 用存量 level 对齐
        - R/D 用一阶差分（新增量）对齐
        """
        # 模型与数据均裁剪到正数，避免log(0)
        Qm = np.maximum(Qm, 0.0)
        Rm = np.maximum(Rm, 0.0)
        Dm = np.maximum(Dm, 0.0)
        Qd = np.maximum(Qd, 0.0)
        Rd = np.maximum(Rd, 0.0)
        Dd = np.maximum(Dd, 0.0)
        # Q: level 残差（对数）
        res_Q = np.log(Qm + eps) - np.log(Qd + eps)
        # R, D: flow 残差（对数），使用一阶差分
        dRm = np.diff(Rm, prepend=Rm[0])
        dRd = np.diff(Rd, prepend=Rd[0])
        dDm = np.diff(Dm, prepend=Dm[0])
        dDd = np.diff(Dd, prepend=Dd[0])

        res_dR = np.log(dRm + eps) - np.log(dRd + eps)
        res_dD = np.log(dDm + eps) - np.log(dDd + eps)

        return np.concatenate([w[0]*res_Q, w[1]*res_dR, w[2]*res_dD])