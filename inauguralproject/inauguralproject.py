from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:
    #function that is called when the class is activated
    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 
        par.theta = 0.
        
        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan
    # The function underneith calculates: self,LM,HM,LF,HF
    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF


        # b. home production
        H = np.nan

        power = (par.sigma - 1)/par.sigma

        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.fmin(HM, HF)
        else: 
            H = (  (1-par.alpha)  * (HM+0.00000000001) **(power) + par.alpha * (HF+0.0000000001)**(power)  )**(1/power)
            # we write 0000.1 on the chance that we get 0 hours. 
    

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work and norms
        epsilon_= 1+1/par.epsilon
        epsilon_ = 1+1/par.epsilon 
        theta_ = 1+1/par.theta 
        TM = LM+HM
        TF = LF+HF

        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_+LM**theta_/theta_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
    
    # a. Define objective function   
        obj = lambda x: -self.calc_utility(x[0], x[1], x[2], x[3])
    
    #b. Define Constraints and Bounds (to minimize) 
        constraints = ({'type': 'ineq', 'fun': lambda x: [24 - x[0] - x[1], 24 - x[2] - x[3]]}) # cannot work more than 24 hours
        bounds = ((0,24), (0,24), (0,24), (0,24))
        initial_guess = [6,6,6,6]

    #c. Define solver
        solution = optimize.minimize(obj, initial_guess, method="nelder-mead", bounds=bounds, constraints=constraints)

        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]
        
        return opt 

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol
        
        #relative_hours = np.zeros(par.wF_vec.size) # make a vector of zeros as the same size of the wF_vec
        #log_relative_hours = np.zeros(par.wF_vec.size) # make a vector of zeros as the same size of the wF_vec

        for i, wF in enumerate(par.wF_vec): #solve the model over different values of the wage_vector
            par.wF = wF
            
            opt = self.solve()

            sol.LM_vec[i]= opt.LM
            sol.HM_vec[i]= opt.HM
            sol.LF_vec[i]= opt.LF
            sol.HF_vec[i]= opt.HF

            # relative_hours[i] = sol.HF/sol.HM
            # log_relative_hours[i] = np.log(sol.HF/sol.HM)

       #  sol.relative_hours = relative_hours
        # sol.log_relative_hours = log_relative_hours 
        
        return sol

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        self.solve_wF_vec()

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None,theta=None):
        """ estimate alpha and sigma """
        ## Needs to estimate them such that they yield the estimated results

        par = self.par
        sol = self.sol

        par.alpha = alpha
        par.sigma = sigma
        par.theta = theta

        self.solve_wF_vec()

        self.run_regression()

        return (par.beta0_target-sol.beta0)**2 + (par.beta1_target-sol.beta1)**2
        