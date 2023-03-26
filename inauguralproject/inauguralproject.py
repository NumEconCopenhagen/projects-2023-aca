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
        
        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # question 5
        par.kappa = 0.
        par.dummy = 0.

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
    def calc_utility(self,LM,HM,LF,HF,sigma=None,alpha=None):
        """ calculate utility """

        par = self.par
        sol = self.sol

        sigma = par.sigma if sigma is None else sigma
        alpha = par.alpha if alpha is None else alpha

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production, NEW: adjusted for different values of sigma
        if np.isclose(sigma,1):
            H = HM**(1-alpha)*HF**alpha
        elif np.isclose(sigma,0.0):
            H = min(HM,HF)
        else:
            H = ((1-alpha)*HM**((sigma-1)/sigma)+alpha*HF**((sigma-1)/sigma))**(sigma/(sigma-1))
    

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work and norms
        epsilon_= 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF

        #NEW: disutility of norms
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_ + par.dummy*(LF**par.kappa))
        
        return utility - disutility

    def solve_discrete(self,sigma=None,alpha=None,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

         #Accounting for None values of Sigma and Alpha
        sigma = par.sigma if sigma is None else sigma
        alpha = par.alpha if alpha is None else alpha
        
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

    def solve_continuous(self,sigma=None,alpha=None,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

         #Accounting for None values of Sigma and Alpha
        sigma = par.sigma if sigma is None else sigma
        alpha = par.alpha if alpha is None else alpha
    
    # a. Define objective function
        def obj(x):
            return -self.calc_utility(x[0], x[1], x[2], x[3])
    
    #b. Define Constraints and Bounds (to minimize) 
        cons = []
        cons.append({'type': 'ineq', 'fun': lambda x: 24 - x[0] - x[1]})
        cons.append({'type': 'ineq', 'fun': lambda x: 24 - x[2] - x[3]})
        bnds = ((0,24), (0,24), (0,24), (0,24))
        initial_guess = (4.5,4.5,4.5,4.5) 

    #c. Define solver: DIFFERENCE BETWEEN SLSQP og Nelder Mead
        res = optimize.minimize(obj, x0=initial_guess, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-10)

        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        if do_print:
            print(res.message)

            print(f'LM: {opt.LM:.4f}')
            print(f'HM: {opt.HM:.4f}')
            print(f'LF: {opt.LF:.4f}')
            print(f'HF: {opt.HF:.4f}')
        
        return opt 

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        #Setting up parameters
        par = self.par
        sol = self.sol

         # Vectors for results
        par.lw_vec = np.zeros(len(par.wF_vec)) # log wages
        par.lH_vec = np.zeros(len(par.wF_vec)) # log hours
        sol.HM_vec = np.zeros(len(par.wF_vec)) # male home hours
        sol.HF_vec = np.zeros(len(par.wF_vec)) # female home hours
        
        # We Loop through values of wages
        for i, wF in enumerate(par.wF_vec): #solve the model over different values of the wage_vector
            par.wF = wF # call wage

            if discrete: # solving with discrete method
                opt = self.solve_discrete()
            
            else: #solving with continues method 

                opt = self.solve_continuous()

            #Save results 
            par.lw_vec[i] = np.log(par.wF/par.wM) # log wage 
            par.lH_vec[i] = np.log(opt.HF/opt.HM) # log hours 

            sol.LM_vec[i]= opt.LM
            sol.HM_vec[i]= opt.HM
            sol.LF_vec[i]= opt.LF
            sol.HF_vec[i]= opt.HF

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        self.solve_wF_vec()

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    def estimate_1(self,alpha_vals=None,sigma_vals=None,do_print=False):  #Question 4
        """ estimate alpha and sigma """
        ## Needs to estimate them such that they yield the estimated results

        par = self.par
        sol = self.sol 

          #Accounting for None values of Sigma and Alpha
        sigma_vals = par.sigma if sigma_vals is None else sigma_vals
        alpha_vals = par.alpha if alpha_vals is None else alpha_vals

        #Initial parameters
        b0 = par.beta0_target
        b1 = par.beta1_target
            
        #Solve optimal choice set, account for different wF
        self.solve_wF_vec(discrete=False)
            
        #Run regression for beta_0 and beta_1
        self.run_regression()
            
        return (b0-sol.beta0)**2 + (b1-sol.beta1)**2
    
    def estimate_2(self,alpha=None,sigma=None,do_print=False):  #Question 4
        """ estimate alpha and sigma """
        ## Needs to estimate them such that they yield the estimated results

        par = self.par
        sol = self.sol 

        def obj(x):

            #Initial parameters
            b0 = par.beta0_target
            b1 = par.beta1_target
            par.alpha = x[0]
            par.sigma = x[1]
            
            #Solve optimal choice set, account for different wF
            self.solve_wF_vec(discrete=False)
            
            #Run regression for beta_0 and beta_1
            self.run_regression()
            
            return (b0-sol.beta0)**2 + (b1-sol.beta1)**2
        
        #Setting bounds for alpha and sigma
        bnds = ((0,1),(0,5))

        # Initialize the best objective function value to a large numbe

        # Minimize objective function for alpha and sigma. Start value based on 3Dplot
        res = optimize.minimize(obj, x0=(0.9, 0.1), method='Nelder-Mead', bounds=bnds)

         #Saving results of alpha and sigma
        sol.alpha_hat = res.x[0]
        sol.sigma_hat = res.x[1]

        if do_print:
            print(res.message)
            print(f'alpha_hat: {res.x[0]:.4f}')
            print(f'sigma_hat: {res.x[1]:.4f}')

            print(f'beta0_hat: {sol.beta0:.4f}')
            print(f'beta1_hat: {sol.beta1:.4f}')
            print(f'Termination value: {obj(res.x):.4f}')

    def estimate_3(self,kappa_vals=None,sigma_vals=None,do_print=False):  #Question 5
        """ estimate kappa and sigma """
        ## Needs to estimate them such that they yield the estimated results

        par = self.par
        sol = self.sol 

          #Accounting for None values of Sigma and Alpha
        sigma_vals = par.sigma if sigma_vals is None else sigma_vals
        kappa_vals = par.kappa if kappa_vals is None else kappa_vals

        #Initial parameters
        b0 = par.beta0_target
        b1 = par.beta1_target
            
        #Solve optimal choice set, account for different wF
        self.solve_wF_vec(discrete=False)
            
        #Run regression for beta_0 and beta_1
        self.run_regression()
            
        return (b0-sol.beta0)**2 + (b1-sol.beta1)**2

    
    def estimate_4(self,sigma=None,kappa=None,do_print=False): #Question 5
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol 

        def obj(x):
            
            #Initial parameters
            b0 = par.beta0_target
            b1 = par.beta1_target
            par.sigma = x[0]
            par.kappa = x[1]
            
            #Solve optimal choice set, account for different wF
            self.solve_wF_vec(discrete=False)
            
            #Run regression for beta_0 and beta_1
            self.run_regression()
            
            return (b0-sol.beta0)**2 + (b1-sol.beta1)**2
        
        #Setting bounds for sigma and kappa
        bnds = ((0,5),(0,24))
        
        #Minimize objective function for alpha and sigma
        res = optimize.minimize(obj,x0=(0.8,5),method='Nelder-Mead',bounds = bnds)
        
        #Saving results of alpha and sigma
        sol.sigma_hat = res.x[0]
        sol.kappa_hat = res.x[1]

        if do_print:
            print(res.message)
            print(f'sigma_hat: {res.x[0]:.4f}')
            print(f'kappa_hat: {res.x[1]:.4f}')

            print(f'beta0_hat: {sol.beta0:.4f}')
            print(f'beta1_hat: {sol.beta1:.4f}')
            print(f'Termination value: {obj(res.x):.4f}')