from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

# write your code here

class TaxationClass:
    
    def __init__(self):

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # setup parameters 
        par.alpha = 0.5
        par.kappa = 1.0
        par.upsilon = 1/(2*16**2)
        par.omega = 1.0
        par.tau = 0.30

        #question 5 and 6
        par.sigma = 1.0
        par.rho = 0.
        par.epsilon = 1.0

    def consumption(self,labor):
        par = self.par

        C = par.kappa + (1-par.tau)*par.omega*labor 

        return C
    
    def government(self,labor):
        par = self.par

        G = par.tau*par.omega*labor

        return G
    
    def calc_utility(self,labor):
        par = self.par

        C = self.consumption(labor)

        G = self.government(labor)

        if par.sigma == 1 and par.rho == 0.: 

            utility = np.log(C**par.alpha * G**(1-par.alpha)) 
            disutility = par.upsilon*(labor**2/2) 

        else:
                
            power1 = (par.sigma-1)/par.sigma
            power2 = par.sigma / (par.sigma-1)
    
            utility = (((par.alpha*C**power1 + (1-par.alpha)*G**power1)**power2)**(1-par.rho) - 1) / (1-par.rho)
            disutility = par.upsilon*(labor**(1+par.epsilon))/(1+par.epsilon)

        V = utility - disutility 
            
        return V

    def solve(self):

        par = self.par
        opt = SimpleNamespace() 

        # a. define objective function
        def obj(x):
            return -self.calc_utility(x[0])
        
        # b. define contracints and bounds to minimize
        cons = []
        cons.append({'type': 'ineq', 'fun': lambda x: 24 - x[0]})
        bns = [(0, 24)] 
        initial_guess = 15

        # c. Define solver
        res = optimize.minimize(obj, x0=initial_guess, method='SLSQP', constraints = cons, bounds = bns, tol=1e-10)

        opt.labor = res.x[0]

        return opt
    
    def optimal_tax(self,tau=None,do_print=False):  #Question 4
       
        par = self.par
        sol = self.sol 

        def obj(x):

            #Initial parameters
            par.tau = x[0]

            # 
            opt = self.solve()
            #
            utility = -self.calc_utility(labor=opt.labor)
            
            return utility
        
        #Setting bounds for alpha and sigma
        bns = [(1e-10, 1)] 

        # 
        res = optimize.minimize(obj, x0=(0.2), method='Nelder-Mead', bounds=bns,tol=1e-10)

         #Saving results of alpha and sigma
        sol.tau_hat = -res.x[0]

        if do_print:
            print(res.message)
            print(f'tau_hat: {res.x[0]:.2f}')
            print(f'Utility value: {-obj(res.x):.2f}') # We negate the value because it represents a minimizer.