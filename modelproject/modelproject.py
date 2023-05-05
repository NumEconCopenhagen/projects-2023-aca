import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynHouseholdLaborModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods

        # b. preferences
        par.eta = 2.0 # risk aversion
        par.omega = 0.5 # weight on consumption
        par.gamma = 2.5 # curvature on hours 

        par.rho_01 = 0.1 # weight on labor and home production dis-utility of men
        par.rho_02 = 0.1 # weight on labor and home production dis-utility of women
        par.rho_11 = 0.05 # extra distutility of total labor  for women 
        par.rho_12 = 0.05 # extra distutility of total labor for women 
        par.rho_21 = 0.025 # extra extra utility of working in home for men
        par.rho_22 = 0.025 # extra extra utility of working in home for women  

        par.norms = 1. # 1: equal gender norms, 0: non-equal gender norms 

        par.beta = 0.98 # discount factor

        # c. household production 
        par.sigma = 1. # elasticity of substitition between male and female home production
        par.alpha = 0.5 #productivity in home production 
        par.upsilon = 2. #curvature of home production
        
        # d. labor supply
        par.gamma = 2.5 # curvature on labor hours 
       
        # income
        par.wage_const_1 = 1.0 # constant, men
        par.wage_const_2 = 1.0 # constant, women
        par.wage_K_1 = 0.1 # return on human capital, men
        par.wage_K_2 = 0.1 # return on human capital, women

        par.delta = 0.1 # depreciation in human capital
        par.delta2 = 0. #decreasing returns to wages for women

        # grids        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 20 #30 # number of grid points in wealth grid 

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals

        #kids 
        par.Nn = 2
        par.p_birth = 0.1


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        par.n_grid = np.arange(par.Nn)

        # d. solution arrays
        
        shape = (par.T,par.Nn,par.Nk,par.Nk)
        sol.l1 = np.nan + np.zeros(shape)
        sol.h1 = np.nan + np.zeros(shape)
        sol.l2 = np.nan + np.zeros(shape)
        sol.h2 = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.l1 = np.nan + np.zeros(shape)
        sim.h1 = np.nan + np.zeros(shape)
        sim.l2 = np.nan + np.zeros(shape)
        sim.h2 = np.nan + np.zeros(shape)
        sim.k1 = np.nan + np.zeros(shape)
        sim.k2 = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)

        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)
        
        sim.income1 = np.nan + np.zeros(shape)
        sim.income2 = np.nan + np.zeros(shape)
        sim.income_hh = np.nan + np.zeros(shape)

        sim.transfers = np.nan + np.zeros(shape)

        # g. initialization
        sim.k1_init = np.zeros(par.simN)
        sim.k2_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)

    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # i. loop over state variables: human capital for each household member
            for i_n, kids in enumerate(par.n_grid):
                for i_k1,capital1 in enumerate(par.k_grid):
                    for i_k2,capital2 in enumerate(par.k_grid):
                        idx = (t,i_n,i_k1,i_k2)
                            
                            # ii. find optimal consumption and hours at this level of wealth in this period t.
                        if t==(par.T-1): # last period
                            obj = lambda x: -self.util(x[0],x[1],x[2],x[3],capital1,capital2,kids)

                        else:
                            obj = lambda x: - self.value_of_choice(x[0],x[1],x[2],x[3],capital1,capital2,kids,sol.V[t+1])  

                        # call optimizer
                        bounds = ((0,24), (0,24), (0,24), (0,24))
                            
                        init = np.array([6,6,6,6])

                        cons = []
                        cons.append({'type': 'ineq', 'fun': lambda x: 24 - x[0] - x[2]})
                        cons.append({'type': 'ineq', 'fun': lambda x: 24 - x[1] - x[3]})

                        res = minimize(obj,init,bounds=bounds,constraints=cons,tol=1e-8) 

                        # store results
                        sol.l1[idx] = res.x[0]
                        sol.l2[idx] = res.x[1]
                        sol.h1[idx] = res.x[2]
                        sol.h2[idx] = res.x[3]
                        sol.V[idx] = -res.fun

    def value_of_choice(self,labor1,labor2,home1,home2,capital1,capital2,kids,V_next):

        # a. unpack
        par = self.par

        # b. current utility
        util = self.util(labor1,labor2,home1,home2,capital1,capital2,kids)

        EV_next = 0.0
        k1_next = (1.0-par.delta)*capital1 + labor1
        k2_next = (1.0-par.delta)*capital2 + labor2

        prob = [1-par.p_birth,par.p_birth]
        num_birth = 2 if kids<par.Nn-1 else 1
        for birth in range(num_birth):
            p_n_next = prob[birth] if num_birth > 1 else 0 
            n_next = kids + birth 
            V_next_interp = interp_2d(par.k_grid,par.k_grid,V_next[n_next],k1_next,k2_next)

            EV_next = EV_next + p_n_next*V_next_interp

        # d. return value of choice
        return util + par.beta*EV_next

  # relevant functions
    def consumption(self,labor1,labor2,capital1,capital2,kids):
        par = self.par

        income1 = self.wage_func(capital1,1,kids) * labor1
        income2 = self.wage_func(capital2,2,kids) * labor2
        income_hh = income1+income2

        return income_hh
    
    def home(self,home1,home2):

        par = self.par 

        if par.sigma==1:
            H = home1**(1-par.alpha)*home2**par.alpha
        elif par.sigma==0:
            H = np.min(home1,home2)
        else:
            H = ((1-par.alpha)*home1**((par.sigma-1)/par.sigma)+par.alpha*home2**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        return H

    def wage_func(self,capital,sex,kids):
        # before tax wage rate
        par = self.par

        if kids == 1: 
            constant = par.wage_const_1
            return_K = par.wage_K_1
            if sex>1:
                constant = par.wage_const_2
                return_K = par.wage_K_2*(1. - par.delta2)
            
        else: 
            constant = par.wage_const_1
            return_K = par.wage_K_1
            if sex>1:
                constant = par.wage_const_2
                return_K = par.wage_K_2

        return np.exp(constant + return_K * capital)

    def util(self,labor1,labor2,home1,home2,capital1,capital2,kids):
        par = self.par

        C = self.consumption(labor1,labor2,capital1,capital2,kids)

        H = self.home(home1,home2)

        Q = C**par.omega*H**(1-par.omega)

        total1 = labor1 + home1
        total2 = labor2 + home2 

        rho1 = par.rho_01 + par.rho_11*kids*par.norms
        rho2 = par.rho_02 + par.rho_12*kids

        util_1 = ((Q/2))**(1.0-par.eta) / (1.0-par.eta) - rho1*(total1)**(1.0+par.gamma) / (1.0+par.gamma) + par.norms*par.rho_21*kids*(home1)**(1.0+par.gamma) / (1.0+par.gamma)
        util_2 = ((Q/2))**(1.0-par.eta) / (1.0-par.eta) - rho2*(total2)**(1.0+par.gamma) / (1.0+par.gamma) + par.rho_22*kids*(home2)**(1.0+par.gamma) / (1.0+par.gamma)

        return util_1 + util_2

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.k1[i,0] = sim.k1_init[i]
            sim.k2[i,0] = sim.k2_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal hours
                idx_sol = (t,sim.n[i,t])
                #idx_sol = t
                sim.l1[i,t] = interp_2d(par.k_grid,par.k_grid,sol.l1[idx_sol],sim.k1[i,t],sim.k2[i,t])
                sim.h1[i,t] = interp_2d(par.k_grid,par.k_grid,sol.h1[idx_sol],sim.k1[i,t],sim.k2[i,t])
                sim.l2[i,t] = interp_2d(par.k_grid,par.k_grid,sol.l2[idx_sol],sim.k1[i,t],sim.k2[i,t])
                sim.h2[i,t] = interp_2d(par.k_grid,par.k_grid,sol.h2[idx_sol],sim.k1[i,t],sim.k2[i,t])  

                # store income
                sim.income1[i,t] = self.wage_func(sim.k1[i,t],1,sim.n[i,t])*sim.l1[i,t]
                sim.income2[i,t] = self.wage_func(sim.k2[i,t],2,sim.n[i,t])*sim.l2[i,t]
                sim.income_hh[i,t] = sim.income1[i,t] + sim.income2[i,t]
                        
                # iii. store next-period states
                if t<par.simT-1:
                    sim.k1[i,t+1] = (1.0-par.delta)*sim.k1[i,t] + sim.l1[i,t]
                    sim.k2[i,t+1] = (1.0-par.delta)*sim.k2[i,t] + sim.l2[i,t]


                    birth = 0 
                    if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1))):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth
                    

                    
                    


