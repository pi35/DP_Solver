import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf as RBFInterpolator
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint

# We attempt to minimize
# min c_T(x_T) + sum_{t=0}^{T-1} c_t(x_t,u_t)
# such that h(x_t,u_t) = 0, g(x_t,u_t) <= 0
# Here, total number of timesteps is T, so self.T = T

class dp_solver():
    def __init__(self, parameters, approximation_mode='interpolation'):
        self.state_set = parameters['state_set']
        self.discretize_precision = parameters['discretize_precision']
        self.action_set = parameters['action_set']
        self.forward_func = parameters['forward_func']
        self.cost_func = parameters['cost_func']
        self.terminal_cost_value = parameters['terminal_cost_values']
        self.T = parameters['total_timesteps']
        self.state_dimension = len(self.state_set)
        self.action_dimension = len(self.action_set)
        self.t = self.T - 1
        self.discretized_state = self.discretize(self.state_set)
        self.permuted_state = self.permute()
        self.num_ineq_constraints = parameters['num_ineq_constraint']
        self.g = parameters['g']
        self.lb = self.action_set[:,0]
        self.ub = self.action_set[:,1]
        self.noise = parameters['noise']
        if self.state_dimension == 1:
            self.num_state_1 = len(self.discretized_state[0])
            self.num_state_2 = 0
            self.dim_action = len(self.action_set)
            self.V = np.zeros((self.num_state_1, self.T))
            self.gamma = np.zeros((self.dim_action, self.num_state_1, self.T))
            self.Lagrange_ineq = np.ones((self.num_ineq_constraints, self.num_state_1, self.T))
            

        elif self.state_dimension == 2:
            self.num_state_1 = len(self.discretized_state[0])
            self.num_state_2 = len(self.discretized_state[1])
            self.dim_action = len(self.action_set)
            self.V = np.zeros((self.num_state_1, self.num_state_2, self.T))
            self.gamma = np.zeros((self.dim_action, self.num_state_1, self.num_state_2, self.T))
            self.Lagrange_ineq = np.ones((self.num_ineq_constraints, self.num_state_1, self.num_state_2, self.T))

    def terminal_cost_func(self, new_state):
        tmp = RBFInterpolator(self.permuted_state, self.terminal_cost_value, function='thin_plate')
        return tmp(new_state)
    
    def solve_dp(self):
        if self.state_dimension == 1:
            while(self.t >= 0):
                i = 0
                while(i < self.num_state_1):
                    self.current_state = np.reshape(np.array([self.discretized_state[0][i]]), (1,1))
                    self.gamma[:, i, self.t], self.V[i,self.t], self.Lagrange_ineq[:, i, self.t] = self.optimize()
                    i = i + 1
                print("Timestep", self.t, "done!")
                self.t = self.t -1
        elif self.state_dimension == 2:
            while(self.t >= 0):
                i = 0
                while(i < self.num_state_1):
                    j = 0
                    while(j < self.num_state_2):
                        self.current_state = np.reshape(np.array([self.discretized_state[0][i], self.discretized_state[1][j]]), (2,1))
                        self.gamma[:, i, j, self.t], self.V[i,j,self.t], self.Lagrange_ineq[:, i, j, self.t] = self.optimize()
                        j = j + 1
                    i = i + 1
                print("Timestep", self.t, "done!")
                self.t = self.t - 1
                
        print("All done")
        return


    def optimize(self):
        u_star = ((self.lb + self.ub) / 2).flatten()
        
        opt_func = lambda u: self.bellman_eqn(self.current_state.flatten(), u)
        non_cons = lambda u: self.g(self.current_state.flatten(), u, self.noise[:,self.t].flatten())
                                            
#        print(self.current_state, self.t)
#        print(opt_func(-2.))
#        print(non_cons(-2.))
        hess = lambda x: np.zeros((2,2))

        nonlinear_constraints = NonlinearConstraint(non_cons, \
                                  -np.inf*np.ones((self.num_ineq_constraints,1)).flatten(), \
                                  0.*np.ones((self.num_ineq_constraints,1)).flatten())
#        linear_constraints = LinearConstraint(1, \
#                                  -6*np.ones((1,1)).reshape(1,1), \
#                                  -2.*np.ones((1,1)).reshape(1,1))
        
        bounds = Bounds(self.lb, self.ub)
        
        res = minimize(opt_func, u_star, method = 'trust-constr', bounds = bounds, constraints = nonlinear_constraints, options = {'verbose': 0})
        
        
        u_star = res.x
        lm = np.array(res.v[0])
#        print(res)
#        print(lm)
        return u_star, self.bellman_eqn(self.current_state, u_star), lm
    

    def bellman_eqn(self, state, action):
        if self.t == self.T-1:
            # permuted state shape = (number of points, state dimension)
            # values shape = (number of points, 1)
            # state shape = (number of test points, state dimension), in our case (1,2)
            cost_at_state_action = self.cost_func(state, action, self.noise[:, self.t]) + \
                    self.terminal_cost_func(self.forward_func(state, action, self.noise[:, self.t]))
            return np.asscalar(cost_at_state_action) 
        else:
            if self.state_dimension == 1:
                interpolate_input = self.forward_func(state, action, self.noise[:,self.t])
                values = np.reshape(self.V[:, self.t+1], newshape=(self.discretized_state.shape[1],1))
                tmp = RBFInterpolator(self.permuted_state, values, function='thin_plate')
            elif self.state_dimension == 2:
                values = np.reshape(self.V[:, :, self.t+1], newshape=(self.discretized_state.shape[1] * self.discretized_state.shape[1], 1))
                tmp = RBFInterpolator(self.permuted_state, values, function='thin_plate')
                interpolate_input = np.reshape(self.forward_func(state, action, self.noise[:,self.t]), newshape=(1,2))
        return np.asscalar(self.cost_func(state, action, self.noise[:, self.t]) + tmp(interpolate_input))

    def discretize(self,M):
        dimension = len(M)
        row = 0
        result = []
        while(row < dimension):
            tmp = []
            i = M[row][0]
            while(i <= M[row][1]):
                tmp.append(i)
                i = i + self.discretize_precision
            row = row + 1
            result.append(tmp)
        return np.array(result)

    def permute(self):
        if self.state_dimension == 1:
            points = self.discretized_state.T
        elif self.state_dimension == 2:
            points = np.zeros(shape=(self.discretized_state.shape[1] * self.discretized_state.shape[1], 2))
            k = 0
            for i in range(self.discretized_state.shape[1]):
                for j in range(self.discretized_state.shape[1]):
                    points[k, 0] = self.discretized_state[0, i]
                    points[k, 1] = self.discretized_state[1, j]
                    k = k + 1
        return points

    def get_value(self):
        return self.V

    def get_policy(self):
        return self.gamma

    def get_Lagrange_multiplier_ineq(self):
        return self.Lagrange_ineq

    def get_permuted_state(self):
        return self.permuted_state
