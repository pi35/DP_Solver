import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.interpolate import RBFInterpolator
from scipy.optimize import Bounds
class dp_solver():
    def __init__(self, parameters, approximation_mode='interpolation'):
        self.state_set = parameters['state_set']
        self.discretize_precision = parameters['discretize_precision']
        self.action_set = parameters['action_set']
        self.forward_func = parameters['forward_func']
        self.cost_func = parameters['cost_func']
        self.terminal_cost_func_parameters = parameters['terminal_cost_func']
        self.T = parameters['total_timesteps']
        self.state_dimension = len(self.state_set)
        self.action_dimension = len(self.action_set)
        self.Lagrange_penalty = parameters['Lagrange_penalty']
        self.Lagrange_total_timestep = parameters['Lagrange_total_timestep']
        self.t = self.T - 2
        self.discretized_state = self.discretize(self.state_set)
        self.permuted_state = self.permute()
        self.num_eq_constraints = parameters['num_eq_constraint']
        self.num_ineq_constraints = parameters['num_ineq_constraint']
        self.h = parameters['h']
        self.g = parameters['g']
        self.lb = self.action_set[0,0]
        self.ub = self.action_set[0,-1]
        self.bound = Bounds(self.lb, self.ub)
        self.generate_noise = parameters['noise_generator']
        self.approximation_mode = approximation_mode
        self.noise = parameters['noise']
        if self.state_dimension == 1:
            self.num_state_1 = len(self.discretized_state[0])
            self.num_state_2 = 0
            self.dim_action = len(self.action_set)
            self.V = np.zeros((self.num_state_1, self.T))
            self.gamma = np.zeros((self.dim_action, self.num_state_1, self.T))
            self.Lagrange_eq = np.ones((self.num_eq_constraints, self.num_state_1, self.T))
            self.Lagrange_ineq = np.ones((self.num_ineq_constraints, self.num_state_1, self.T))

        elif self.state_dimension == 2:
            self.num_state_1 = len(self.discretized_state[0])
            self.num_state_2 = len(self.discretized_state[1])
            self.dim_action = len(self.action_set)
            self.V = np.zeros((self.num_state_1, self.num_state_2, self.T))
            self.gamma = np.zeros((self.dim_action, self.num_state_1, self.num_state_2, self.T))
            self.Lagrange_eq = np.ones((self.num_eq_constraints, self.num_state_1, self.num_state_2, self.T))
            self.Lagrange_ineq = np.ones((self.num_ineq_constraints, self.num_state_1, self.num_state_2, self.T))


    def solve_dp(self):
        if self.state_dimension == 1:
            while(self.t >= 0):
                i = 0
                while(i < self.num_state_1):
                    self.current_state = np.reshape(np.array([self.discretized_state[0][i]]), (1,1))
                    self.gamma[:, i, self.t], self.V[i,self.t], self.Lagrange_eq[:, i, self.t], self.Lagrange_ineq[:, i, self.t] = self.optimize()
                    i = i + 1
                self.t = self.t - 1
                print("Timestep", self.t+1, "done!")
        elif self.state_dimension == 2:
            while(self.t >= 0):
                i = 0
                while(i < self.num_state_1):
                    j = 0
                    while(j < self.num_state_2):
                        self.current_state = np.reshape(np.array([self.discretized_state[0][i], self.discretized_state[1][j]]), (2,1))
                        self.gamma[:, i, j, self.t], self.V[i,j,self.t], self.Lagrange_eq[:, i, j, self.t], self.Lagrange_ineq[:, i, j, self.t] = self.optimize()
                        j = j + 1
                    i = i + 1
                self.t = self.t - 1
                print("Timestep", self.t+1, "done!")
        print("All done")
        return



    def optimize(self):
        u_star = (self.lb + self.ub) / 2
        #Initialize lambda and mu
        self.lbda = np.ones(shape=(self.num_eq_constraints, self.Lagrange_total_timestep))
        self.mu = np.ones(shape=(self.num_ineq_constraints, self.Lagrange_total_timestep))

        self.c_tmp = self.Lagrange_penalty
        self.L_idx = 0  #Iteration of augmented L-multiplier algorithm
        while(self.L_idx < self.Lagrange_total_timestep - 1):
            # optimization of L_c, need u_star, and J
            u_star = minimize(self.Lc, u_star, method='Nelder-Mead', bounds = self.bound, options={'disp': False})["x"][0]
            # Update Lagrange multipliers
            if self.num_eq_constraints > 0:
                self.lbda[:, self.L_idx+1] = self.lbda[:, self.L_idx] + self.c_tmp * self.h(self.current_state, u_star)
            if self.num_ineq_constraints > 0:
                self.mu[:, self.L_idx+1] = np.maximum(np.zeros_like(self.mu[:, self.L_idx] + self.c_tmp * np.transpose(self.g(self.current_state,u_star, self.noise))), self.mu[:, self.L_idx] + self.c_tmp * np.transpose(self.g(self.current_state,u_star, self.noise)))
            self.c_tmp = self.c_tmp * 1.5
            self.L_idx = self.L_idx + 1
        L_eq = self.lbda[:, -1]
        L_ineq = self.mu[:, -1]
        print(u_star)
        return u_star, self.Lc(u_star), L_eq, L_ineq

    def Lc(self, action):
        lbda = 0
        mu = np.reshape(np.array(self.mu[:, self.L_idx]), (self.num_ineq_constraints, 1))
        c = self.c_tmp
        m21 = np.maximum(np.zeros_like(mu+c*self.g(self.current_state, action, self.noise)), mu+c*self.g(self.current_state, action, self.noise))
        m2 = np.multiply(m21, m21)
        lc_output = self.bellman_eqn(self.current_state, action) + \
            np.multiply(np.transpose(lbda), self.h(self.current_state, action)) + \
            c/2*np.multiply(np.transpose(self.h(self.current_state, action)), self.h(self.current_state,action)) + \
            1/(2*c) * np.sum(m2 - np.multiply(mu,mu))
        return lc_output

    def bellman_eqn(self, state, action):
        if self.approximation_mode == 'interpolation':
            if self.state_dimension == 1:
                interpolate_input = self.forward_func(state, action, self.noise)
                values = np.reshape(self.V[:, self.t+1], newshape=(self.discretized_state.shape[1],1))
                tmp = RBFInterpolator(self.permuted_state, values, kernel='cubic')
            elif self.state_dimension == 2:
                values = np.reshape(self.V[:, :, self.t+1], newshape=(self.discretized_state.shape[1] * self.discretized_state.shape[1], 1))
                tmp = RBFInterpolator(self.permuted_state, values, kernel='cubic')
                interpolate_input = np.reshape(self.forward_func(state, action, self.noise), newshape=(1,2))
        return self.cost_func(state, action, self.noise) + tmp(interpolate_input)

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
        print("value:",self.V)

    def get_policy(self):
        print("policy : ", self.gamma)

    def get_Lagrange_multiplier_eq(self):
        print(self.Lagrange_eq)

    def get_Lagrange_multiplier_ineq(self):
        print("Lagrange multiplier ineq:", self.Lagrange_ineq)
