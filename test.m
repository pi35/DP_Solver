clc
clear
% State description
X1 = 0:10;
X2 = 0:10;
X3 = [];
X4 = [];
X5 = [];
T_ = 11;

% Lagrangian description
Lc_T_ = 20; % Maximum iteration for augmented Lagrangian
c_ = 1000; % Initial coefficient for augmented Lagrangian

% % System description (1d)
% f_ = @(x, u) x - u; % State transition function
% J_t_ = @(x, u) (u - 15) * (u - 15);
% J_T_ = @(x, u) 0;
% lb_ = 0; % Lower bound for action
% ub_ = 10; % Upper bound for action
% f_lb_ = 0;
% f_ub_ = 10;
% h_ = @(x, u) 0;
% g_ = @(x, u) [1; -1; -1; 1]*u + [-1; 1; 0; 0]*x - [-f_lb_; f_ub_; -lb_; ub_];
% num_eq_con = 1;
% num_ineq_con = 4;
% nonlcon_ = []; % a function of x, u; if there's no nonlinear constraints, set to [].

% System description (2d)
f_ = @(x, u) x - u; % State transition function
J_t_ = @(x, u) (u - 15) * (u - 15);
J_T_ = @(x, u) 0;
lb_ = 0; % Lower bound for action, has same dimension with action
ub_ = 10; % Upper bound for action, has same dimension with action
h_ = @(x, u) 0;
g_ = @(x, u) [1; -1; 1; -1; 1]*u - [1, 0; 0, 0; 0, 1; 0, 0; 0, 0] * x - [0; 0; 0; -lb_; ub_];
h_N = h_;
g_N = g_;
num_eq_con = 1;
num_ineq_con = 5;
nonlcon_ = [];


% Use functions from DP_solver class
dp_solver = DP_solver(X1, X2, X3, X4, X5, T_, Lc_T_, f_, J_t_, J_T_, c_, num_eq_con, num_ineq_con, h_, h_N, g_, g_N, nonlcon_, lb_, ub_);
dp_solver = dp_solver.solve_dp();
V = dp_solver.get_V();
gamma = dp_solver.get_gamma();
% L_eq, L_ineq have dimension (num_constraints, num_state, total timestep)
[L_eq, L_ineq] = dp_solver.get_L_multiplier();
% 
% %L_eq(:,1,t)
% %L_ineq(:,2,t)



% for t = 1:T_
%     figure(t)
%     mesh(X1_, X2_, V(:, :, t))
%     xlabel('State x1')
%     ylabel('State x2')
%     zlabel('Value Function')
%     title("value at t = " + t)
% end
% 
% % Transfer Gamma from shape (1, n, m, t) to (n, m, t) for plotting
% gamma_ = reshape(gamma, [dp_solver.num_states_1, dp_solver.num_states_2, T_]);
% 
% for t = 1:T_
%     figure(t+T_)
%     mesh(X1_, X2_, gamma_(:, :, t))
%     xlabel('State x1')
%     ylabel('State x2')
%     zlabel('Value Function')
%     title("\gamma at t = " + t)
% end
