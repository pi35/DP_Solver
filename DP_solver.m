classdef DP_solver
    
    properties
        X1
        X2
        X3
        X4
        X5
        dim_state
        num_states_1
        num_states_2
        num_states_3
        num_states_4
        num_states_5
        V
        gamma
        t
        Lc_T
        L_epsilon_lambda
        L_epsilon_mu
        T
        num_eq_constraints
        num_ineq_constraints
        f
        J_t
        J_T
        c
        h
        h_N
        g
        g_N
        lb
        ub
        L_eq
        L_ineq
        dim_action
        nonlcon
    end
    
    methods
        function obj = DP_solver(X1_, X2_, X3_, X4_, X5_, T_, Lc_T_, L_epsilon_lambda_, L_epsilon_mu_, f_, J_t_, J_T_, c_, num_eq_con, num_ineq_con, h_, h_N_, g_, g_N_, nonlcon_, lb_, ub_)
            % Detailed explanation goes here
            obj.X1 = X1_;
            obj.X2 = X2_;
            obj.X3 = X3_;
            obj.X4 = X4_;
            obj.X5 = X5_;
            obj.T = T_;
            obj.Lc_T = Lc_T_;
            obj.L_epsilon_lambda = L_epsilon_lambda_;
            obj.L_epsilon_mu = L_epsilon_mu_;
            obj.f = f_;
            obj.J_t = J_t_;
            obj.J_T = J_T_;
            obj.c = c_;
            obj.dim_state = find_dim_state(obj);
            obj.num_states_1 = length(obj.X1);
            obj.num_states_2 = length(obj.X2);
            obj.num_states_3 = length(obj.X3);
            obj.num_states_4 = length(obj.X4);
            obj.num_states_5 = length(obj.X5);
            obj.num_eq_constraints = num_eq_con; 
            obj.num_ineq_constraints = num_ineq_con; 
            obj.dim_action = length(lb_);
            if(obj.dim_state == 1)
                obj.V = zeros(obj.num_states_1, obj.T);
                obj.gamma = zeros(obj.dim_action, obj.num_states_1, obj.T);
                obj.L_eq = ones(obj.num_eq_constraints, obj.num_states_1, obj.T);
                obj.L_ineq = ones(obj.num_ineq_constraints, obj.num_states_1, obj.T);
            elseif(obj.dim_state == 2)
                obj.V = zeros(obj.num_states_1, obj.num_states_2, obj.T);
                obj.gamma = zeros(obj.dim_action, obj.num_states_1, obj.num_states_2, obj.T);
                obj.L_eq = ones(obj.num_eq_constraints, obj.num_states_1, obj.num_states_2, obj.T);
                obj.L_ineq = ones(obj.num_ineq_constraints, obj.num_states_1, obj.num_states_2, obj.T);
            elseif(obj.dim_state == 3)
                obj.V = zeros(obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.T);
                obj.gamma = zeros(obj.dim_action, obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.T);
                obj.L_eq = ones(obj.num_eq_constraints, obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.T);
                obj.L_ineq = ones(obj.num_ineq_constraints, obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.T);
            elseif(obj.dim_state == 4)
                obj.V = zeros(obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.num_states_4, obj.T);
                obj.gamma = zeros(obj.dim_action, obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.num_states_4, obj.T);
                obj.L_eq = ones(obj.num_eq_constraints, obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.num_states_4, obj.T);
                obj.L_ineq = ones(obj.num_ineq_constraints, obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.num_states_4, obj.T);
            elseif(obj.dim_state == 5)
                obj.V = zeros(obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.num_states_4, obj.num_states_5, obj.T);
                obj.gamma = zeros(obj.dim_action, obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.num_states_4, obj.num_states_5, obj.T);
                obj.L_eq = ones(obj.num_eq_constraints, obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.num_states_4, obj.num_states_5, obj.T);
                obj.L_ineq = ones(obj.num_ineq_constraints, obj.num_states_1, obj.num_states_2, obj.num_states_3, obj.num_states_4, obj.num_states_5, obj.T);
            end
            obj.t = T_ - 1;
            obj.h = h_; % equality constraints, a function of (x, u)
            obj.h_N = h_N_; % terminal equality constraints
            obj.g = g_; % inequality constraints, a function of (x, u)
            obj.g_N = g_N_; % terminal inequality constraints
            obj.nonlcon = nonlcon_;
            obj.lb = lb_;
            obj.ub = ub_;           
        end
        
        function dim_state = find_dim_state(self)
            if(isempty(self.X2))
                dim_state = 1;
            elseif(isempty(self.X3))
                dim_state = 2;
            elseif(isempty(self.X4))
                dim_state = 3;
            elseif(isempty(self.X5))
                dim_state = 4;
            else 
                dim_state = 5;
            end
        end
        
        function V_ = get_V(self)
            V_ = self.V;
        end
        
        function gamma_ = get_gamma(self)
            gamma_ = self.gamma;
        end
        
        function [L_eq_, L_ineq_] = get_L_multiplier(self)
            L_eq_ = self.L_eq;
            L_ineq_ = self.L_ineq;
        end
        
        function self = solve_dp(self)
            while(self.t > 0)
                fprintf("Timestep %d done! \n", self.t);
                if(self.dim_state == 1)
                    for i = 1:self.num_states_1
                        x = self.X1(i);
                        [self.gamma(:, i, self.t), self.V(i, self.t), self.L_eq(:, i, self.t), self.L_ineq(:, i, self.t)]...
                            = optimize(x, self);
                    end
                elseif(self.dim_state == 2)
                    for i = 1 : self.num_states_1
                        for j = 1: self.num_states_2
                        x = [self.X1(i); self.X2(j)];
                        [self.gamma(:, i, j, self.t), self.V(i, j, self.t), self.L_eq(:, i, j, self.t), self.L_ineq(:, i, j, self.t)]...
                            = optimize(x, self);
                        end
                    end
                elseif(self.dim_state == 3)
                    for i = 1 : self.num_states_1
                        for j = 1: self.num_states_2
                            for k = 1: self.num_states_3
                                x = [self.X1(i); self.X2(j); self.X3(k)];
                                [self.gamma(:, i, j, k, self.t), self.V(i, j, k, self.t), self.L_eq(:, i, j, k, self.t), self.L_ineq(:, i, j, k, self.t)]...
                                = optimize(x, self);
                            end
                        end
                    end
                elseif(self.dim_state == 4)
                    for i = 1 : self.num_states_1
                        for j = 1: self.num_states_2
                            for k = 1: self.num_states_3
                                for l = 1: self.num_states_4
                                    x = [self.X1(i); self.X2(j); self.X3(k); self.X4(l)];
                                    [self.gamma(:, i, j, k, l, self.t), self.V(i, j, k, l, self.t), self.L_eq(:, i, j, k, l, self.t), self.L_ineq(:, i, j, k, l, self.t)]...
                                    = optimize(x, self);
                                end
                            end
                        end
                    end
                elseif(self.dim_state == 5)
                    for i = 1 : self.num_states_1
                        for j = 1: self.num_states_2
                            for k = 1: self.num_states_3
                                for l = 1: self.num_states_4
                                    for m = 1: self.num_states_5
                                    x = [self.X1(i); self.X2(j); self.X3(k); self.X4(l); self.X5(m)];
                                    [self.gamma(:, i, j, k, l, m, self.t), self.V(i, j, k, l, m, self.t), self.L_eq(:, i, j, k, l, m, self.t), self.L_ineq(:, i, j, k, l, m, self.t)]...
                                    = optimize(x, self);
                                    end
                                end
                            end
                        end
                    end
                end
                self.t = self.t - 1;
            end
            fprintf("All done!\n");
        end
        
        function [gamma_star, J_star, L_eq, L_ineq] = optimize(x, self)
            options = optimoptions('fmincon','Display','off','TolCon',1e-8,'TolX',1e-25,'TolFun',1e-15);
            % The start point of the optimization is x (for this specific problem)
            % Get initial point for optimization
            u_star = (self.ub + self.lb) / 2;
            % We need h and g for updating lambda and mu
            % Initialize lambda and mu 
            lambda_prev = ones(self.num_eq_constraints, 1);
            lambda_new = ones(self.num_eq_constraints, 1);
            mu_prev = ones(self.num_ineq_constraints, 1);
            mu_new = ones(self.num_ineq_constraints, 1);
            c_tmp = self.c; % Penalty coeeficient in augmented Lagrangian method
            [h_tmp, g_tmp] = get_constraints(self);
            % Iterate Lc_T timesteps, in each time step, increase c.
            for L_idx = 1:self.Lc_T
                [u_star, J_val, ~, ~, ~] = ...
                    fmincon(@(u) L_c(x, u, lambda_prev, mu_prev, c_tmp, self), u_star, [], [], [], [], self.lb, self.ub, self.nonlcon, options);
                % If L_multipliers converge, break out the loop
                if((L_idx > 1) && (norm(lambda_new - lambda_prev, 2) <= self.L_epsilon_lambda) && (norm(mu_new - mu_prev, 2) <= self.L_epsilon_mu))
                    break
                end
                lambda_prev = lambda_new;
                mu_prev = mu_new;
                % Update L-multipliers
                lambda_new = lambda_prev + c_tmp * h_tmp(x, u_star);
                mu_new = max(0, mu_prev + c_tmp * g_tmp(x, u_star));
                % increase c
                c_tmp = c_tmp * 1.5;
            end
            fprintf("L_multipliers converges in %d iterations\n", L_idx)
            % Return the variables
            gamma_star = u_star;
            J_star = J_val;
            L_eq = lambda_new;
            L_ineq = mu_new;
        end
        
        function [h, g] = get_constraints(self)
           if(self.t == self.T - 1)
               h = self.h_N;
               g = self.g_N;
           else
               h = self.h;
               g = self.g;
           end
        end
        
        function Lc = L_c(x, u, lambda, mu, c, self)
            % Refer to Nonlinear Programming book (2nd version)
            % h(x) = 0, g(x) <= 0
            [h_tmp, g_tmp] = get_constraints(self);
            Lc = bellman_eqn(x, u, self) + lambda'*h_tmp(x,u) + c/2*h_tmp(x,u)'*h_tmp(x,u)...
                + 1/(2*c) * (ones(1,self.num_ineq_constraints)*(max(0,mu+c*g_tmp(x,u)).*max(0,mu+c*g_tmp(x,u)) - mu.*mu));
        end
        
        function bellman_eqn = bellman_eqn(x, u, self)
            if(self.dim_state == 1)
                bellman_eqn = self.J_t(x, u) + interp1(self.X1, self.V(:, self.t + 1), self.f(x, u), 'spline');
            elseif(self.dim_state == 2)
                x_next = self.f(x, u);
                bellman_eqn = self.J_t(x, u) + interp2(self.X1, self.X2, self.V(:, :, self.t + 1).', x_next(1), x_next(2), 'spline');
            elseif(self.dim_state == 3)
                x_next = self.f(x, u);
                newV = permute(self.V(:, :, :, self.t + 1), [2, 1, 3]); % See documentation of interp3. The dimension should be like this.
                bellman_eqn = self.J_t(x, u) + interp3(self.X1, self.X2, self.X3, newV, x_next(1), x_next(2), x_next(3), 'spline');
            elseif(self.dim_state == 4)
                x_next = self.f(x, u);
                bellman_eqn = self.J_t(x, u) + interpn(self.X1, self.X2, self.X3, self.X4, self.V(:, :, :, :, self.t + 1), x_next(1), x_next(2), x_next(3), x_next(4), 'spline');
            elseif(self.dim_state == 5)
                x_next = self.f(x, u);
                bellman_eqn = self.J_t(x, u) + interpn(self.X1, self.X2, self.X3, self.X4, self.X5, self.V(:, :, :, :, :, self.t + 1), x_next(1), x_next(2), x_next(3), x_next(4), x_next(5), 'spline');
            else
                bellman_eqn = 0; % code if >=6 dimension needed
            end
        end
    end
end

