# DP Solver
Note: To view the equations correctly, download [MathJax](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en)

This project is a solver to general dynamic programming problems. The solver takes a dynamic problem and outputs the value, optimal action and lagrangian for each state at each timestep. The solver supports at most 5-dimensional state problems. 

## Using the solver
[test.m](test.m) shows the way of using the solver. For some specific dynamic problems, the user needs to change the state description, lagrangian description and system description.

## Dynamic Programming (DP)
DP yields an optimal closed loop policy (A policy that makes action with considering the state). Also, DP yields a strongly time-consistent policy, i.e. the policy $(\gamma_t, ..., \gamma_{T-1})$ is optimal no matter what happens in the past.

Consider a finite horizon problem with finite time $T$. We have the following notations

&emsp; 1) State $ x_t $

&emsp; 2) Action $ u_t $

&emsp; 3) Dynamic of the system $x_{t+1} = f_t(x_t, u_t) $

&emsp; 4) Strategy(Policy) $\gamma_t : \mathbb{X}_t \to U_t$, and $\gamma = (\gamma_1, \gamma_2, ..., \gamma_T)$

&emsp; Policy $\bar{\gamma}^* $ is optimal iff the policy is optimal for the truncated problem from time $t$ to $T$. This holds for all $ t = 0, 1, ..., T - 1 $.

For every time step, we compute the following

&emsp; 1) $\gamma_t^* (x_t) = argmin_{u_t}[g_t(x_t, u_t) + V_{t+1}(f_t(x_t, u_t))]$

&emsp; 2) $V_t(x_t) = min_{u_t}[g_t(x_t, u_t) + V_{t+1}(f_t(x_t, u_t))]$

Starting with $V_T(x_T) = g_T(x_T)$, we compute the policy, value and Lagrange multipliers from time $t=T-1$ to $t=0$.
