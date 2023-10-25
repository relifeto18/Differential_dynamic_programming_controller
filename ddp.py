import autograd.numpy as np
from autograd import grad, jacobian

class DifferentialDynamicProgramming():
    """
    Differential Dynamic PrograDmming Control
    Based on: http://www.imgeorgiev.com/2023-02-01-ddp/
    """
    def __init__(self, 
                 dynamics, 
                 compute_cost,
                 running_cost, 
                 terminal_cost,
                 max_iters=100,
                 epsilon=1e-3, 
                 horizon=10,
                 backtrack_max_iters=5, 
                 decay=0.9, 
                 state_dim=6, 
                 action_dim=1):
        self.dynamics = dynamics           # x_{t+1} = f(x_t, u_t): analytic dynamics function
        self.compute_cost = compute_cost   # J: cost function
        self.running_cost = running_cost   # g/L: running cost function
        self.terminal_cost = terminal_cost # h/phi: terminal cost function
        self.max_iters = max_iters         # maximal number of iterations
        self.epsilon = epsilon             # tolerance of cost
        self.T = horizon                   # length of state (0,1,...,T) action length (0,1,...,T-1)
        self.B = backtrack_max_iters       # maxmimal number of backtracking iterations
        self.decay = decay                 # decay: decay coefficients
        self.state_dim = state_dim         # dimension of state
        self.action_dim = action_dim       # dimension of action

        self.f = self.dynamics
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_xu = jacobian(self.f_x, 1)
        self.f_ux = jacobian(self.f_u, 0)
        self.f_uu = jacobian(self.f_u, 1)        
        
        self.g = self.running_cost
        self.g_x = grad(self.g, 0)
        self.g_u = grad(self.g, 1)
        self.g_xx = jacobian(self.g_x, 0)
        self.g_xu = jacobian(self.g_x, 1)
        self.g_ux = jacobian(self.g_u, 0)
        self.g_uu = jacobian(self.g_u, 1)

        self.h = self.terminal_cost
        self.h_x = grad(self.h)
        self.h_xx = jacobian(self.h_x)


    def _rollout_dynamics(self, state, actions):
        '''
        :param state: np array of shape (state_dim, )
        :param actions: np array of shape (T-1, action_dim)
        :returns states: np arrary of shape (T, state_dim)
        '''
        states = [state]
        for i in range(actions.shape[0]):
            states.append(self.dynamics(states[i], actions[i]))
        states = np.stack(states)
        return states


    def command(self, state):
        '''
        :param state: np array of shape (state_dim, )
        :returns action: np arrary of shape (action_dim, ), which is the best action
        '''
        
        assert isinstance(state, np.ndarray)
        U = U = np.random.uniform(-1.0, 1.0, (self.T-1, self.action_dim))
        X = self._rollout_dynamics(state, U)
        
        counter1 = 0
        prev_cost = 0
        curr_cost = self.compute_cost(X, U)
        mu1 = 0.01
        mu2 = 0.01
        while counter1 < self.max_iters and abs(curr_cost - prev_cost) > self.epsilon:
            restart_flag = False
            
            # backward pass
            V = self.h(X[-1])
            Vx = self.h_x(X[-1])
            Vxx = self.h_xx(X[-1])

            k_gains = []
            K_gains = []
            for t in range(self.T-2, -1, -1):
                fx = self.f_x(X[t], U[t])
                fu = self.f_u(X[t], U[t])

                gx = self.g_x(X[t], U[t])
                gu = self.g_u(X[t], U[t])
                gxx = self.g_xx(X[t], U[t])
                guu = self.g_uu(X[t], U[t])
                gux = self.g_ux(X[t], U[t])

                Qx = gx + fx.T @ Vx
                Qu = gu + fu.T @ Vx
                
                Qxx = gxx + fx.T @ (Vxx + mu1 * np.eye(self.state_dim)) @ fx
                Qux = gux + fu.T @ (Vxx + mu1 * np.eye(self.state_dim)) @ fx
                Quu = guu + fu.T @ (Vxx + mu1 * np.eye(self.state_dim)) @ fu + mu2 * np.eye(self.action_dim)
                
                if abs(np.linalg.det(Quu)) < 1e-6:
                    restart_flag = True
                    mu1 += 0.01
                    mu2 += 0.01
                    break
                
                k = -np.linalg.pinv(Quu) @ Qu
                K = -np.linalg.pinv(Quu) @ Qux
                k_gains.append(k)
                K_gains.append(K)

                Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
                Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
                
            if restart_flag:
                continue

            # line search
            counter2 = 0
            alpha = 1.0
            k_gains_reversed = k_gains[::-1]
            K_gains_reversed = K_gains[::-1]

            while counter2 < self.B:
                X_bar = np.zeros_like(X)
                U_bar = np.zeros_like(U)
                X_bar[0] = X[0].copy()

                for t in range(self.T-1):
                    U_bar[t] = U[t] + alpha * k_gains_reversed[t] + K_gains_reversed[t] @ (X_bar[t] - X[t])
                    X_bar[t + 1] = self.f(X_bar[t], U_bar[t])

                cost = self.compute_cost(X_bar, U_bar)
                if cost < curr_cost:
                    X = X_bar
                    U = U_bar
                    break
                else:
                    alpha *= self.decay
                
                counter2 += 1
            
            prev_cost = curr_cost
            curr_cost = cost
            counter1 += 1
        return U[0]
