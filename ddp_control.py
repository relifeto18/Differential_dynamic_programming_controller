from ddp import *
from ddp_utils import *
import autograd.numpy as np

class InvertedDoublePendulumDDPController(object):
    def __init__(self, start_state, goal_state, hyperparams):
        self.start_state = start_state
        self.curr_state = start_state
        self.goal_state = goal_state
        self.state_dim = 6
        self.action_dim = 1
        self.counter = 0
        
        self.error_Q = hyperparams['error_Q']
        max_iters = hyperparams['max_iters']
        epsilon = hyperparams['epsilon']
        horizon = hyperparams['horizon']
        backtrack_max_iters = hyperparams['backtrack_max_iters']
        decay = hyperparams['decay']
        
        self.ddp = DifferentialDynamicProgramming(dynamics=idp_dynamics_analytic_numpy,
                                                  compute_cost=compute_cost,
                                                  running_cost=running_cost,
                                                  terminal_cost=terminal_cost,
                                                  max_iters=max_iters,
                                                  epsilon=epsilon,  
                                                  horizon=horizon,
                                                  backtrack_max_iters=backtrack_max_iters,
                                                  decay=decay)
    
    def control(self):
        action = self.ddp.command(self.curr_state)
        self.curr_state = idp_dynamics_analytic_numpy(self.curr_state, action)
        self.counter += 1
        
    def render(self):
        plot_idp(self.curr_state, self.counter)
        
    def calculate_error(self):
        '''calculate error between current state and goal state'''
        state_diff = (self.curr_state - self.goal_state)
        # normalize theta1, theta2 to [-pi, pi]
        state_diff[1] = np.mod(state_diff[1]+np.pi, 2*np.pi) - np.pi
        state_diff[2] = np.mod(state_diff[2]+np.pi, 2*np.pi) - np.pi
        return np.linalg.norm(np.matmul(self.error_Q, state_diff))
