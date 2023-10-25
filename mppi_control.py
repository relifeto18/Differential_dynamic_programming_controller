from mppi import *
import numpy as np
import matplotlib.pyplot as plt

from mppi_utils import idp_dynamics_analytic_torch_batch, idp_dynamics_analytic_torch, plot_idp, swing_up_costs

class InvertedDoublePendulumMPPIController(object):
    def __init__(self, start_state, goal_state, hyperparams):
        self.start_state = start_state # torch tensor
        self.curr_state = start_state # torch tensor
        self.goal_state = goal_state # torch tensor
        self.state_dim = 6
        self.action_dim = 1
        self.counter = 0
        
        # tune these hyperparameters
        self.swing_up_Q = hyperparams['swing_up_Q']
        self.horizontal_shift_Q = hyperparams['horizontal_shift_Q']
        self.error_Q = hyperparams['error_Q']
        noise_sigma = hyperparams['noise_sigma']
        lambda_value = hyperparams['lambda_value']
        num_samples = hyperparams['num_samples']
        horizon = hyperparams['horizon']
        # u_min = hyperparams['u_min']
        # u_max = hyperparams['u_max']
        # tune these hyperparameters

        self.mppi = MPPI(self._compute_dynamics,
                         self._compute_costs,
                         nx=self.state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value)
                        #  u_min=u_min,
                        #  u_max=u_max)

    def _compute_dynamics(self, state, action):
        '''return next_state based on the analytical dynamics'''
        # print('state.shape', state.shape)
        # print('action.shape', action.shape)
        return idp_dynamics_analytic_torch_batch(state, action) # torch tensor
    
    def _compute_costs(self, state, action):
        return swing_up_costs(state, action, self.goal_state, self.swing_up_Q)

    def control(self):
        '''run mppi to return next_action'''
        # print("self.curr_state.shape", self.curr_state.shape)
        action = self.mppi.command(self.curr_state)
        # print("action.shape", action.shape)
        self.curr_state = idp_dynamics_analytic_torch(self.curr_state, action) # torch tensor
        self.counter += 1

    def render(self):
        """render and save an image of IDP at current state"""
        plot_idp(self.curr_state, self.counter)

    def calculate_error(self):
        '''calculate error between current state and goal state'''
        state_diff = (self.curr_state - self.goal_state)
        return torch.linalg.norm(torch.matmul(self.error_Q, state_diff))