import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from params import *

def idp_dynamics_analytic_torch_batch(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
    Args:
        state: torch.tensor of shape (B, 6) representing the cartpole state
        control: torch.tensor of shape (B, 1) representing the force to apply
    Returns:
        next_state: torch.tensor of shape (B, 6) representing the next cartpole state
    """
    B = state.shape[0]
    next_states = []
    for i in range(B):
        next_states.append(idp_dynamics_analytic_torch(state[i], action[i]))
    next_states = torch.stack(next_states, dim=0)
    return next_states

def idp_dynamics_analytic_torch(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
    Args:
        state: torch.tensor of shape (6, ) representing the cartpole state
        control: torch.tensor of shape (1, ) representing the force to apply
    Returns:
        next_state: torch.tensor of shape (6, ) representing the next cartpole state
    """

    [theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot] = state
    d1 = m0 + m1 + m2
    d2 = m1*l1 + m2*L1
    d3 = m2*l2
    d4 = m1*l1*l1 + m2*L1*L1 + I1
    d5 = m2*L1*l2
    d6 = m2*l2*l2 + I2
    g1 = (m1*l1 + m2*L1)*g
    g2 = m2*g*l2

    D = torch.tensor([[d1,                   d2*torch.cos(theta1),        d3*torch.cos(theta2)       ],
                      [d2*torch.cos(theta1), d4,                          d5*torch.cos(theta1-theta2)],
                      [d3*torch.cos(theta2), d5*torch.cos(theta1-theta2), d6                         ]]).to(torch.float32)
    C = torch.tensor([[0, -d2*torch.sin(theta1)*theta1_dot,        -d3*torch.sin(theta2)*theta2_dot      ], 
                      [0, 0,                                       d5*torch.sin(theta1-theta2)*theta2_dot], 
                      [0, -d5*torch.sin(theta1-theta2)*theta1_dot, 0                                     ]]).to(torch.float32)
    G = torch.tensor([0, -g1*torch.sin(theta1), -g2*torch.sin(theta2)]).to(torch.float32)
    H = torch.tensor([1, 0, 0]).to(torch.float32)

    state_dot = state[3:] # [theta0_dot, theta1_dot, theta2_dot]
    D_inv = torch.linalg.pinv(D)
    state_dot_dot = -D_inv@C@state_dot - D_inv@G + D_inv@(H*action[0])
    [theta0_dot2, theta1_dot2, theta2_dot2] = state_dot_dot

    theta0_next_dot = theta0_dot + dt*theta0_dot2
    theta1_next_dot = theta1_dot + dt*theta1_dot2
    theta2_next_dot = theta2_dot + dt*theta2_dot2

    theta0_next = theta0 + dt*theta0_next_dot
    theta1_next = theta1 + dt*theta1_next_dot
    theta2_next = theta2 + dt*theta2_next_dot

    next_state = torch.tensor([theta0_next, theta1_next, theta2_next, theta0_next_dot, theta1_next_dot, theta2_next_dot])
    return next_state


def horizontal_shift_costs(state, action, target_state, Q):
    pass


def swing_up_costs(state, action, target_state, Q):
    """
    Compute the state cost for MPPI.
    state_dim=6, action_dim=1
    :param state: torch tensor of shape (B, state_dim)
    :param action: torch tensor of shape (B, action_dim )
    :param target_state: torch tensor of shape (state_dim,)
    :param Q: torch tensor of shape (state_dim, state_dim)

    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    state_diffs = (state - target_state).unsqueeze(dim=1) # shape: (B, 1, 6)
    cost = torch.matmul(torch.matmul(state_diffs, Q), torch.permute(state_diffs, dims=(0,2,1))) # shape: (B,1,1)
    cost = cost.squeeze() # shape: (B,)
    return cost

def running_cost(state, action, target_state, Q, R):
    error = state - target_state
    result = error.T @ Q @ error + action.T @ R @ action
    return result
        
def terminal_cost(state, target_state, Q):
    error = state - target_state
    result = error.T @ Q @ error
    return result

def plot_idp(state, i):
    [theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot] = state

    fig, ax = plt.subplots()
    cart_x = theta0
    cart_y = 0
    img_w = 10

    ax.set_xlim([cart_x - img_w/2, cart_x + img_w/2])
    ax.set_ylim([-(L1+L2+cart_h), (L1+L2+cart_h)])
    ax.set_aspect('equal')

    slidebar = Rectangle((cart_x - img_w/2, cart_y - slidebar_h/2), 
                         img_w, slidebar_h, linewidth=1, edgecolor='g', facecolor='g')
    ax.add_patch(slidebar)

    cart = Rectangle((cart_x - cart_w/2, cart_y - cart_h/2), 
                     cart_w, cart_h, linewidth=1, edgecolor='b', facecolor='b')
    ax.add_patch(cart)

    pendulum1_x = cart_x + L1*torch.sin(theta1)
    pendulum1_y = cart_y + L1*torch.cos(theta1)
    ax.plot((cart_x, pendulum1_x), (cart_y, pendulum1_y), 
            color='red', linestyle='-', linewidth=4)

    pendulum2_x = pendulum1_x + L2*torch.sin(theta2)
    pendulum2_y = pendulum1_y + L2*torch.cos(theta2)
    ax.plot((pendulum1_x, pendulum2_x), (pendulum1_y, pendulum2_y), 
            color='black', linestyle='-', linewidth=4)

    ax.set_title('MPPI Controller, Time Elapsed: {:.2f}'.format(i*dt))
    # This will produce a name like 'plot_0000.png'
    filename = 'results/mppi/idp_mppi_{:04d}.png'.format(i)  
    fig.savefig(filename)
    plt.close()