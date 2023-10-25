import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from params import *

def idp_dynamics_analytic_numpy(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Numpy
    Args:
        state: np.array of shape (6, ) representing the cartpole state
        control: np.array of shape (1, ) representing the force to apply
    Returns:
        next_state: np.array of shape (6, ) representing the next cartpole state
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

    D = np.array([[d1,                d2*np.cos(theta1),        d3*np.cos(theta2)       ],
                  [d2*np.cos(theta1), d4,                       d5*np.cos(theta1-theta2)],
                  [d3*np.cos(theta2), d5*np.cos(theta1-theta2), d6                      ]]).astype(np.float64)
    C = np.array([[0, -d2*np.sin(theta1)*theta1_dot,        -d3*np.sin(theta2)*theta2_dot      ], 
                  [0, 0,                                    d5*np.sin(theta1-theta2)*theta2_dot], 
                  [0, -d5*np.sin(theta1-theta2)*theta1_dot, 0                                  ]]).astype(np.float64)
    G = np.array([0, -g1*np.sin(theta1), -g2*np.sin(theta2)]).astype(np.float64)
    H = np.array([1, 0, 0]).astype(np.float64)

    state_dot = state[3:] # [theta0_dot, theta1_dot, theta2_dot]
    D_inv = np.linalg.pinv(D)
    state_dot_dot = -D_inv @ C @ state_dot - D_inv @ G + D_inv @ (H * action[0])
    [theta0_dot2, theta1_dot2, theta2_dot2] = state_dot_dot

    theta0_next_dot = theta0_dot + dt*theta0_dot2
    theta1_next_dot = theta1_dot + dt*theta1_dot2
    theta2_next_dot = theta2_dot + dt*theta2_dot2

    theta0_next = theta0 + dt*theta0_next_dot
    theta1_next = theta1 + dt*theta1_next_dot
    theta2_next = theta2 + dt*theta2_next_dot

    next_state = np.array([theta0_next, theta1_next, theta2_next, theta0_next_dot, theta1_next_dot, theta2_next_dot])
    return next_state


def running_cost(state, action):
    '''
    :param state: np array of shape (state_dim, )
    :param action: np array of shape (action_dim, )
    :param goal_state: np array of shape (state_dim, )

    :returns cost: np array of shape (1, )
    '''
    goal_state = np.array([0., 0., 0., 0., 0., 0.])
    Q = np.diag([0.5, 10., 10., 1., 1., 1.])
    R = np.array([[0.01]])
    diff = state - goal_state
    # normalize theta1, theta2 to [-pi, pi]
    # autograd doesn't support assignment to array
    [theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot] = diff
    theta1 = np.mod(theta1+np.pi, 2*np.pi) - np.pi
    theta2 = np.mod(theta2+np.pi, 2*np.pi) - np.pi
    diff = np.array([theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot])
    cost = diff.T @ Q @ diff + action.T @ R @ action
    return cost


def terminal_cost(state):
    '''
    :param state: np array of shape (state_dim, )
    :param goal_state: np array of shape (state_dim, )

    :returns cost: np array of shape (1, )
    '''
    goal_state = np.array([0., 0., 0., 0., 0., 0.])
    Q = np.diag([1, 100, 100, 1, 1, 1])
    diff = state - goal_state
    # normalize theta1, theta2 to [-pi, pi]
    # autograd doesn't support assignment to array
    [theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot] = diff
    theta1 = np.mod(theta1+np.pi, 2*np.pi) - np.pi
    theta2 = np.mod(theta2+np.pi, 2*np.pi) - np.pi
    diff = np.array([theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot])
    cost = diff.T @ Q @ diff
    return cost

def compute_cost(states, actions):
    '''
    :param states: np array of shape (T, state_dim)
    :param actions: np array of shape (T-1, action_dim)
    :param goal_state: np array of shape (state_dim, )

    :returns cost: integer
    '''
    cost = 0.0
    for i in range(actions.shape[0]):
        cost += running_cost(states[i], actions[i])
    cost += terminal_cost(states[-1])
    return cost

def plot_idp(state, i):
    '''for numpy inputs'''
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

    pendulum1_x = cart_x + L1*np.sin(theta1)
    pendulum1_y = cart_y + L1*np.cos(theta1)
    ax.plot((cart_x, pendulum1_x), (cart_y, pendulum1_y), 
            color='red', linestyle='-', linewidth=4)

    pendulum2_x = pendulum1_x + L2*np.sin(theta2)
    pendulum2_y = pendulum1_y + L2*np.cos(theta2)
    ax.plot((pendulum1_x, pendulum2_x), (pendulum1_y, pendulum2_y), 
            color='black', linestyle='-', linewidth=4)

    ax.set_title('DDP Controller, Time Elapsed: {:.2f}'.format(i*dt))
    # This will produce a name like 'plot_0000.png'
    filename = 'results/ddp/idp_ddp_{:04d}.png'.format(i)  
    fig.savefig(filename)
    plt.close()