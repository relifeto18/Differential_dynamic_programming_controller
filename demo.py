import torch
import autograd.numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio

from mppi_control import InvertedDoublePendulumMPPIController
from ddp_control import InvertedDoublePendulumDDPController

state_dim = 6
action_dim = 1

# erase all previous idp_mppi results
image_dir = 'results/mppi'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if os.path.isfile(os.path.join(image_dir, f)) and f.endswith('.png')]
for file in image_files:
    try:
        os.remove(file)
    except:
        pass

# adjust these parameters
idp_mppi_hyperparams = {'swing_up_Q': torch.diag(torch.tensor([0.5, 10., 10., 1., 1., 1.])),
                        'horizontal_shift_Q': torch.diag(torch.tensor([0.01, 1, 1, 0.1, 0.1, 0.1])),
                        'error_Q': torch.diag(torch.tensor([0.05, 1, 1, 0.1, 0.1, 0.1])),
                        'noise_sigma': torch.eye(action_dim) * 60,
                        'lambda_value': 0.08,
                        'num_samples': 300,
                        'horizon': 20
                        }
# adjust these parameters

# consider different configuations
start_state = torch.tensor([0., np.pi, np.pi, 0., 0., 0.]) # fully down
goal_state = torch.tensor([0., 0., 0., 0., 0., 0.])
idp_controller = InvertedDoublePendulumMPPIController(start_state, goal_state, idp_mppi_hyperparams)

# adjust these parameters
num_steps = 150
error_threshold = 0.25
# adjust these parameters

# mppi controller
print("Starting MPPI control")
lowest_error = None
pbar = tqdm(range(num_steps))
idp_controller.render() # render the image of starting state
for i in pbar:
    idp_controller.control()
    idp_controller.render()
    error_i = idp_controller.calculate_error()
    if lowest_error is None or error_i < lowest_error:
        lowest_error = error_i
    pbar.set_description(f'Goal Error: {error_i:.4f}, Lowest Error: {lowest_error:.4f}')
    if error_i < error_threshold:
        print("Break")
        break

# save gif 
print("creating animated gif, please wait about 10 seconds")
frames = []
image_dir = 'results/mppi'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith('.png')]
image_files.sort()
frames = []
for file in image_files:
    frames.append(imageio.imread(file))
imageio.mimsave('results/idp_mppi.gif', frames, format='GIF', duration=0.05)
print("Finished!\n")


# DDP --------------------

# erase all previous idp_ddp results
image_dir = 'results/ddp'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if os.path.isfile(os.path.join(image_dir, f)) and f.endswith('.png')]
for file in image_files:
    try:
        os.remove(file)
    except:
        pass
     
# adjust these parameters
idp_ddp_hyperparams = {'epsilon': 1e-3,
                       'max_iters': 100,
                       'horizon': 20,
                       'backtrack_max_iters': 10,
                       'decay': 0.5,
                       'error_Q': np.diag([0.05, 1, 1, 0.1, 0.1, 0.1])
                       }
# adjust these parameters

# consider different configuations
start_state = np.array([0., np.pi, np.pi, 0., 0., 0.]) # fully down
goal_state = np.array([0., 0., 0., 0., 0., 0.])
idp_controller = InvertedDoublePendulumDDPController(start_state, goal_state, idp_ddp_hyperparams)

# adjust these parameters
num_steps = 150
error_threshold = 0.25
# adjust these parameters

# ddp controller
print("Starting DDP control")
lowest_error = None
pbar = tqdm(range(num_steps))
idp_controller.render() # render the image of starting state
for i in pbar:
    idp_controller.control()
    idp_controller.render()
    error_i = idp_controller.calculate_error()
    if lowest_error is None or error_i < lowest_error:
        lowest_error = error_i
    pbar.set_description(f'Goal Error: {error_i:.4f}, Lowest Error: {lowest_error:.4f}')
    if error_i < error_threshold:
        print("Break")
        break

# save gif 
print("creating animated gif, please wait about 10 seconds")
frames = []
image_dir = 'results/ddp'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith('.png')]
image_files.sort()
frames = []
for file in image_files:
    frames.append(imageio.imread(file))
imageio.mimsave('results/idp_ddp.gif', frames, format='GIF', duration=0.05)
print("Finished!\n")

# erase all previous idp_mppi results
image_dir = 'results/mppi'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if os.path.isfile(os.path.join(image_dir, f)) and f.endswith('.png')]
for file in image_files:
    try:
        os.remove(file)
    except:
        pass

# erase all previous idp_ddp results
image_dir = 'results/ddp'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if os.path.isfile(os.path.join(image_dir, f)) and f.endswith('.png')]
for file in image_files:
    try:
        os.remove(file)
    except:
        pass