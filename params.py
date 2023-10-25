# parameters of inverted double pendulum on a cart environment
dt = 0.05 # time interval
g = 9.81 # gravity constatnt
m0 = 1.0 # weight of the cart
m1 = 0.1 # weight of 1st pendulum
m2 = 0.1 # weight of 2nd pendulum
L1 = 1.0 # length of 1st pendulum
L2 = 1.0 # length of 2nd pendulum
l1 = L1/2 # length of 1st pendulum's mass center
l2 = L2/2 # length of 2nd pendulum's mass center
I1 = m1*L1*L1/12 # moment of inertia of 1st pendulum
I2 = m2*L2*L2/12 # moment of inertia of 2nd pendulum

cart_w = 1 # length of the cart
cart_h = 0.5 # height of the cart
pendulum_w = 0.1 # width of the pendulum
slidebar_h = 0.1 # height of the slidebar