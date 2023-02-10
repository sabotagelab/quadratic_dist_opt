import numpy as np
from trajectorygenerationcodeandreas import quadrocoptertrajectory as quadtraj

QUAD_INPUT_LIMITS = {
    'fmin': 5,  #[m/s**2]
    'fmax': 25, #[m/s**2]
    'wmax': 20, #[rad/s]
    'minTimeSec': 0.02 #[s]
}


def generate_agent_states(control_inputs, init_state, init_pos, model, dt=0.1):
    # should return an array of shape (N, H, m), for m state size, H = num_timesteps, and N = num_agents

    states = [init_state]
    pos = [init_pos]
    system = model(init_state, dt=dt)
    state_size = system.state_size
    control_input_size = system.control_input_size
    for a in control_inputs:
        # print(a.flatten())
        state, position = system.forward(a.flatten())
        states.append(state)
        pos.append(position)

    # print(pos)
    pos_array = np.array(pos).reshape((len(pos), init_pos.shape[0]))
    states_array = np.array(states).reshape((len(states), state_size))
    return states_array, pos_array


def generate_init_traj_quad(init_pos, goal_pos, H, Tf=1, input_limits=QUAD_INPUT_LIMITS):
    # returns an array of shape (H, m), m state size, H = num_timesteps

    # Define the trajectory starting state:
    pos0 = init_pos  #position
    vel0 = [0, 0, 0] #velocity
    acc0 = [0, 0, 0] #acceleration

    # Define the goal state:
    posf = goal_pos  # position
    # velf = [0, 0, 1]  # velocity
    velf = [0, 0, 0]  # velocity
    # accf = [0, 9.81, 0]  # acceleration
    accf = [0, 0, 0]  # acceleration

    # Define the input limits:
    fmin = input_limits['fmin']             #[m/s**2]
    fmax = input_limits['fmax']             #[m/s**2]
    wmax = input_limits['wmax']             #[rad/s]
    minTimeSec = input_limits['minTimeSec'] #[s]

    # Define how gravity lies:
    gravity = [0,0,-9.81]

    # Generate the trjaectory
    traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, gravity)
    traj.set_goal_position(posf)
    traj.set_goal_velocity(velf)
    traj.set_goal_acceleration(accf)
    traj.generate(Tf)

    # Test input feasibility
    inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, minTimeSec)

    time = np.linspace(0, Tf, H)
    # print(time)

    traj_pos = []
    traj_coeffs = []
    traj_accels = []  # the inputs
    for t in time:
        traj_pos.append(traj.get_position(t))
        traj_accels.append(traj.get_acceleration(t))

    return np.array(traj_pos), np.array(traj_accels)




##############################################################################
# SIMPLE STUPID DYNAMICS
##############################################################################
class SystemSimple():
    def __init__(self, init_state, dt=0.1):
        """
        init_state: 2d array
        """
        self.control_input_size = 2
        self.state_size = 2
        self.state = init_state
        self.dt = dt

    def forward(self, u):
        new_state = self.state + 2 * u.reshape((2, 1))
        position = new_state
        self.state = new_state
        
        return new_state, position

##############################################################################
# MORE COMPLEX DYNAMICS FROM ANDREAS CODE
##############################################################################
class Quadrocopter():
    def __init__(self, init_state, dt=0.1):
        """
        init_state: 6d array (x,y,z position, velocity)
        """
        self.control_input_size = 3
        self.state_size = 6
        self.state = init_state
        self.dt = dt

    def forward(self, u):
        t = self.dt
        pos = self.state[0:3]
        velo = self.state[3:6]
        accel = u

        new_velo = velo + accel*t 

        new_pos = pos + velo*t + (1.0/2.0)*accel*(t**2)

        new_state = np.array([new_pos, new_velo]).flatten()
        self.state = new_state

        return new_state, new_pos

        
# init_state = np.array([[0], [0]])
# states = [init_state]
# system = SystemConstantAccel(init_state)

# # control_inputs = generate_agent_control_inputs(10)
# control_inputs = generate_agent_control_inputs(10, model='simple')
# print(control_inputs)
# # system_states = generate_agent_states(control_inputs, model='constant_accel')
# system_states = generate_agent_states(control_inputs, model='simple')
# print(system_states)

# init_u, init_x = generate_trajectories(3, 10, model='simple')
# print(np.array(init_u).shape)
# print(np.array(init_x).shape)

# USING ANDREAS DYNAMICS
# Define the trajectory starting state:
# pos0 = [0, 0, 2] #position
# vel0 = [0, 0, 0] #velocity
# acc0 = [0, 0, 0] #acceleration

# # Define the goal state:
# posf = [1, 0, 1]  # position
# velf = [0, 0, 1]  # velocity
# accf = [0, 9.81, 0]  # acceleration

# # Define the duration:
# Tf = 1

# # Define the input limits:
# fmin = 5  #[m/s**2]
# fmax = 25 #[m/s**2]
# wmax = 20 #[rad/s]
# minTimeSec = 0.02 #[s]

# # Define how gravity lies:
# gravity = [0,0,-9.81]

# traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, gravity)
# traj.set_goal_position(posf)
# traj.set_goal_velocity(velf)
# traj.set_goal_acceleration(accf)
# traj.generate(Tf)

# # Test input feasibility
# inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, minTimeSec)
# # print("x Alpha {}".format(traj._axis[0]._a))
# # print("y Alpha {}".format(traj._axis[1]._a))
# # print("z Alpha {}".format(traj._axis[2]._a))
# # print("x Beta {}".format(traj._axis[0]._b))
# # print("y Beta {}".format(traj._axis[1]._b))
# # print("z Beta {}".format(traj._axis[2]._b))
# # print("x Gamma {}".format(traj._axis[0]._g))
# # print("y Gamma {}".format(traj._axis[1]._g))
# # print("z Gamma {}".format(traj._axis[2]._g))

# init_coeffs = np.array([
#     traj._axis[0]._a,
#     traj._axis[1]._a,
#     traj._axis[2]._a,
#     traj._axis[0]._b,
#     traj._axis[1]._b,
#     traj._axis[2]._b,
#     traj._axis[0]._g,
#     traj._axis[1]._g,
#     traj._axis[2]._g,
# ])


# init_state = np.array([pos0, vel0, acc0]).flatten()
# print(init_state)
# system = Quadrocopter(init_state, dt=0.25)
# init_coeffs_reshape = init_coeffs.reshape((3, 3))
# new_state, new_pos = system.forward(init_coeffs_reshape)
# print(new_state)
# print(new_pos)
# print(system.state)

# numPlotPoints = 5  # THIS IS THE SAME AS H
# time = np.linspace(0, Tf, numPlotPoints)
# print(time)
# print(traj.get_position(time[1]))


# new_state, new_pos = system.forward(init_coeffs_reshape + np.random.random_sample(init_coeffs_reshape.shape))
# print(new_state)
# print(new_pos)
# print(system.state)

# numPlotPoints = 5  # THIS IS THE SAME AS H
# time = np.linspace(0, Tf, numPlotPoints)
# print(time)
# print(traj.get_position(time[2]))

# init_traj, init_coeffs = generate_init_traj_quad([0, 0, 2], [1, 0, 1], 5)
# print(init_traj.shape)
# print(init_coeffs.shape)