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
    # also resturns a position array of shape (N, H, init_pos.size)

    states = [init_state]
    pos = [init_pos]
    system = model(init_state, dt=dt)
    state_size = system.state_size
    control_input_size = system.control_input_size
    for a in control_inputs:
        state, position = system.forward(a.flatten())
        states.append(state)
        pos.append(position)

    pos_array = np.array(pos).reshape((len(pos), init_pos.shape[0]))
    states_array = np.array(states).reshape((len(states), state_size))
    return states_array, pos_array


def generate_init_traj_quad(init_pos, goal_pos, H, Tf=1, input_limits=QUAD_INPUT_LIMITS):
    # returns an array of shape (H, m), m state size, H = num_timesteps

    # Define the trajectory starting state:
    if init_pos.size == 6:
        pos0 = init_pos[0:3]  #position
        vel0 = init_pos[3:6]  #velocity
    else:
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
# SIMPLE DYNAMICS
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
# MORE COMPLEX DYNAMICS
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
