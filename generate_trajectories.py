import numpy as np


# TODO: REFACTOR TO CLASS
# TODO: use model from unmesh email

# def generate_agent_control_inputs(num_timesteps, model='constant_accel'):
#     # should return an array of shape (N, H, m), for m-sized control input, H = num_timesteps, and N = num_agents
#     if model == 'constant_accel':
#         control_inputs = []
#         for i in range(num_timesteps):
#             control_inputs.append(np.random.uniform(low=-1, high=1))
    
#         return np.array(control_inputs)
    
#     elif model == 'simple':
#         control_inputs = []
#         for i in range(num_timesteps):
#             control_inputs.append(np.random.uniform(low=-1, high=1, size=2))
    
#         return np.array(control_inputs)

def generate_agent_states(control_inputs, init_state, model):
    # should return an array of shape (N, H, m), for m state size, H = num_timesteps, and N = num_agents

    states = [init_state]
    system = model(init_state)
    state_size = system.state_size
    control_input_size = system.control_input_size
    for a in control_inputs:
        state, position = system.forward(a.reshape((control_input_size, 1)))
        states.append(state)

    return np.array(states).reshape((len(states), state_size))

# def create_u_vector(control_inputs, model='constant_accel'):
#     # create a 1D vector of size m * H * N, for m-sized control input, H = num_timesteps, and N = num_agents
#     if model == 'constant_accel':
#         N, H = control_inputs.shape
#         control_inputs = control_inputs.reshape((1, H, N))

#     elif model == 'simple':
#         N, H = control_inputs.shape
#         control_inputs = control_inputs.reshape((2, H, N))
    
#     m, H, N = control_inputs.shape
#     return control_inputs.reshape(m*H*N)

# def create_x_vector(states):
#     # create a 1D vector of size m * H * N, for m state size, H = num_timesteps, and N = num_agents
#     N, H, m = states.shape
#     states = states.reshape((m, H, N))
#     return states.reshape(m*H*N)

# def generate_trajectories(num_agents, num_timesteps, model):
#     control_inputs = []
#     for n in range(num_agents):
#         control_inputs.append(generate_agent_control_inputs(num_timesteps, model=model))

#     states = []
#     for input_set in control_inputs:
#         states.append(generate_agent_states(input_set, model=model))

#     init_u = create_u_vector(np.array(control_inputs), model=model)
#     init_x = create_x_vector(np.array(states))
#     # init_u = control_inputs
#     # init_x = states

#     return init_u, init_x


##############################################################################
# DEFINE SYSTEM DYNAMICS FOR BODY INCREASING ALTITUDE AT CONSTANT ACCELERATION
# https://www.kalmanfilter.net/multiExamples.html
##############################################################################
class SystemConstantAccel():
    def __init__(self, init_state, dt=0.1):
        """
        init_state: 2d array [altitude, velocity]
        """
        self.control_input_size = 1
        self.state_size = 2
        self.state = init_state
        self.dt = dt

        self.F = np.array([[1, dt], [0, 1]])
        self.G = np.array([[0.5 * dt**2], [dt]])
        self.g = 9.8

    def forward(self, accel):
        new_state = np.dot(self.F, self.state) + self.G * (accel + self.g)
        self.state = new_state
        position = new_state[0]
        return new_state, position


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
        new_state = self.state + 2 * u 
        position = new_state
        self.state = new_state
        
        return new_state, position

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