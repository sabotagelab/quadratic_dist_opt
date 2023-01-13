import argparse
# import matplotlib.pyplot as plt
import numpy as np

from objective import Objective
from generate_trajectories import SystemSimple
from generate_trajectories import generate_trajectories, generate_agent_states, create_x_vector

if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Centralized Optimization')
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--kappa', type=float, default=1)
    parser.add_argument('--eps_bounds', type=float, default=.1)
    parser.add_argument('--r', type=float, default=5)
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument('--Ubox', type=float, default=10)
    parser.add_argument('--iter', type=int, default=1000)

    args = parser.parse_args()
    print(args)

    N = args.N  # number of agents
    alpha = args.alpha   # parameter for fairness constraint
    beta = args.beta    # parameter for weighting of obstacle avoidance constraint
    gamma = args.gamma   # parameter for smoothmin in calculating obstacle avoidance constraint
    kappa = args.kappa   # parameter for weighting change in epsilon for local problem
    eps_bounds = args.eps_bounds  # bounds for eps in an iteration
    r = args.r  # radius of circle
    c = np.array([args.c, args.c])  # center of circle
    obstacles = {'center': c, 'radius': r}
    Ubox = args.Ubox  # box constraint
    
    H = 3
    # GENERATE INITIAL STATES
    init_states = []
    for i in range(N):
        init_states.append(np.array([[0], [0]]))
    
    system_model = SystemSimple(init_states[0])
    control_input_size = system_model.control_input_size

    # GENERATE INITIAL CONTROL INPUTS
    init_u = np.random.uniform(low=-1, high=1, size=(N, H, control_input_size))
    print('Initial Inputs')
    print(init_u)
    
    # SOLVE USING CENTRAL
    Q = np.eye(N*H*control_input_size)   # variable for quadratic objective
    obj = Objective(N, H, system_model, init_states, obstacles, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox)
    final_obj, final_u = obj.solve_central(init_u)
    print('Central Final Obj {}'.format(final_obj))
    print('Solved Inputs')
    print(final_u.reshape(N, H, control_input_size))
    

    # SOLVE USING DISTRIBUTED SA
    print('Distributed Final Obj')
    final_u = obj.solve_distributed(init_u)
    print(obj.central_obj(final_u.flatten()))
    print('Solved Inputs')
    print(final_u)
    
