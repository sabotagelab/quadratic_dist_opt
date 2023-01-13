import argparse
import matplotlib.pyplot as plt
import numpy as np

from objective import Objective
from generate_trajectories import SystemSimple
from generate_trajectories import generate_trajectories, generate_agent_states, create_x_vector

if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Centralized Optimization')
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--kappa', type=float, default=1)
    parser.add_argument('--eps_bounds', type=float, default=.1)
    parser.add_argument('--r', type=float, default=0.5)
    parser.add_argument('--c', type=float, default=3)
    parser.add_argument('--Ubox', type=float, default=1)
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

    # GET INITIAL TRAEJECTORIES FROM CONTROL INPUTS
    init_trajectories = []
    fix, ax = plt.subplots()
    for i in range(N):
        traj = generate_agent_states(init_u[i], model='simple')
        ax.scatter(traj[:,0], traj[:,1], label=i)
        init_trajectories.append(traj)
    circle = plt.Circle(c, r, fill=False)
    ax.add_patch(circle)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend()
    plt.savefig('agent_init_trajectories.png')
    plt.clf()
     
    # INIT SOLVER
    Q = np.eye(N*H*control_input_size)   # variable for quadratic objective
    # n = N*H*control_input_size
    # Q = np.random.randn(n, n)   # variable for quadratic objective
    # Q = Q.T @ Q
    obj = Objective(N, H, system_model, init_states, obstacles, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox)

    # METRICS FOR INITIAL TRAJECTORY
    print('Initial Obj {}'.format(obj.central_obj(init_u.flatten())))
    print('Initial Total Energy Cost (Lower is better)')
    print(obj.quad(init_u.flatten()))
    print('Initial Fairness (Close to 0 is better)')
    print(obj.fairness(init_u.flatten()))
    print('Initial Obstacle Avoidance Cost (More Negative Is Better)')
    print(obj.obstacle(init_u.flatten()))

    # SOLVE USING CENTRAL
    final_obj, final_u = obj.solve_central(init_u)
    print('Central Final Obj {}'.format(final_obj))
    print('Solved Inputs')
    final_u = final_u.reshape(N, H, control_input_size)
    print(final_u)

    # METRICS FOR FINAL TRAJECTORY AFTER SOLVING CENTRAL PROBLEM
    print('Final Total Energy Cost (Lower is better)')
    print(obj.quad(final_u.flatten()))
    print('Final Fairness (Close to 0 is better)')
    print(obj.fairness(final_u.flatten()))
    print('Final Obstacle Avoidance Cost (More Negative Is Better)')
    print(obj.obstacle(final_u.flatten()))

    # PLOT FINAL TRAEJECTORIES FROM CONTROL INPUTS
    final_trajectories = []
    fix, ax = plt.subplots()
    for i in range(N):
        traj = generate_agent_states(final_u[i], model='simple')
        ax.scatter(traj[:,0], traj[:,1], label=i)
        final_trajectories.append(traj)
    circle = plt.Circle(c, r, fill=False)
    ax.add_patch(circle)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend()
    plt.savefig('agent_final_trajectories_central.png')
    plt.clf()


    # SOLVE USING DISTRIBUTED OPTIMIZATION
    final_u, local_sols = obj.solve_distributed(init_u, steps=200)
    print('Distributed Final Obj {}'.format(obj.central_obj(final_u.flatten())))
    print(obj.central_obj(final_u.flatten()))
    print('Solved Inputs')
    print(final_u)

    # METRICS FOR FINAL TRAJECTORY AFTER SOLVING DISTRIBUTED PROBLEM
    print('Final Total Energy Cost (Lower is better)')
    print(obj.quad(final_u.flatten()))
    print('Final Fairness (Close to 0 is better)')
    print(obj.fairness(final_u.flatten()))
    print('Final Obstacle Avoidance Cost (More Negative Is Better)')
    print(obj.obstacle(final_u.flatten()))

    # PLOT FINAL TRAEJECTORIES FROM CONTROL INPUTS
    final_trajectories = []
    fix, ax = plt.subplots()
    for i in range(N):
        traj = generate_agent_states(final_u[i], model='simple')
        ax.scatter(traj[:,0], traj[:,1], label=i)
        final_trajectories.append(traj)
    # print(final_trajectories)
    circle = plt.Circle(c, r, fill=False)
    ax.add_patch(circle)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend()
    plt.savefig('agent_final_trajectories_dist.png')
    plt.clf()
    
    for i in range(N):
        plt.plot(local_sols[i])
    plt.savefig('local_objective.png')
