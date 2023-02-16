import argparse
import matplotlib.pyplot as plt
import numpy as np

from objective import Objective
from generate_trajectories import SystemSimple
from generate_trajectories import generate_agent_states

from trajectorygenerationcodeandreas import quadrocoptertrajectory as quadtraj


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Centralized Optimization')
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--kappa', type=float, default=.1)
    parser.add_argument('--eps_bounds', type=float, default=.1)
    
    parser.add_argument('--r', type=float, default=0.5)
    parser.add_argument('--c', type=float, default=3)
    parser.add_argument('--Ubox', type=float, default=1)
    parser.add_argument('--iter', type=int, default=100)

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
    rg = r
    cg = np.array([5, 5])
    obstacles = {'center': c, 'radius': r}
    target = {'center': np.array([5, 5]), 'radius': r}
    Ubox = args.Ubox  # box constraint
    
    H = 5
    # GENERATE INITIAL STATES
    # init_states = []
    # for i in range(N):
    #     init_states.append(np.array([[0], [0]]))
    init_states = [np.array([[-3], [3]]), np.array([[-3], [-3]]), np.array([[3], [-3]])]
    
    #system_model = SystemSimple(init_states[0])
    # control_input_size = system_model.control_input_size
    system_model = SystemSimple
    control_input_size = 2
    system_model_config = (SystemSimple, control_input_size)

    # GENERATE INITIAL CONTROL INPUTS
    # init_u = np.random.uniform(low=-1, high=1, size=(N, H, control_input_size))
    init_u1 = np.array([[1, 0], [.75, 1], [.75, 0], [.75, 0], [.5, 0]])
    init_u2 = np.array([[1, 1], [.75, 1], [.75, 1], [.5, 1], [1, 0]])
    init_u3 = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, .8]])
    init_u = np.array([init_u1, init_u2, init_u3])
    init_u = init_u.reshape((N, H, control_input_size))

    # GET INITIAL TRAEJECTORIES FROM CONTROL INPUTS
    init_trajectories = []
    fix, ax = plt.subplots()
    for i in range(N):
        _, traj = generate_agent_states(init_u[i], init_states[i], init_states[i], model=SystemSimple)
        ax.scatter(traj[:,0], traj[:,1], label=i)
        init_trajectories.append(traj)
    obs = plt.Circle(c, r, fill=True, alpha=0.2, color='red')
    ax.add_patch(obs)
    goal = plt.Circle(cg, rg, fill=True, alpha=0.2, color='green')
    ax.add_patch(goal)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend()
    plt.savefig('plots/simple/agent_init_trajectories.png')
    plt.clf()

    solo_energies = []
    for i in range(N):
        n = 1*H*control_input_size
        Q = np.eye(n)
        obj = Objective(1, H, system_model_config, [init_states[i]], [init_states[i]], obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox)
        final_obj, final_u = obj.solve_central(init_u[i], steps=args.iter)
        init_solo_energy = obj.quad(final_u.flatten())
        solo_energies.append(init_solo_energy)


    # INIT SOLVER
    # Q = np.eye(N*H*control_input_size)   # variable for quadratic objective
    n = N*H*control_input_size
    Q = np.random.randn(n, n)   # variable for quadratic objective
    Q = Q.T @ Q
    obj = Objective(N, H, system_model_config, init_states, init_states, obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox)
    obj.solo_energies = solo_energies

    # METRICS FOR INITIAL TRAJECTORY
    init_obj = obj.quad(init_u.flatten())
    print('Initial Obj {}'.format(init_obj))
    print('Initial Total Energy Cost (Lower is better)')
    init_energy = obj.quad(init_u.flatten())
    print(init_energy)
    print('Initial Fairness (Close to 0 is better)')
    init_fairness = obj.fairness(init_u.flatten())
    print(init_fairness)
    print('Initial Obstacle Avoidance Cost (More Negative Is Better)')
    init_obstacle = obj.obstacle(init_u.flatten())
    print(init_obstacle)
    print('Initial Collision Avoidance Cost (More Negative Is Better)')
    init_collision = obj.avoid_constraint(init_u.flatten())
    print(init_collision)

    # SOLVE USING CENTRAL
    final_obj, final_u = obj.solve_central(init_u, steps=args.iter)
    
    # METRICS FOR FINAL TRAJECTORY AFTER SOLVING CENTRAL PROBLEM
    central_sol_obj = final_obj
    print('Central Final Obj {}'.format(central_sol_obj))
    final_u = final_u.reshape(N, H, control_input_size)    
    print('Central Final Total Energy Cost (Lower is better)')
    central_sol_energy = obj.quad(final_u.flatten())
    print(central_sol_energy)
    print('Central Final Fairness (Close to 0 is better)')
    central_sol_fairness = obj.fairness(final_u.flatten())
    print(central_sol_fairness)
    print('Central Final Obstacle Avoidance Cost (More Negative Is Better)')
    central_sol_obstacle = obj.obstacle(final_u.flatten())
    print(central_sol_obstacle)
    print('Central Final Collision Avoidance Cost (More Negative Is Better)')
    central_sol_collision = obj.avoid_constraint(final_u.flatten())
    print(central_sol_collision)

    # PLOT FINAL TRAEJECTORIES FROM CONTROL INPUTS
    final_trajectories = []
    fix, ax = plt.subplots()
    for i in range(N):
        _, traj = generate_agent_states(final_u[i], init_states[i], init_states[i], model=SystemSimple)
        ax.scatter(traj[:,0], traj[:,1], label=i)
        final_trajectories.append(traj)
    obs = plt.Circle(c, r, fill=True, alpha=0.2, color='red')
    ax.add_patch(obs)
    goal = plt.Circle(cg, rg, fill=True, alpha=0.2, color='green')
    ax.add_patch(goal)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend()
    plt.savefig('plots/simple/agent_final_trajectories_central.png')
    plt.clf()
    
    valid_sol = obj.check_avoid_constraints(final_u)
    print('Central: Valid Solution? All Agents Avoid Obstacle: {}'.format(valid_sol))


    # SOLVE USING DISTRIBUTED OPTIMIZATION
    final_u, local_sols, fairness = obj.solve_distributed(init_u, steps=args.iter)

    # METRICS FOR FINAL TRAJECTORY AFTER SOLVING DISTRIBUTED PROBLEM
    dist_sol_obj = obj.central_obj(final_u.flatten())
    print('Dist Final Obj {}'.format(dist_sol_obj))
    final_u = final_u.reshape(N, H, control_input_size)    
    print('Dist Final Total Energy Cost (Lower is better)')
    dist_sol_energy = obj.quad(final_u.flatten())
    print(dist_sol_energy)
    print('Dist Final Fairness (Close to 0 is better)')
    dist_sol_fairness = obj.fairness(final_u.flatten())
    print(dist_sol_fairness)
    print('Dist Final Obstacle Avoidance Cost (More Negative Is Better)')
    dist_sol_obstacle = obj.obstacle(final_u.flatten())
    print(dist_sol_obstacle)
    print('Dist Final Collision Avoidance Cost (More Negative Is Better)')
    dist_sol_collision = obj.avoid_constraint(final_u.flatten())
    print(dist_sol_collision)

    # PLOT FINAL TRAEJECTORIES FROM CONTROL INPUTS
    final_trajectories = []
    fix, ax = plt.subplots()
    for i in range(N):
        _, traj = generate_agent_states(final_u[i], init_states[i], init_states[i], model=system_model)
        ax.scatter(traj[:,0], traj[:,1], label=i)
        final_trajectories.append(traj)
    obs = plt.Circle(c, r, fill=True, alpha=0.2, color='red')
    ax.add_patch(obs)
    goal = plt.Circle(cg, rg, fill=True, alpha=0.2, color='green')
    ax.add_patch(goal)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend()
    plt.savefig('plots/simple/agent_final_trajectories_dist.png')
    plt.clf()

    valid_sol = obj.check_avoid_constraints(final_u)
    print('Distributed: Valid Solution? All Agents Avoid Obstacle: {}'.format(valid_sol))
    
    # Convergence Plot
    for i in range(N):
        plt.plot(local_sols[i], label='Agent {} Local Objective Value'.format(i))
    plt.legend()
    plt.savefig('plots/simple/local_objective_solution_convergence.png')
    plt.clf()

    # Fairness Plot
    plt.plot(fairness)
    plt.savefig('plots/simple/distributed_fairness_over_time.png')
    plt.clf()

    # Comparison Plots
    labels = ['J', 'Energy', 'Fairness', 'Collision']
    distributed_values = [dist_sol_obj, dist_sol_energy, dist_sol_fairness, dist_sol_obstacle+dist_sol_collision]
    central_values = [central_sol_obj, central_sol_energy, central_sol_fairness, central_sol_obstacle+central_sol_collision]
    init_values = [init_obj, init_energy, init_fairness, init_obstacle+init_collision]

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, distributed_values, width, label='Distributed')
    rects2 = ax.bar(x, central_values, width, label='Central')
    rects3 = ax.bar(x + width, init_values, width, label='Initial')

    # ax.set_ylabel('Scores')
    ax.set_title('Objective Values')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, labels=[f'{x:.1f}' for x in rects1.datavalues])
    ax.bar_label(rects2, padding=3, labels=[f'{x:.1f}' for x in rects2.datavalues])
    ax.bar_label(rects3, padding=3, labels=[f'{x:.1f}' for x in rects3.datavalues])

    fig.tight_layout()

    plt.savefig('plots/simple/objective_values_comparison.png')
    plt.clf()

    # Objective Comparison Plot
    plt.hlines(dist_sol_obj, xmin=0, xmax=args.iter, linestyles='solid', label='Distributed Objective Value')
    plt.hlines(central_sol_obj, xmin=0, xmax=args.iter, linestyles='dashed', label='Central Objective Value')
    plt.hlines(init_obj, xmin=0, xmax=args.iter, linestyles='dotted', label='Initial Objective Value')
    plt.legend()
    plt.savefig('plots/simple/final_objective_value_comparison.png')
    plt.clf()

    # Energy Comparison Plot
    plt.hlines(dist_sol_energy, xmin=0, xmax=args.iter, linestyles='solid', label='Distributed Objective Value')
    plt.hlines(central_sol_energy, xmin=0, xmax=args.iter, linestyles='dashed', label='Central Objective Value')
    plt.hlines(init_energy, xmin=0, xmax=args.iter, linestyles='dotted', label='Initial Objective Value')
    plt.legend()
    plt.savefig('plots/simple/final_energy_value_comparison.png')
    plt.clf()

    # Fairness Comparison Plot
    plt.hlines(dist_sol_fairness, xmin=0, xmax=args.iter, linestyles='solid', label='Distributed Objective Value')
    plt.hlines(central_sol_fairness, xmin=0, xmax=args.iter, linestyles='dashed', label='Central Objective Value')
    plt.hlines(init_fairness, xmin=0, xmax=args.iter, linestyles='dotted', label='Initial Objective Value')
    plt.legend()
    plt.savefig('plots/simple/final_fairness_value_comparison.png')
    plt.clf()

    # Obstacle Avoidance Comparison Plot
    plt.hlines(dist_sol_obstacle, xmin=0, xmax=args.iter, linestyles='solid', label='Distributed Objective Value')
    plt.hlines(central_sol_obstacle, xmin=0, xmax=args.iter, linestyles='dashed', label='Central Objective Value')
    plt.hlines(init_obstacle, xmin=0, xmax=args.iter, linestyles='dotted', label='Initial Objective Value')
    plt.legend()
    plt.savefig('plots/simple/final_obstacle_value_comparison.png')
    plt.clf()

    # Collision Avoidance Comparison Plot
    plt.hlines(dist_sol_collision, xmin=0, xmax=args.iter, linestyles='solid', label='Distributed Objective Value')
    plt.hlines(central_sol_collision, xmin=0, xmax=args.iter, linestyles='dashed', label='Central Objective Value')
    plt.hlines(init_collision, xmin=0, xmax=args.iter, linestyles='dotted', label='Initial Objective Value')
    plt.legend()
    plt.savefig('plots/simple/final_collision_value_comparison.png')
    plt.clf()
