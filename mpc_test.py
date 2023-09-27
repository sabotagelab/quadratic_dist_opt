import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from skspatial.objects import Sphere

from objective import Objective
from generate_trajectories import Quadrocopter
from generate_trajectories import generate_agent_states, generate_init_traj_quad
import sys
from csv import writer
import time


EPS = 1e-2
np.random.seed(43)

parser = argparse.ArgumentParser(description='Optimization')
# MAIN VARIABLES
parser.add_argument('--results_file', type=str, default='test')
parser.add_argument('--N', type=int, default=3)
parser.add_argument('--H', type=int, default=7)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--notion', type=int, default=0)


# THESE MAY CHANGE (BUT IDEALLY NOT)
parser.add_argument('--alpha', type=float, default=.001)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--gamma', type=float, default=1.)
parser.add_argument('--kappa', type=float, default=500)
parser.add_argument('--eps_bounds', type=float, default=25)
   

# TRY NOT TO CHANGE THESE
parser.add_argument('--ro', type=float, default=1)
parser.add_argument('--co', type=float, default=-4)
parser.add_argument('--rg', type=float, default=5)
parser.add_argument('--cg', type=float, default=5)
parser.add_argument('--Ubox', type=float, default=300)
parser.add_argument('--sd', type=float, default=0.1)
parser.add_argument('--iter', type=int, default=1000)
parser.add_argument('--Tf', type=int, default=1)

args = parser.parse_args()
print(args)
results_file = args.results_file

N = args.N  # number of agents
alpha = args.alpha   # parameter for fairness constraint
beta = args.beta    # parameter for weighting of obstacle avoidance constraint
gamma = args.gamma   # parameter for smoothmin in calculating obstacle avoidance constraint
kappa = args.kappa   # parameter for weighting change in epsilon for local problem
eps_bounds = args.eps_bounds  # bounds for eps in an iteration

ro = args.ro  # radius of circle
co = np.array([args.co, args.co, 0])  # center of circle
rg = args.rg
cg = np.array([args.cg, args.cg, 0])
obstacles = {'center': co, 'radius': ro+1}
target = {'center': cg, 'radius': rg}
Ubox = args.Ubox  # box constraint
safe_dist = args.sd
notion = args.notion  # NOTION SHOULD ALWAYS BE 2

H = args.H

Tf = args.Tf

# dt = Tf/H*1.5
dt = Tf/H

trials = args.trials

fair_planner_solver_errors = 0
nbf_solver_errors = 0

successful_trials = 0
collide_with_obstacle = 0
collide_with_drone = 0
misses_goal = 0

# SET INITIAL POSITIONS AND STATES
for t in range(trials):
    print('Trial {}'.format(t))
    trial_error = False
    if N >= 10:
        # x = np.random.uniform(low=-3, high=0, size=1)[0]
        # y = np.random.uniform(low=-15, high=-12, size=1)[0]
        x = np.random.uniform(low=-10, high=-5, size=1)[0]
        y = np.random.uniform(low=-15, high=-10, size=1)[0]

    init_pos = []
    init_states = []
    for i in range(N):
        if N < 10:
            x = np.random.uniform(low=-10, high=-5, size=1)[0]
            y = np.random.uniform(low=-10, high=-5, size=1)[0]
            init_pos.append(np.array([x, y, 0]))
        else:
            init_pos.append(np.array([x+(i), y, 0]))
        
        s = [init_pos[i]]
        s.append(np.array([0, 0, 0]))  # velo
        init_states.append(np.array(s).flatten())
    orig_init_states = list(init_states)
    orig_init_pos = list(init_pos)

    # Create Quad systems for each 
    system_model = Quadrocopter
    control_input_size = 3
    system_model_config = (Quadrocopter, control_input_size)

    robots = []
    for r in range(N):
        rob = system_model(init_states[r], dt=dt)
        robots.append(rob)

    # JUST FOR TESTING MARGELLOS IMPLEMENTATION
    test_robots = []
    for r in range(N):
        rob = system_model(init_states[r], dt=dt)
        test_robots.append(rob)

    # DO RECEDING HORIZON CONTROL
    Tbar = Tf
    final_us = []
    final_trajectories = [init_pos]
    clf_values = []
    cbf_values = []
    last_alpha = None
    for Hbar in range(H, 0, -1):
        # GENERATE INITIAL "CONTROL INPUTS" AND TRAJECTORIES
        init_u = []
        init_traj = []
        for i in range(N):
            leftright = 1 if i % 2 == 0 else -1
            x_adj = 1 if i % 2 == 0 else 0
            z_adj = 1 if (i % 3 == 0) and N >= 10 else 0
            pos_adj = np.array([x_adj, 1, z_adj])
            traj_pos, traj_accel = generate_init_traj_quad(robots[i].state, cg+leftright*i*args.sd*pos_adj, Hbar+1, Tf=Tbar)
            init_u.append(traj_accel)
            init_traj.append(traj_pos)

        init_u = np.array(init_u)

        # if t == 11:
        #     print('Figure For Singular Trajectories')
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     times = np.linspace(0, Tf, H)
        #     for i in range(N):
        #         _, traj = generate_agent_states(init_u[i], init_states[i], init_pos[i], model=Quadrocopter, dt=dt)
        #         ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
        #         ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)

        #     obs_sphere = Sphere([co[0], co[1], co[2]], ro)
        #     obs_sphere.plot_3d(ax, alpha=0.2, color='red')
        #     goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
        #     goal_sphere.plot_3d(ax, alpha=0.2, color='green')
        #     plt.show()

        # GENERATE SOLO ENERGIES
        # use the above inputs from generate_init_traj_quad to get the solo energies 
        solo_energies = []
        for i in range(N):
            solo_energies.append(np.linalg.norm(init_u[i])**2)
        
        # INIT SOLVER
        n = N*Hbar*control_input_size
        Q = np.random.randn(n, n)   # variable for quadratic objective
        Q = Q.T @ Q
        # seed_u = init_u  # COMMENTING OUT FAIRNESS FOR TESTING MARGELLOS IMPLEMENTATION
        obj = Objective(N, Hbar, system_model_config, init_states, init_pos, obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=dt, notion=notion, safe_dist=args.sd)
        obj.solo_energies = solo_energies
        obj.with_safety = False

        try:
            seed_u, local_sols, fairness, converge_iter = obj.solve_distributed(init_u, steps=args.iter, dyn='quad')
            # seed_obj, seed_u = obj.solve_central(init_u, steps=args.iter)
            seed_u = np.array(seed_u).reshape((N, Hbar, control_input_size))

            # if t == 11:
            #     print('Figure For Fair Trajectories')
            #     fig = plt.figure()
            #     ax = fig.add_subplot(projection='3d')
            #     for i in range(N):
            #         _, traj = generate_agent_states(seed_u[i], init_states[i], init_pos[i], model=Quadrocopter, dt=dt)
            #         ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
            #         ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)
            #     obs_sphere = Sphere([co[0], co[1], co[2]], ro)
            #     obs_sphere.plot_3d(ax, alpha=0.2, color='red')
            #     goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
            #     goal_sphere.plot_3d(ax, alpha=0.2, color='green')
            #     plt.show()
        except Exception as e:
            print(e)
            print('Fair Planner error at time {}'.format(H-Hbar))
            # use previous fair trajectory input instead of solo trajectory input if possible
            seed_u = seed_u[:,1:,:]        
            fair_planner_solver_errors += 1

        obj = Objective(N, Hbar, system_model_config, init_states, init_pos, obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=dt, notion=notion, safe_dist=args.sd)
        obj.solo_energies = solo_energies
        try:
            final_u, cbf_value, clf_value, nbf_alpha = obj.solve_nbf(seed_u=seed_u, last_alpha=last_alpha, mpc=True)
            final_u = np.array(final_u)  # H, N, control_input    
            final_u = final_u.transpose(1, 0, 2)  # N, H, control_input
            cbf_values.append(cbf_value)
            clf_values.append(clf_value)
            last_alpha = nbf_alpha

            # TESTING DIST NBF STUFF
            # test_uis = obj.solve_distributed_nbf(seed_u, last_alpha)
            # final_u = test_uis[:,0:3]
            # last_alpha = np.max(test_uis[:,3])
            
        except Exception as e:
            print(e)
            print('Cant find next step in trajectory at time {}'.format(H-Hbar))
            nbf_solver_errors += 1
            trial_error = True
            break
        
        Tbar = Tf - (H-Hbar)*Tf/H

        new_us = []
        new_states = []
        new_poses = []
        for r in range(N):
            new_state, new_pos = robots[r].forward(final_u[r][0])
            # new_state, new_pos = robots[r].forward(final_u[r])  # TESTING DIST NBF STUFF
            new_states.append(new_state)
            new_poses.append(new_pos)
            new_us.append(final_u[r][0])
            # new_us.append(final_u[r]) # TESTING DIST NBF STUFF
        final_us.append(new_us)
        final_trajectories.append(new_poses)
        init_states = new_states
        init_pos = new_poses

        # if t == 11:
        #     print('Figure For Safe Trajectories')
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     times = np.linspace(0, Tf, H)
        #     temp_trajectories = np.array(final_trajectories)
        #     temp_trajectories = temp_trajectories.transpose(1, 0, 2)  # N, H, positions
        #     for i in range(N):
        #         traj = temp_trajectories[i]
        #         ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
        #         ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)
        #     obs_sphere = Sphere([co[0], co[1], co[2]], ro)
        #     obs_sphere.plot_3d(ax, alpha=0.2, color='red')
        #     goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
        #     goal_sphere.plot_3d(ax, alpha=0.2, color='green')
        #     plt.show()

    if trial_error:
        continue
    final_trajectories = np.array(final_trajectories)
    final_trajectories = final_trajectories.transpose(1, 0, 2)  # N, H, positions

    n = N*H*control_input_size
    Q = np.random.randn(n, n)   # variable for quadratic objective
    Q = Q.T @ Q
    obj = Objective(N, H, system_model_config, orig_init_states, orig_init_pos, obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=dt, notion=notion, safe_dist=args.sd)
    obj.solo_energies = solo_energies
    obj.with_safety = True

    final_us = np.array(final_us)
    final_us = final_us.transpose(1, 0, 2)
    trial_result = obj.check_avoid_constraints(final_us)

    if trial_result == 0:
        successful_trials += 1
    elif trial_result == 1:
        collide_with_obstacle += 1
        print('Collide')
    elif trial_result == 2:
        collide_with_drone += 1
        print('Hit Drone')
    else:
        misses_goal += 1
        print('Missed Goal')
    
    print('Figure Final Trajectories')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    times = np.linspace(0, Tf, H)
    for i in range(N):
        traj = final_trajectories[i]
        ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
        ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)
    obs_sphere = Sphere([co[0], co[1], co[2]], ro)
    obs_sphere.plot_3d(ax, alpha=0.2, color='red')
    goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
    goal_sphere.plot_3d(ax, alpha=0.2, color='green')
    plt.show()

    print("Figure CLF and CBF Values")
    fig, axs = plt.subplots(2)
    axs[0].plot(list(range(len(cbf_values))), cbf_values)
    axs[0].set_title('h_min')
    axs[1].plot(list(range(len(clf_values))), clf_values)
    axs[1].set_title('V_max')
    plt.show()

print('Successful Trials {}'.format(successful_trials))
print('Hit Obstacle {}'.format(collide_with_obstacle))
print('Hit Drone {}'.format(collide_with_drone))
print('Misses Goal {}'.format(misses_goal))

print('Solver Errors {}'.format(nbf_solver_errors))
print('Fair Planner Iteration Errors {}, Fraction of all iterations, {}'.format(fair_planner_solver_errors, fair_planner_solver_errors/(trials*args.H)))