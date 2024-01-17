import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import os
from skspatial.objects import Sphere

from objective import Objective
from generate_trajectories import Quadrocopter
from generate_trajectories import generate_agent_states, generate_init_traj_quad
import sys
from csv import writer
import time


EPS = 1e-2
np.random.seed(44)

parser = argparse.ArgumentParser(description='Optimization')
# MAIN VARIABLES
parser.add_argument('--results_file', type=str, default='test')
parser.add_argument('--N', type=int, default=3)
parser.add_argument('--H', type=int, default=12)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--notion', type=int, default=0)
parser.add_argument('--dist', action='store_true')


# THESE MAY CHANGE (BUT IDEALLY NOT)
parser.add_argument('--alpha', type=float, default=.001)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--gamma', type=float, default=1.)
parser.add_argument('--kappa', type=float, default=500)
parser.add_argument('--eps_bounds', type=float, default=100)
   

# TRY NOT TO CHANGE THESE
parser.add_argument('--ro', type=float, default=1.25)
parser.add_argument('--co', type=float, default=-2)
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
# co = np.array([args.co, args.co, 0])  # center of circle
co = np.array([0, args.co, 0])  # center of circle
rg = args.rg
cg = np.array([args.cg, args.cg, 0])
obstacles = {'center': co, 'radius': ro+1}
target = {'center': cg, 'radius': rg}
Ubox = args.Ubox  # box constraint
safe_dist = args.sd
notion = args.notion  # NOTION SHOULD ALWAYS BE 2
dist_nbf = args.dist

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
    trial_dir = 'test_results/trial{}'.format(t)
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)
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
            # x = np.random.uniform(low=-10, high=-5, size=1)[0]
            x = np.random.uniform(low=-10, high=10, size=1)[0]
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
    final_vels = [[s[3:6] for s in init_states]]
    clf_values = []
    cbf_values = []
    all_alphas = []
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

        # if t == 0:
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

            # if t == 0:
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
            # seed_u = seed_u[:,1:,:]        

            # use solo trajs as ref
            seed_u = init_u
            fair_planner_solver_errors += 1

        obj = Objective(N, Hbar, system_model_config, init_states, init_pos, obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=dt, notion=notion, safe_dist=args.sd)
        obj.solo_energies = solo_energies
        try:
            if dist_nbf:
                test_uis = obj.solve_distributed_nbf(seed_u, last_alpha)
                final_u = test_uis[:,0:3]
                # last_alpha = np.max(test_uis[:,3])
                last_alpha = test_uis[:,3]
                all_alphas.append(last_alpha)
            else:
                final_u, cbf_value, clf_value, nbf_alpha = obj.solve_nbf(seed_u=seed_u, last_alpha=last_alpha, mpc=True)
                final_u = np.array(final_u)  # H, N, control_input    
                final_u = final_u.transpose(1, 0, 2)  # N, H, control_input
                cbf_values.append(cbf_value)
                clf_values.append(clf_value)
                last_alpha = nbf_alpha
                all_alphas.append(last_alpha)
        except Exception as e:
            print(e)
            print('Cant find next step in trajectory at time {}'.format(H-Hbar))
            nbf_solver_errors += 1
            # if infeasible, try using fair trajectories at this time
            if dist_nbf:
                final_u = seed_u[:,0,:]
            else:
                final_u = seed_u
            trial_error = True
            # break
        
        Tbar = Tf - (H-Hbar)*Tf/H

        new_us = []
        new_states = []
        new_poses = []
        for r in range(N):
            if dist_nbf:
                new_state, new_pos = robots[r].forward(final_u[r])
                new_us.append(final_u[r])
            else:
                new_state, new_pos = robots[r].forward(final_u[r][0])
                new_us.append(final_u[r][0])
            new_states.append(new_state)
            new_poses.append(new_pos)
    
        final_us.append(new_us)
        final_trajectories.append(new_poses)
        final_vels.append([s[3:6] for s in new_states])
        init_states = new_states
        init_pos = new_poses

        # if t == 0:
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

    # if trial_error:
    #     continue

    n = N*H*control_input_size
    Q = np.random.randn(n, n)   # variable for quadratic objective
    Q = Q.T @ Q
    obj = Objective(N, H, system_model_config, orig_init_states, orig_init_pos, obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=dt, notion=notion, safe_dist=args.sd)
    obj.solo_energies = solo_energies
    obj.with_safety = True

    final_us = np.array(final_us)
    final_us = final_us.transpose(1, 0, 2)
    # trial_result = obj.check_avoid_constraints(final_us)
    drone_results = obj.check_avoid_constraints(final_us)
    trial_result = max(drone_results)

    sol_energy = np.round(obj.quad(final_us.flatten()), 3)
    sol_fairness1 = np.round(obj.fairness(final_us.flatten()), 3)
    sol_fairness4 = np.round(obj.surge_fairness(final_us.flatten()), 3)

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
    
    if trial_error:
        print('Error in Safety Planner')
        trial_result += 10

    trial_res = [t, trial_result, sol_energy, sol_fairness1, sol_fairness4]
    with open('{}/trial_results.csv'.format('test_results'), 'a') as file_obj:
        writer_obj = writer(file_obj)
        writer_obj.writerow(trial_res)
    
    final_trajectories = np.array(final_trajectories)
    final_trajectories = final_trajectories.transpose(1, 0, 2)  # N, H, positions
    
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
    plt.savefig('{}/final_traj.png'.format(trial_dir))
    plt.clf()
    # plt.show()

    print("Figure CLF and CBF Values")
    if not dist_nbf:
        fig, axs = plt.subplots(2)
        axs[0].plot(list(range(len(cbf_values))), cbf_values)
        axs[0].set_title('h_min')
        axs[1].plot(list(range(len(clf_values))), clf_values)
        axs[1].set_title('V_max')
        plt.savefig('{}/final_cbf_clf.png'.format(trial_dir))
        plt.clf()
        # plt.show()

    # Also plot Alpha values 
    plt.plot(range(list(len(all_alphas))), all_alphas)
    plt.savefig('{}/alphas.png'.format(trial_dir))
    plt.clf()
        

    # SAVE FINAL TRAJ
    final_trajectories = np.round(final_trajectories.reshape((N, H*3+3)), 3)
    drone_results = np.array(drone_results).reshape(N, 1)
    final_trajectories_result = np.concatenate([drone_results, final_trajectories], axis=1)
    np.savetxt('{}/final_traj.csv'.format(trial_dir), final_trajectories_result, fmt='%f')

    # SAVE FINAL VELS
    final_vels = np.array(final_vels)
    final_vels = final_vels.transpose(1, 0, 2)  # N, H, vels
    final_vels = np.round(final_vels.reshape((N, H*3+3)), 3)
    final_vels_result = np.concatenate([drone_results, final_vels], axis=1)
    np.savetxt('{}/final_vels.csv'.format(trial_dir), final_vels_result, fmt='%f')

    if len(cbf_values) > 0:
        save_cbf_clf_vals = np.round(np.array([cbf_values, clf_values, all_alphas]), 3)
        np.savetxt('{}/cbf_clf_alpha_values.csv'.format(trial_dir), save_cbf_clf_vals.T, fmt='%f')

print('Successful Trials {}'.format(successful_trials))
print('Hit Obstacle {}'.format(collide_with_obstacle))
print('Hit Drone {}'.format(collide_with_drone))
print('Misses Goal {}'.format(misses_goal))

print('Solver Errors {}'.format(nbf_solver_errors))
print('Fair Planner Iteration Errors {}, Fraction of all iterations, {}'.format(fair_planner_solver_errors, fair_planner_solver_errors/(trials*args.H)))
