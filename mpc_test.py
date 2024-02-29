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
from csv import writer
import time
import yaml


EPS = 1e-2
np.random.seed(42)

parser = argparse.ArgumentParser(description='Optimization')
# MAIN VARIABLES
parser.add_argument('--exp_dir', type=str, default='')
parser.add_argument('--N', type=int, default=3)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--notion', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--exp_params', type=str, default='IROS2024_experiments/toy_params.yaml')

args = parser.parse_args()
print(args)
exp_dir = 'test_results{}'.format(args.exp_dir)
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

N = args.N  # number of agents
trials = args.trials  # number of trials in experiment
notion = args.notion  # fairness notion
dist_nbf = args.dist  # using distributed NBF or non-distributed NBF

with open(args.exp_params, "r") as stream:
    exp_params = yaml.safe_load(stream)
exp_params = exp_params[N]
print(exp_params)

Tf = exp_params['Tf']  # time frame for mission
H = exp_params['H']  # horizon
Ubox = exp_params['Ubox']  # box constraint for controls
safe_dist = exp_params['sd']  # safe distance between drones

alpha = exp_params['fair_planner']['alpha']  # parameter for fairness constraint
kappa = exp_params['fair_planner']['distributed']['kappa']  # parameter for weighting change in epsilon for local problem in distributed fairness planner
eps_bounds = exp_params['fair_planner']['distributed']['eps_bounds']  # bounds for eps in an iteration of distributed fairness planner
fair_dist_iter = exp_params['fair_planner']['distributed']['iter']  # max iterations for distributed fairness planner to converge

safe_planner_central_params = exp_params['safe_planner']['central']
h_gamma = safe_planner_central_params['h_gamma']
V_alpha = safe_planner_central_params['V_alpha']
safe_planner_dist_params = exp_params['safe_planner']['distributed']
h_i = safe_planner_dist_params['hi']
h_o = safe_planner_dist_params['ho']
h_v = safe_planner_dist_params['hv']
nbf_dist_step_size = safe_planner_dist_params['step_size']
nbf_dist_trade_param = safe_planner_dist_params['trade_param']

obstacles = exp_params['obstacles']
goals = exp_params['goals']
starts = exp_params['starts']
print(obstacles)
print(goals)
print(starts)

dt = Tf/H

fair_planner_solver_errors = 0
nbf_solver_errors = 0

successful_trials = 0
collide_with_obstacle = 0
collide_with_drone = 0
misses_goal = 0

# SET INITIAL POSITIONS AND STATES
for t in range(trials):
    print('Trial {}'.format(t))
    trial_dir = '{}/trial{}'.format(exp_dir, t)
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)
    fair_planner_error = False
    trial_error = False
    trial_error_after_relax = False
    relax_num = 0

    init_pos = []
    init_states = []
    drone_starts = []
    drone_goals = []
    cbf_obstacles = []
    cbf_separation = []
    clf_reach = []
    runtimes_fair_planner = []
    runtimes_safe_planner = []
    for i in range(N):
        # Pick Start Area and Generate Random Position Within Area
        start = np.random.choice(list(starts.keys()))
        start_pos = starts[start]['center']
        start_rad = starts[start]['radius']
        x = np.random.uniform(low=start_pos[0]-start_rad, high=start_pos[0]+start_rad, size=1)[0]
        y = np.random.uniform(low=start_pos[1]-start_rad, high=start_pos[1]+start_rad, size=1)[0]
        init_pos.append(np.array([x, y, start_pos[2]]))
        
        s = [init_pos[i]]
        s.append(np.array([0, 0, 0]))  # velo
        init_states.append(np.array(s).flatten())

        # # Also Pick a Random goal
        # if start == 'a1':
        #     goal = 'g2'
        # else:
        #     goal = 'g1'
        goal = np.random.choice(list(goals.keys()))
        drone_starts.append(starts[start])
        drone_goals.append(goals[goal])
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

    # DO RECEDING HORIZON CONTROL
    Tbar = Tf
    final_us = []
    final_trajectories = [init_pos]
    final_vels = [[s[3:6] for s in init_states]]
    clf_values = []
    cbf_values = []
    all_deltas = []
    all_J_sequences = []
    last_delta = None
    for Hbar in range(H, 0, -1):
        # GENERATE INITIAL "CONTROL INPUTS" AND TRAJECTORIES
        init_u = []
        init_traj = []
        for i in range(N):
            goal = drone_goals[i]
            # if drone is already in goal area then, set goal to be current pos
            if np.linalg.norm(robots[i].state[0:3] - goal['center'])**2 <= goal['radius']**2:
                traj_pos, traj_accel = generate_init_traj_quad(robots[i].state, robots[i].state[0:3], Hbar+1, Tf=Tbar)    
            else:
                traj_pos, traj_accel = generate_init_traj_quad(robots[i].state, goal['center'], Hbar+1, Tf=Tbar)
            init_u.append(traj_accel)
            init_traj.append(traj_pos)

        init_u = np.array(init_u)

        # if t == 0 and Hbar == H:
        #     print('Figure For Singular Trajectories')
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     times = np.linspace(0, Tf, H)
        #     for i in range(N):
        #         _, traj = generate_agent_states(init_u[i], init_states[i], init_pos[i], model=Quadrocopter, dt=dt)
        #         ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
        #         ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)

        #     for obsId, obs in obstacles.items():
        #         co = obs['center']
        #         ro = obs['radius']
        #         obs_sphere = Sphere([co[0], co[1], co[2]], ro)
        #         obs_sphere.plot_3d(ax, alpha=0.2, color='red')
        #     for gId, g in goals.items():
        #         cg = g['center']
        #         rg = g['radius']
        #         goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
        #         goal_sphere.plot_3d(ax, alpha=0.2, color='green')
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
        seed_u = init_u      
        fair_planner_time_start = time.time()
        if notion != 2:
            obj = Objective(N, Hbar, system_model_config, init_states, init_pos, obstacles, drone_goals, drone_starts,  Q, alpha, kappa, eps_bounds, Ubox, dt=dt, notion=notion, safe_dist=safe_dist)
            obj.solo_energies = solo_energies
            # print('Running Fair Planner at time {}'.format(H-Hbar))
            try:
                # seed_u, local_sols, fairness, converge_iter = obj.solve_distributed(init_u, steps=fair_dist_iter, dyn='quad')
                seed_u = obj.solve_distributed(init_u, steps=fair_dist_iter, dyn='quad')
                # seed_obj, seed_u = obj.solve_central(init_u, steps=fair_dist_iter)
                seed_u = np.array(seed_u).reshape((N, Hbar, control_input_size))

                # if dist_nbf:
                if Hbar == H:
                    fair_us = np.array(seed_u)
                    fair_drone_results = obj.check_avoid_constraints(fair_us)
                    fair_trial_result = max(fair_drone_results)

                    fair_sol_energy = np.round(obj.quad(fair_us.flatten()), 3)
                    fair_sol_fairness1 = np.round(obj.fairness(fair_us.flatten()), 3)
                    fair_sol_fairness4 = np.round(obj.surge_fairness(fair_us.flatten()), 3)

                    fair_trial_res = [t, fair_trial_result, fair_sol_energy, fair_sol_fairness1, fair_sol_fairness4]
                    with open('{}/trial_results_init_fair_sol.csv'.format(exp_dir), 'a') as file_obj:
                        writer_obj = writer(file_obj)
                        writer_obj.writerow(fair_trial_res)

            except Exception as e:
                print(e)
                print('Fair Planner error at time {}'.format(H-Hbar))

                # use solo trajs as ref
                seed_u = init_u
                # if Hbar == H:
                #     seed_u = init_u
                # else:
                #     seed_u = seed_u[:, 1:, ]
                fair_planner_solver_errors += 1
                fair_planner_error = True
        runtimes_fair_planner.append(time.time() - fair_planner_time_start)

        # if t == 0 and ((Hbar - H) % 5 == 0):
        #     print('Figure For Fair Trajectories')
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     for i in range(N):
        #         _, traj = generate_agent_states(seed_u[i], init_states[i], init_pos[i], model=Quadrocopter, dt=dt)
        #         ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
        #         ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)
        #     for obsId, obs in obstacles.items():
        #         co = obs['center']
        #         ro = obs['radius']
        #         obs_sphere = Sphere([co[0], co[1], co[2]], ro)
        #         obs_sphere.plot_3d(ax, alpha=0.2, color='red')
        #     for gId, g in goals.items():
        #         cg = g['center']
        #         rg = g['radius']
        #         goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
        #         goal_sphere.plot_3d(ax, alpha=0.2, color='green')
        #     for sId, s in starts.items():
        #         cs = s['center']
        #         rs = s['radius']
        #         start_sphere = Sphere([cs[0], cs[1], cs[2]], rs)
        #         start_sphere.plot_3d(ax, alpha=0.2, color='blue')
        #     plt.show()

        # print('Running Safety Planner at time {}'.format(H-Hbar))
        safe_planner_time_start = time.time()
        obj = Objective(N, Hbar, system_model_config, init_states, init_pos, obstacles, drone_goals, drone_starts, Q, alpha, kappa, eps_bounds, Ubox, dt=dt, notion=notion, safe_dist=safe_dist)
        obj.solo_energies = solo_energies
        # final_u = seed_u
        try:
            if dist_nbf:
                test_uis, all_Js, cbfs, clfs = obj.solve_distributed_nbf(seed_u, last_delta,
                    h_i=h_i, h_o=h_o, h_v=h_v, step_size=nbf_dist_step_size, trade_param=nbf_dist_trade_param, steps=fair_dist_iter)
                final_u = test_uis[:,0:3]
                last_delta = test_uis[:,3]
                all_deltas.append(last_delta)
                all_J_sequences.append(all_Js)
                cbf_values.append(cbfs)
                clf_values.append(clfs)
            else:
                final_u, cbf_value, clf_value, nbf_delta, h_os, h_cs, Vs, relaxed = obj.solve_nbf(seed_u=seed_u, last_delta=last_delta, mpc=True, h_gamma=h_gamma, V_alpha=V_alpha)
                final_u = np.array(final_u)  # H, N, control_input    
                final_u = final_u.transpose(1, 0, 2)  # N, H, control_input
                cbf_values.append(cbf_value)
                clf_values.append(clf_value)
                last_delta = nbf_delta
                all_deltas.append(last_delta)
                
                cbf_obstacles.append(h_os)
                if len(h_cs) < (int(N*(N-1)/2)+1):
                    for ex in range(len(h_cs), (int(N*(N-1)/2))):
                        h_cs.append(0)
                cbf_separation.append(h_cs)
                clf_reach.append(Vs)
                
                if relaxed:
                    relax_num += 1
        except Exception as e:
            cbf_values.append(np.nan)
            clf_values.append(np.nan)
            print(e)
            if 'relax' in str(e):
                trial_error_after_relax = True
                last_delta = None
            print('Cant find next step in trajectory at time {}'.format(H-Hbar))
            nbf_solver_errors += 1

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # times = np.linspace(0, Tf, H)
            # temp_trajectories = np.array(final_trajectories)
            # temp_trajectories = temp_trajectories.transpose(1, 0, 2)  # N, H, positions
            # for i in range(N):
            #     traj = temp_trajectories[i]
            #     ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
            #     ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)
            # for obsId, obs in obstacles.items():
            #     co = obs['center']
            #     ro = obs['radius']
            #     obs_sphere = Sphere([co[0], co[1], co[2]], ro)
            #     obs_sphere.plot_3d(ax, alpha=0.2, color='red')
            # for gId, g in goals.items():
            #     cg = g['center']
            #     rg = g['radius']
            #     goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
            #     goal_sphere.plot_3d(ax, alpha=0.2, color='green')
            # for sId, s in starts.items():
            #     cs = s['center']
            #     rs = s['radius']
            #     start_sphere = Sphere([cs[0], cs[1], cs[2]], rs)
            #     start_sphere.plot_3d(ax, alpha=0.2, color='blue')
            # plt.show()

            # if infeasible, try using fair trajectories at this time
            if dist_nbf:
                final_u = seed_u[:,0,:]
            else:
                final_u = seed_u
            trial_error = True

        runtimes_safe_planner.append(time.time() - safe_planner_time_start)
        
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

        # if t == 0 and ((Hbar - H) % 5 == 0):
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
        #     for obsId, obs in obstacles.items():
        #         co = obs['center']
        #         ro = obs['radius']
        #         obs_sphere = Sphere([co[0], co[1], co[2]], ro)
        #         obs_sphere.plot_3d(ax, alpha=0.2, color='red')
        #     for gId, g in goals.items():
        #         cg = g['center']
        #         rg = g['radius']
        #         goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
        #         goal_sphere.plot_3d(ax, alpha=0.2, color='green')
        #     plt.show()

    if dist_nbf:
        # print('plot J values of first iteration')
        plt.plot(list(range(len(all_J_sequences[0]))), all_J_sequences[0])
        plt.savefig('{}/dist_cost_init.png'.format(trial_dir))
        plt.clf()

        plt.plot(list(range(len(all_J_sequences[-1]))), all_J_sequences[-1])
        plt.savefig('{}/dist_cost.png'.format(trial_dir))
        plt.clf()

    n = N*H*control_input_size
    Q = np.random.randn(n, n)   # variable for quadratic objective
    Q = Q.T @ Q
    obj = Objective(N, H, system_model_config, orig_init_states, orig_init_pos, obstacles, drone_goals, drone_starts, Q, alpha, kappa, eps_bounds, Ubox, dt=dt, notion=notion, safe_dist=safe_dist)
    obj.solo_energies = solo_energies

    final_us = np.array(final_us)
    final_us = final_us.transpose(1, 0, 2)
    drone_results = obj.check_avoid_constraints(final_us)
    trial_result = max(drone_results)

    sol_energy = np.round(obj.quad(final_us.flatten()), 3)
    sol_fairness1 = np.round(obj.fairness(final_us.flatten()), 3)
    sol_fairness4 = np.round(obj.surge_fairness(final_us.flatten()), 3)

    fair_planner_avg_runtime = np.round(np.mean(runtimes_fair_planner), 3)
    safe_planner_avg_runtime = np.round(np.mean(runtimes_safe_planner), 3)
    goals_made = obj.goals_made
    if goals_made < N:
        max_missed_distance = np.round(np.mean(obj.dist_to_goal), 3)
    else:
        max_missed_distance = 0

    if trial_result == 0:
        successful_trials += 1
        print('Success')
    elif trial_result == 1:
        misses_goal += 1
        print('Missed Goal')
    elif trial_result == 2:
        collide_with_obstacle += 1
        print('Collide')
    else:
        collide_with_drone += 1
        print('Hit Drone')
    
    if trial_error:
        print('Error in Safety Planner without relaxtion')
        trial_result += 10
    if fair_planner_error:
        print('Error in Fair Planner')
        trial_result += 20
    if trial_error_after_relax:
        print('Error in Safety Planner after relaxtion')
        trial_result += 50

    trial_res = [t, trial_result, sol_energy, sol_fairness1, sol_fairness4, fair_planner_avg_runtime, safe_planner_avg_runtime, relax_num, goals_made, max_missed_distance]
    with open('{}/trial_results.csv'.format(exp_dir), 'a') as file_obj:
        writer_obj = writer(file_obj)
        writer_obj.writerow(trial_res)
    
    final_trajectories = np.array(final_trajectories)
    final_trajectories = final_trajectories.transpose(1, 0, 2)  # N, H, positions
    
    # print('Figure Final Trajectories')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    times = np.linspace(0, Tf, H)
    for i in range(N):
        traj = final_trajectories[i]
        ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
        ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)
    for obsId, obs in obstacles.items():
        co = obs['center']
        ro = obs['radius']
        obs_sphere = Sphere([co[0], co[1], co[2]], ro)
        obs_sphere.plot_3d(ax, alpha=0.2, color='red')
    for gId, g in goals.items():
        cg = g['center']
        rg = g['radius']
        goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg + 0.4)
        goal_sphere.plot_3d(ax, alpha=0.2, color='green')
    for sId, s in starts.items():
        cs = s['center']
        rs = s['radius']
        start_sphere = Sphere([cs[0], cs[1], cs[2]], rs)
        start_sphere.plot_3d(ax, alpha=0.2, color='blue')
    plt.savefig('{}/final_traj.png'.format(trial_dir))
    plt.clf()
    plt.close()
    # plt.show()

    # print("Figure CLF and CBF Values")
    fig, axs = plt.subplots(2)
    axs[0].plot(list(range(len(cbf_values))), cbf_values)
    axs[0].set_title('h_min')
    axs[1].plot(list(range(len(clf_values))), clf_values)
    axs[1].set_title('V_max')
    plt.savefig('{}/final_cbf_clf.png'.format(trial_dir))
    plt.clf()
    plt.close()

    # Also plot delta values 
    # plt.plot(list(range(len(all_deltas))), all_deltas)
    # plt.savefig('{}/deltas.png'.format(trial_dir))
    # plt.clf()

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

    if not dist_nbf:
        save_cbf_clf_vals = np.round(np.array([cbf_values, clf_values]), 3)
        np.savetxt('{}/cbf_clf_values.csv'.format(trial_dir), save_cbf_clf_vals.T, fmt='%f')

        # if N == 3:
        np.savetxt('{}/ind_cbf_obstacles.csv'.format(trial_dir), np.array(cbf_obstacles), fmt='%f')
        np.savetxt('{}/ind_cbf_separation.csv'.format(trial_dir), np.array(cbf_separation), fmt='%f')
        np.savetxt('{}/ind_clf_reach.csv'.format(trial_dir), np.array(clf_reach), fmt='%f')
            
            # plt.plot(list(range(len(cbf_obstacles))), cbf_obstacles)
            # plt.savefig('{}/ind_cbf_obstacles.png'.format(trial_dir))
            # plt.clf()

            # plt.plot(list(range(len(cbf_separation))), cbf_separation)
            # plt.savefig('{}/ind_cbf_separation.png'.format(trial_dir))
            # plt.clf()

            # plt.plot(list(range(len(clf_reach))), clf_reach)
            # plt.savefig('{}/ind_cld_reach.png'.format(trial_dir))
            # plt.clf()
    else:
        save_cbf_clf_vals = np.round(np.array([np.min(cbf_values, axis=1), np.max(clf_values, axis=1)]), 3)
        np.savetxt('{}/cbf_clf_values.csv'.format(trial_dir), save_cbf_clf_vals.T, fmt='%f')

print('Successful Trials {}'.format(successful_trials))
print('Hit Obstacle {}'.format(collide_with_obstacle))
print('Hit Drone {}'.format(collide_with_drone))
print('Misses Goal {}'.format(misses_goal))

print('Solver Errors {}'.format(nbf_solver_errors))
print('Fair Planner Iteration Errors {}, Fraction of all iterations, {}'.format(fair_planner_solver_errors, fair_planner_solver_errors/(trials*H)))
