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
np.random.seed(42)

parser = argparse.ArgumentParser(description='Optimization')
# MAIN VARIABLES
parser.add_argument('--results_file', type=str, default='test')
parser.add_argument('--N', type=int, default=3)
parser.add_argument('--H', type=int, default=5)
parser.add_argument('--trials', type=int, default=100)
parser.add_argument('--notion', type=int, default=0)


# THESE MAY CHANGE (BUT IDEALLY NOT)
parser.add_argument('--alpha', type=float, default=.001)
parser.add_argument('--beta', type=float, default=1) # 10 for N < 10, 5 for N=15, 1 for N=20
parser.add_argument('--gamma', type=float, default=1.)
parser.add_argument('--kappa', type=float, default=0.5)
parser.add_argument('--eps_bounds', type=float, default=10)
   

# TRY NOT TO CHANGE THESE
parser.add_argument('--ro', type=float, default=.5)
parser.add_argument('--co', type=float, default=3)
parser.add_argument('--rg', type=float, default=3)
parser.add_argument('--cg', type=float, default=5)
parser.add_argument('--Ubox', type=float, default=100)
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
co = np.array([args.co, args.co, args.co])  # center of circle
rg = args.rg
cg = np.array([args.cg, args.cg, 0])
obstacles = {'center': co, 'radius': ro}
target = {'center': cg, 'radius': rg}
Ubox = args.Ubox  # box constraint
safe_dist = args.sd
notion = 2  # NOTION SHOULD ALWAYS BE 2

H = args.H

Tf = args.Tf

dt = Tf/H*1.5

trials = args.trials

# CREATE CSV TO SAVE RESULTS
# csv_cols = ['trial_num', 'success', 'obj', 'energy', 'f1', 'f4', 'obstacle', 'collision', 'walltime', 'cputime']
csv_cols = ['trial_num', 'success', 'energy', 'f1', 'f4', 'obstacle', 'collision', 'walltime', 'cputime']
save_time = datetime.now()
csv_name = 'seed_results/N{}_H{}_{}.csv'.format(N, H, save_time)
bc_obj_name = 'seed_results/control_inputs_N{}_H{}_{}.npy'.format(N, H, save_time)
file_obj = open(csv_name, 'a')
writer_obj = writer(file_obj)
writer_obj.writerow(csv_cols)

base_case_res = []
np.save(bc_obj_name, np.array(base_case_res))

count_reach = 0
for trial in range(trials):
    # SET INITIAL POSITIONS AND STATES
    if N < 10:
        x = np.random.uniform(low=-15, high=0, size=1)[0]
        y = np.random.uniform(low=-15, high=15, size=1)[0]
    else:
        x = np.random.uniform(low=-3, high=0, size=1)[0]
        y = np.random.uniform(low=-15, high=-12, size=1)[0]

    init_pos = []
    init_states = []
    for i in range(N):
        init_pos.append(np.array([x+(i), y, 0]))
        s = [init_pos[i]]
        s.append(np.array([0, 0, 0]))  # velo
        init_states.append(np.array(s).flatten())

    # GENERATE INITIAL "CONTROL INPUTS" AND TRAJECTORIES
    init_u = []
    init_traj = []
    for i in range(N):
        traj_pos, traj_accel = generate_init_traj_quad(init_pos[i], cg, H, Tf=Tf)
        init_u.append(traj_accel)
        init_traj.append(traj_pos)

    init_u = np.array(init_u)

    # GENERATE SOLO ENERGIES
    # use the above inputs from generate_init_traj_quad to get the solo energies 
    solo_energies = []
    for i in range(N):
        solo_energies.append(np.linalg.norm(init_u[i])**2)

    # INIT SOLVER
    system_model = Quadrocopter
    control_input_size = 3
    system_model_config = (Quadrocopter, control_input_size)

    n = N*H*control_input_size
    Q = np.random.randn(n, n)   # variable for quadratic objective
    Q = Q.T @ Q
    obj = Objective(N, H, system_model_config, init_states, init_pos, obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=dt, notion=notion)
    obj.solo_energies = solo_energies

    st = time.time()
    stp = time.process_time()
    try:
        seed_u = obj.solve_nbf()
        seed_u = np.array(seed_u)  # H, N, control_input    
        seed_u = seed_u.transpose(1, 0, 2)  # N, H, control_input
    except:
        print('error solving, continue')
        continue
    etp = time.process_time()
    et = time.time()
    walltime = et - st
    cputime = etp - stp
 
    seed_u = np.array(seed_u).reshape((N, H, control_input_size))

    # Save Results to File
    if obj.check_avoid_constraints(seed_u): #and obj.reach_constraint(seed_u):
        valid_sol = int(obj.reach_constraint(seed_u) <= 0)
        central_sol_energy = obj.quad(seed_u.flatten())
        central_sol_fairness1 = obj.fairness(seed_u.flatten())  # f1
        central_sol_fairness4 = obj.surge_fairness(seed_u.flatten())  # f4
        central_sol_obstacle = obj.obstacle(seed_u.flatten())
        central_sol_collision = obj.avoid_constraint(seed_u.flatten())
        count_reach += 1

        # Save Final Solution To File For use in Distributed Experiment
        base_case_res = np.load(bc_obj_name)
        init_pos_and_final_u = np.append(np.array(init_pos).flatten(), seed_u.flatten())
        if base_case_res.size == 0:
            np.save(bc_obj_name, init_pos_and_final_u.reshape((1, (3*N)+n)))
            del base_case_res
        else:    
            base_case_res = np.append(base_case_res, init_pos_and_final_u.reshape((1, (3*N)+n)), axis=0)
            np.save(bc_obj_name, base_case_res)
            del base_case_res

        res = [trial, valid_sol, central_sol_energy, central_sol_fairness1, central_sol_fairness4, central_sol_obstacle, central_sol_collision, walltime, cputime]
        writer_obj.writerow(res)
    else: 
        valid_sol = 0
        central_sol_energy = 0
        central_sol_fairness1 = 0
        central_sol_fairness4 = 0
        central_sol_obstacle = 0
        central_sol_collision = 0

    # print('Trial {} Result: {}'.format(trial, valid_sol))

    # PLOT TRAEJECTORIES FROM SEEDED SOLUTION
    # final_trajectories = []
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # times = np.linspace(0, Tf, H)
    # for i in range(N):
    #     _, traj = generate_agent_states(seed_u[i], init_states[i], init_pos[i], model=Quadrocopter, dt=dt)
    #     ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
    #     ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)
    #     final_trajectories.append(traj)
    # obs_sphere = Sphere([co[0], co[1], co[2]], ro)
    # obs_sphere.plot_3d(ax, alpha=0.2, color='red')
    # goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
    # goal_sphere.plot_3d(ax, alpha=0.2, color='green')
    # plt.show()


    # obj = Objective(N, H, system_model_config, init_states, init_pos, obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=dt, notion=notion)
    # obj.solo_energies = solo_energies
    # # TEST OBSTACLE LOCAL
    # # PARTIALS  = obj._obstacle_local(seed_u, dyn='quad')
    # # PARTIALS  = obj._avoid_local(seed_u, dyn='quad')
    # # print(PARTIALS)
    # final_u, local_sols, fairness, converge_iter = obj.solve_distributed(seed_u, steps=args.iter, dyn='quad')
    # # final_obj, final_u = obj.solve_central(seed_u, steps=args.iter)
    # success = 0 if len(final_u) == 0 else 1
    # if success == 1:
    #     final_u = final_u.reshape(N, H, control_input_size)
    #     print(obj.check_avoid_constraints(final_u))
    #     print(obj.reach_constraint(final_u))

    #     # PLOT FINAL TRAEJECTORIES FROM CONTROL INPUTS
    #     final_trajectories = []
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     times = np.linspace(0, Tf, H)
    #     for i in range(N):
    #         _, traj = generate_agent_states(final_u[i], init_states[i], init_pos[i], model=Quadrocopter, dt=dt)
    #         ax.plot(traj[:,0], traj[:,1], traj[:,2], label=i)
    #         # ax.scatter(traj[:,0], traj[:,1], traj[:,2], label=i)
    #         # ax.scatter(traj[5,0], traj[5,1], traj[5,2], label=i)
    #         final_trajectories.append(traj)
    #     obs_sphere = Sphere([co[0], co[1], co[2]], ro)
    #     obs_sphere.plot_3d(ax, alpha=0.2, color='red')
    #     goal_sphere = Sphere([cg[0], cg[1], cg[2]], rg)
    #     goal_sphere.plot_3d(ax, alpha=0.2, color='green')
    #     plt.show()

print('reach goal', count_reach)