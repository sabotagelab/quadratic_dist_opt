import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np

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
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--gamma', type=float, default=.1)
parser.add_argument('--kappa', type=float, default=0.5)
parser.add_argument('--eps_bounds', type=float, default=10)
   

# TRY NOT TO CHANGE THESE
parser.add_argument('--ro', type=float, default=.5)
parser.add_argument('--co', type=float, default=3)
parser.add_argument('--rg', type=float, default=3)
parser.add_argument('--cg', type=float, default=5)
parser.add_argument('--Ubox', type=float, default=100)
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
notion = args.notion

H = args.H

Tf = args.Tf

trials = args.trials

# CREATE CSV TO SAVE RESULTS
# csv_cols = ['trial_num', 'success', 'obj', 'energy', 'fairness', 'obstacle', 'collision', 'walltime', 'cputime']
csv_cols = ['trial_num', 'success', 'obj', 'energy', 'f1', 'f4', 'obstacle', 'collision', 'walltime', 'cputime']
csv_name = 'results/central_{}_N{}_H{}_{}.csv'.format(results_file, N, H, datetime.now())
file_obj = open(csv_name, 'a')
writer_obj = writer(file_obj)
writer_obj.writerow(csv_cols)

exp_start_time = time.time()
for trial in range(trials):
    if (trial % 10) == 0:
        # SET INITIAL POSITIONS AND STATES
        x = np.random.uniform(low=-15, high=0, size=1)[0]
        y = np.random.uniform(low=-15, high=15, size=1)[0]

        init_pos = []
        init_states = []
        for i in range(N):
            # x = np.random.uniform(low=-15, high=15, size=1)[0]
            # y = np.random.uniform(low=-15, high=15, size=1)[0]
            # init_pos.append(np.array([x, y+(i), 0]))
            init_pos.append(np.array([x+(i), y, 0]))
            # init_pos.append(np.array([x, y, 0]))
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
        system_model = Quadrocopter
        control_input_size = 3
        system_model_config = (Quadrocopter, control_input_size)
        solo_energies = []
        for i in range(N):
            n = 1*H*control_input_size
            Q = np.eye(n)
            obj = Objective(1, H, system_model_config, [init_states[i]], [init_pos[i]], obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=Tf/H*1.5, notion=notion)
            final_obj, final_u = obj.solve_central(init_u[i], steps=args.iter)
            init_solo_energy = obj.quad(final_u.flatten())
            solo_energies.append(init_solo_energy)

    # INIT SOLVER
    n = N*H*control_input_size
    # Q = np.eye(n)
    Q = np.random.randn(n, n)   # variable for quadratic objective
    Q = Q.T @ Q
    obj = Objective(N, H, system_model_config, init_states, init_pos, obstacles, target, Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=Tf/H*1.5, notion=notion)
    obj.solo_energies = solo_energies

    # SOLVE USING CENTRAL
    st = time.time()
    stp = time.process_time()
    final_obj, final_u = obj.solve_central(init_u, steps=args.iter)
    etp = time.process_time()
    et = time.time()
    walltime = et - st
    cputime = etp - stp

    # Save Results to File
    # ['trial_num', 'success', 'obj', 'energy', 'fairness', 'obstacle', 'collision', 'walltime', 'cputime']
    success = 0 if len(final_u) == 0 else 1
    if success == 1:
        valid_sol = obj.check_avoid_constraints(final_u)
        central_sol_obj = final_obj
        central_sol_energy = obj.quad(final_u.flatten())
        central_sol_fairness1 = obj.fairness(final_u.flatten())  # f1
        central_sol_fairness4 = obj.surge_fairness(final_u.flatten())  # f4
        central_sol_obstacle = obj.obstacle(final_u.flatten())
        central_sol_collision = obj.avoid_constraint(final_u.flatten())
    else: 
        valid_sol = False
        central_sol_obj = 0
        central_sol_energy = 0
        central_sol_fairness1 = 0
        central_sol_fairness4 = 0
        central_sol_obstacle = 0
        central_sol_collision = 0

    print('Trial {} Result: {}'.format(trial, valid_sol))

    # csv_cols = ['trial_num', 'success', 'obj', 'energy', 'f1', 'f4', 'obstacle', 'collision', 'walltime', 'cputime']
    res = [trial, valid_sol, central_sol_obj, central_sol_energy, central_sol_fairness1, central_sol_fairness4, central_sol_obstacle, central_sol_collision, walltime, cputime]

    writer_obj.writerow(res)

    if time.time() - exp_start_time > 3600:
        print('Killing Exp Early')
        break

file_obj.close()

