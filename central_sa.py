import argparse
import cvxpy as cp
import numpy as np
from scipy.optimize import basinhopping


def objective(u, Q, alpha, beta, gamma):
    main_obj = np.dot(u, np.dot(Q, u))

    u_reshape = u.reshape(int(n/2), 2)
    u_mean = np.mean(np.linalg.norm(u_reshape, axis=0))
    # u_mean = np.mean(u_reshape)
    diffs = 0
    logsum = 0
    for i in range(N):
        diffs += (np.linalg.norm(u_reshape[i]) - u_mean) ** 2
        logsum += np.exp(-gamma * ((np.linalg.norm(u_reshape[i] - c) ** 2) - r*r))
    fairness_obj = 1/(N-1) * np.sum(diffs)

    obs_obj = -gamma * np.log(logsum)

    return main_obj + alpha*fairness_obj - beta*obs_obj

# Basin Hopping Process
def basin_hopping(u, Q, alpha, beta, gamma, eps_bounds, steps=1000):
    func = lambda x: objective(x, Q, alpha, beta, gamma)
    x0 = u
    res = basinhopping(func, x0, stepsize=eps_bounds, niter=steps)
    final_u = res.x
    final_obj = res.fun

    return final_obj, final_u

# Simulated Annealing Process
def propose(u, eps_bounds, Ubox):
    random_change = np.random.uniform(low=-eps_bounds, high=eps_bounds, size=u.shape)
    new_u = u + random_change
    new_u[new_u > Ubox] = Ubox
    new_u[new_u < -Ubox] = -Ubox
    return new_u

def simulated_annealing(init_u, Q, alpha, beta, gamma, steps=1000):
    steps = 1000

    H = np.logspace(1, 3, steps)
    temperature = np.logspace(1, -8, steps)

    new_u = np.zeros(n)
    init_obj = objective(init_u, Q, alpha, beta, gamma)
    for i in range(steps):
        T = temperature[i]
        propose_u = propose(init_u, eps_bounds, Ubox)
        propose_u_obj = objective(propose_u, Q, alpha, beta, gamma)

        if propose_u_obj < init_obj:
            new_u = propose_u
            init_obj = propose_u_obj
        else:
            p_accept = np.exp((-1 * (propose_u_obj - init_obj)) / T)
            accept_criteria = np.random.uniform(0, 1)
            if accept_criteria < p_accept:
                new_u = propose_u
                init_obj = propose_u_obj
    
    return init_obj, new_u


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Centralized Optimization')
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--beta', type=float, default=.01)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--kappa', type=float, default=.1)
    parser.add_argument('--eps_bounds', type=float, default=.1)
    parser.add_argument('--r', type=float, default=5)
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument('--Ubox', type=float, default=10)
    parser.add_argument('--iter', type=int, default=1000)


    args = parser.parse_args()
    print(args)

    N = args.N
    n = N*2

    Q = np.random.randn(n, n)   # variable for quadratic objective
    Q = Q.T @ Q

    a = np.random.rand(n)

    alpha = args.alpha   # parameter for fairness constraint
    beta = args.beta    # parameter for weighting of obstacle avoidance constraint
    gamma = args.gamma   # parameter for smoothmin in calculating obstacle avoidance constraint
    kappa = args.kappa   # parameter for weighting change in epsilon for local problem
    eps_bounds = args.eps_bounds  # bounds for eps in an iteration

    r = args.r  # radius of circle
    c = np.array([args.c, args.c])  # center of circle
    Ubox = args.Ubox  # box constraint

    init_u = np.random.rand(n)
    # init_u = np.array([0.73846658, 0.17136828, -0.11564828, -0.3011037 , -1.47852199, -0.71984421])

    print('init u')
    print(init_u)

    print('Init Obj')
    print(objective(init_u, Q, alpha, beta, gamma))

    print("Using Simulated Annealing")
    final_obj, final_u = simulated_annealing(init_u, Q, alpha, beta, gamma)

    print('Final Objective Value')
    print(final_obj)

    print('Final u')
    print(final_u)

    print("Using SciPy Basin Hopping")
    final_obj, final_u = basin_hopping(init_u, Q, alpha, beta, gamma, eps_bounds)

    print('Final Objective Value')
    print(final_obj)

    print('Final u')
    print(final_u)
