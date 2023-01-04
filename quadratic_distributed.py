import argparse

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


parser = argparse.ArgumentParser(description='Distributed Optimization')
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
# Q = np.eye(n)

a = np.random.rand(n)

alpha = args.alpha   # parameter for fairness constraint
beta = args.beta    # parameter for weighting of obstacle avoidance constraint
gamma = args.gamma   # parameter for smoothmin in calculating obstacle avoidance constraint
kappa = args.kappa   # parameter for weighting change in epsilon for local problem
eps_bounds = args.eps_bounds  # bounds for eps in an iteration

r = args.r  # radius of circle
c = np.array([args.c, args.c])  # center of circle
Ubox = args.Ubox  # box constraint



print('Distributed Problem, {} Agents'.format(N))
agents_prev_eps = {}
for i in range(N):
    agents_prev_eps[i] = cp.Parameter(2, value=np.zeros(2))

init_u = np.random.randn(n)  # TODO: always enforce that init_u is feasible
# init_u = np.array([ 0.65978519,  0.20293734, -0.01576553, -0.39146129, -1.38308715, -0.73846262])
agents_prev_u = {}
for i in range(N):
    agents_prev_u[i] = cp.Parameter(n, value=np.copy(init_u))

def generate_agent_variable_stacks(N):
    """

    :return:
        agent_stacks, dictionary mapping agents to their local u parameter
        agent_eps, dictionary mapping agents to their local eps variable

    """
    agent_stacks = {}
    agent_eps = {}

    for i in range(N):
        eps = cp.Variable(2)
        if i == 0:
            eps_zeros_after = cp.Parameter(n-2, value=np.zeros(2*(N-1)))
            stack = cp.hstack([eps, eps_zeros_after])
        elif i < N-1:
            eps_zeros_before = cp.Parameter(2*i, value=np.zeros(2 * i))
            eps_zeros_after = cp.Parameter(2 * (N-1-i), value=np.zeros(2 * (N-1-i)))
            stack = cp.hstack([eps_zeros_before, eps, eps_zeros_after])
        else:
            eps_zeros_before = cp.Parameter(2 * (N - 1), value=np.zeros(2 * (N - 1)))
            stack = cp.hstack([eps_zeros_before, eps])

        agent_stacks[i] = stack
        agent_eps[i] = eps

    return agent_stacks, agent_eps


# Distributed solve
global_solution = []
agent_solutions = {i: [] for i in range(N)}
agent_solutions_fairness = {i: [] for i in range(N)}


for iter in range(args.iter):
    # print('Iter {}'.format(iter))
    agent_stacks, agent_eps = generate_agent_variable_stacks(N)
    solved_values = np.zeros(n)
    for i in range(N):
        stack = agent_stacks[i]

        other_agent_vals = agents_prev_u[i].value
        other_agent_vals = cp.Parameter(n, value=other_agent_vals)
        agent_vals_my_change = other_agent_vals + stack

        # Gradient of Quadratic Objective
        Jobj = cp.Parameter(n, value = 2*np.dot(Q, agents_prev_u[i].value))

        # Gradient of Fairness Constraint (derivative of variance of energy)
        anrg = agents_prev_u[i].value[i*2:i*2+2]

        # fairness_partial = np.linalg.norm(anrg) - np.linalg.norm(np.mean(agents_prev_u[i].value.reshape(int(n/2), 2), axis=0))
        fairness_partial = np.linalg.norm(anrg) - np.mean(np.linalg.norm(agents_prev_u[i].value.reshape(int(n / 2), 2), axis=0))
        Jfair = cp.Parameter(value=((2/(N-1)) * fairness_partial))

        # Gradient of Obstacle Constraint (derivative of smooth min)
        denom = 0
        for j in range(N):
            denom += np.exp(-gamma * np.linalg.norm(agents_prev_u[i].value[j*2:j*2+2] - c) ** 2)

        aobs = 2 * -gamma ** 2 * agents_prev_u[i].value[i*2:i*2+2] * np.exp(
                -gamma * np.linalg.norm(agents_prev_u[i].value[i*2:i*2+2] - c))

        Jobs = cp.Parameter(2, value=aobs/denom)

        objective = cp.Minimize(-1 * (stack.T @ (Jobj + alpha*Jfair)) +
                                -1 * beta * agent_eps[i].T @ Jobs +
                                kappa * cp.norm(agent_eps[i] - agents_prev_eps[i]) ** 2
        )

        constraints = [agent_vals_my_change <= Ubox,
                       -Ubox <= agent_vals_my_change,
                       agent_eps[i] <= eps_bounds,
                       (-1 * eps_bounds) <= agent_eps[i]]

        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)
        if np.abs(prob.value) == np.inf:
                print('Agent {} Local Problem is {}'.format(i, prob.status))
                continue
        solved_values = solved_values + agent_stacks[i].value
        agent_solutions[i].append(prob.value)
        agent_solutions_fairness[i].append(Jfair.value)

    # global_u = init_u + solved_values
    global_u = agents_prev_u[0].value + solved_values
    main_obj = np.dot(global_u, np.dot(Q, global_u))

    u_reshape = global_u.reshape(int(n/2), 2)
    u_mean = np.mean(np.linalg.norm(u_reshape, axis=0))
    # u_mean = np.mean(u_reshape)
    diffs = 0
    logsum = 0
    for i in range(N):
        agents_prev_u[i] = cp.Parameter(n, value=global_u)
        agents_prev_eps[i] = agent_eps[i]

        diffs += (np.linalg.norm(u_reshape[i]) - u_mean)**2
        logsum += np.exp(-gamma * ((np.linalg.norm(u_reshape[i] - c) ** 2) - r*r))
    fairness_obj = 1/(N-1) * np.sum(diffs)

    obs_obj = -gamma * np.log(logsum)

    global_solution.append(main_obj + alpha*fairness_obj - beta*obs_obj)



# Plot to see convergence
plt.plot(global_solution)
plt.title('Centralized Solution')
plt.savefig('CentralSolution_N{}_alpha{}_beta{}_kappa{}_epsbounds{}.png'.format(N, alpha, beta, kappa, eps_bounds))

plt.clf()

for i in range(N):
    plt.plot(agent_solutions[i])

plt.title('Agent Solutions - Local Objective')
plt.savefig('AgentSolutions_N{}_alpha{}_beta{}_kappa{}_epsbounds{}.png'.format(N, alpha, beta, kappa, eps_bounds))

plt.clf()

for i in range(N):
    plt.plot(agent_solutions_fairness[i])

plt.title('Agent Solutions - Fairness Term')
plt.savefig('AgentSolutionsFairness_N{}_alpha{}_beta{}_kappa{}_epsbounds{}.png'.format(N, alpha, beta, kappa, eps_bounds))


print('init u')
print(init_u)

print('Init Obj')
u_reshape = init_u.reshape(int(n/2), 2)
u_mean = np.mean(u_reshape, axis=0)
diffs = 0
logsum = 0
for i in range(N):
    diffs += (np.linalg.norm(u_reshape[i]) - u_mean)**2
    logsum += np.exp(-gamma * ((np.linalg.norm(u_reshape[i] - c) ** 2) - r*r))
fairness_obj = 1/(N-1) * np.sum(diffs)

obs_obj = -gamma * np.log(logsum)
print(main_obj + alpha*fairness_obj - beta*obs_obj)


print('Final Objective Value')
print(global_solution[-1])

print('Final u')
print(global_u)
