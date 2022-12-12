import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
m, n = 6, 6

Q = np.random.randn(n, n)   # variable for quadratic objective
Q = Q.T @ Q

a = np.random.rand(n)

alpha = 1   # parameter for fairness constraint
beta = 1    # parameter for weighting of obstacle avoidance constraint
gamma = 1   # parameter for smoothmin in calculating obstacle avoidance constraint
kappa = 1   # parameter for weighting change in epsilon for local problem
eps_bounds = 1  # bounds for eps in an iteration

r = 5  # radius of circle
c = np.array([1, 1])  # center of circle
Ubox = 10  # box constraint


# print('Central problem')
# c_stack = np.array([1, 1, 1, 1, 1, 1])  # center of circle replicated 3 times
# u = cp.Variable(n)
# u_min = cp.Variable(2)
# objective = cp.Minimize(cp.quad_form(u, Q) + \
# (alpha/n) * cp.sum_squares(u - cp.sum(u)/n) + \
#  a.T @ u )#- \
 #-(cp.square(cp.norm(2*u[0:2] - c)) - r*r))
#  -1/gamma * cp.log(cp.exp(-gamma*(cp.square(cp.norm(2*u[0:2] - c)) - r*r)) +
# cp.exp(-gamma*(cp.square(cp.norm(2*u[2:4] - c)) - r*r)) +
# cp.exp(-gamma*(cp.square(cp.norm(2*u[4:] - c)) - r*r))) )

# constraints = [B @ x <= b]
# constraints = [u <= Ubox, -Ubox <= u]
# prob = cp.Problem(objective, constraints)
# prob.solve(solver='CVXOPT', verbose=True)
# print("\nThe optimal value is", prob.value)
# print("A solution u is")
# print(u.value)
# print(np.dot(u.value, np.dot(Q, u.value)))



print('Distributed Problem, 3 Agents')
agents_prev_eps = {0: cp.Parameter(2, value=np.zeros(2)),
1: cp.Parameter(2, value=np.zeros(2)),
2: cp.Parameter(2, value=np.zeros(2))
}

init_u = np.random.randn(n)
agents_prev_u = {0: cp.Parameter(6, value=np.copy(init_u)),
1: cp.Parameter(6, value=np.copy(init_u)),
2: cp.Parameter(6, value=np.copy(init_u)),
}


def generate_agent_variable_stacks():
    """

    :return:
        agent_stacks, dictionary mapping agents to their local u parameter
        agent_eps, dictionary mapping agents to their local eps variable

    """
    agent_stacks = {}
    agent_eps = {}

    # agent 1 variable
    eps1 = cp.Variable(2)
    eps1_zeros = cp.Parameter(4, value=np.zeros(4))
    agent_stacks[0] = cp.hstack([eps1, eps1_zeros])
    agent_eps[0] = eps1

    # agent 2 variable
    eps2 = cp.Variable(2)
    eps2_zeros1 = cp.Parameter(2, value=np.zeros(2))
    eps2_zeros2 = cp.Parameter(2, value=np.zeros(2))
    agent_stacks[1] =  cp.hstack([eps2_zeros1, eps2, eps2_zeros2])
    agent_eps[1] = eps2

    # agent 3 variable
    eps3 = cp.Variable(2)
    eps3_zeros1 = cp.Parameter(4, value=np.zeros(4))
    agent_stacks[2] =  cp.hstack([eps3_zeros1, eps3])
    agent_eps[2] = eps3

    return agent_stacks, agent_eps


# Distributed solve
global_solution = []
agent_solutions = {0: [], 1: [], 2: []}
for iter in range(10):
    print('Iter {}'.format(iter))
    agent_stacks, agent_eps = generate_agent_variable_stacks()
    solved_values = np.zeros(6)
    for i in range(3):
        stack = agent_stacks[i]

        other_agent_vals = agents_prev_u[i].value
        other_agent_vals = cp.Parameter(6, value=other_agent_vals)
        agent_vals_my_change = other_agent_vals + stack

        # Gradient of Quadratic Objective
        Jobj = cp.Parameter(n, value = 2*np.dot(Q, agents_prev_u[i].value))

        # Gradient of Fairness Constraint (derivative of variance of energy)
        if i == 0:
            anrg = agents_prev_u[i].value[0:2]
        elif i == 1:
            anrg = agents_prev_u[i].value[2:4]
        else:
            anrg = agents_prev_u[i].value[4:]
        fairness_partial = np.linalg.norm(anrg) - np.linalg.norm(np.mean(agents_prev_u[i].value.reshape(3, 2), axis=0))
        Jfair = cp.Parameter(value=((2/(n-1)) * fairness_partial))


        # Gradient of Obstacle Constraint (derivative of smooth min)
        denom = np.exp(-gamma * np.linalg.norm(agents_prev_u[i].value[0:2] - c) ** 2) + \
                np.exp(-gamma * np.linalg.norm(agents_prev_u[i].value[2:4] - c) ** 2) +\
                np.exp(-gamma * np.linalg.norm(agents_prev_u[i].value[4:] - c) ** 2)
        if i == 0:
            # aobs = 2* -gamma**2 * agents_prev_u[i].value[0:2] * \
            #        np.exp(-gamma * np.linalg.norm(agents_prev_u[i].value[0:2] - c))
            aobs = 2* -gamma**2  * \
                   np.exp(-gamma * np.linalg.norm(agents_prev_u[i].value[0:2] - c))
        elif i == 1:
            aobs = 2 * -gamma ** 2 * np.exp(
                -gamma * np.linalg.norm(agents_prev_u[i].value[2:4] - c))
        else:
            aobs = 2 * -gamma ** 2 * np.exp(
                -gamma * np.linalg.norm(agents_prev_u[i].value[4:] - c))
        Jobs = cp.Parameter(value=aobs/denom)

        objective = cp.Minimize(-1 * (stack.T @ (Jobj + alpha*Jfair + a + beta*Jobs)) +
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

    for i in range(3):
        agents_prev_u[i] = cp.Parameter(6, value=init_u + np.copy(solved_values))
        agents_prev_eps[i] = agent_eps[i]

    # print('Iter {} Global Solution'.format(iter))
    # print(solved_values)
    # print(np.dot(solved_values, np.dot(Q, solved_values)))
    global_u = init_u + solved_values
    global_solution.append(np.dot(global_u, np.dot(Q, global_u))
                           + alpha * (1/(n-1)) * np.sum(np.linalg.norm(global_u) - np.linalg.norm(np.mean(global_u)))
                           - beta * -gamma * np.log(np.exp(-gamma * np.linalg.norm(global_u[0:2] - c) ** 2) +
                                                   np.exp(-gamma * np.linalg.norm(global_u[2:4] - c) ** 2) +
                                                   np.exp(-gamma * np.linalg.norm(global_u[4:] - c) ** 2)
                                                   )
                           )
    # global_solution.append(prob.value)
    # print(np.dot(A, solved_values) <= b)

# print('Final')
# print(solved_values)
# print(prob.value)
# print(np.dot(init_u + solved_values, np.dot(Q, init_u + solved_values)))

# Plot to see convergence
plt.plot(global_solution)
plt.title('Centralized Solution')
plt.show()

plt.clf()

plt.plot(agent_solutions[0])
plt.plot(agent_solutions[1])
plt.plot(agent_solutions[2])
plt.title('Agent Solutions')
plt.show()
