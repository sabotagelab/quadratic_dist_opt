import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
m, n = 6, 6

Q = np.random.randn(n, n)
Q = Q.T @ Q

A = np.random.rand(m, n)
A = A.T @ A

B = np.random.rand(m, n)

b = np.random.randn(m)
c = np.random.randn(1)

print('Central problem with linear constraint')
x = cp.Variable(n)
objective = cp.Minimize(cp.quad_form(x, Q) + cp.sum_squares(x - cp.sum(x)/n))
constraints = [B @ x <= b]
prob = cp.Problem(objective, constraints)
prob.solve()
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print(np.dot(x.value, np.dot(Q, x.value)))

# print('Central problem with quadratic constraint')
# x = cp.Variable(n)
# objective = cp.Minimize(cp.quad_form(x, Q))
# constraints = [cp.quad_form(x, A) <= c, B @ x <= b]
# prob = cp.Problem(objective, constraints)
# prob.solve()
# print("\nThe optimal value is", prob.value)
# print("A solution x is")
# print(x.value)
# print(np.dot(x.value, np.dot(Q, x.value)))


print('Distributed')
alpha = 10
eps_bounds = 1
# init_x = np.random.randn(n)
init_x = x.value
agents_prev_sol = {0: cp.Parameter(2, value=np.zeros(2)),
1: cp.Parameter(2, value=np.zeros(2)),
2: cp.Parameter(2, value=np.zeros(2))
# 3: cp.Parameter(2, value=np.zeros(2)),
# 4: cp.Parameter(2, value=np.zeros(2))
}

agents_prev_x = {0: cp.Parameter(6, value=np.copy(init_x)),
1: cp.Parameter(6, value=np.copy(init_x)),
2: cp.Parameter(6, value=np.copy(init_x)),
# 3: cp.Parameter(10, value=np.copy(init_x)),
# 4: cp.Parameter(10, value=np.copy(init_x))
}


def generate_agent_variable_stacks():
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
    # eps3_zeros2 = cp.Parameter(4, value=np.zeros(4))
    # agent_stacks[2] =  cp.hstack([eps3_zeros1, eps3, eps3_zeros2])
    agent_stacks[2] =  cp.hstack([eps3_zeros1, eps3])
    agent_eps[2] = eps3

    # # agent 4 variable
    # eps4 = cp.Variable(2)
    # eps4_zeros1 = cp.Parameter(6, value=np.zeros(6))
    # eps4_zeros2 = cp.Parameter(2, value=np.zeros(2))
    # agent_stacks[3] = cp.hstack([eps4_zeros1, eps4, eps4_zeros2])
    # agent_eps[3] = eps4
    #
    # # agent 4 variable
    # eps5 = cp.Variable(2)
    # eps5_zeros = cp.Parameter(8, value=np.zeros(8))
    # agent_stacks[4] = cp.hstack([eps5_zeros, eps5])
    # agent_eps[4] = eps5

    return agent_stacks, agent_eps

# Distributed solve
global_solution = []
for iter in range(10):
    print('Iter {}'.format(iter))
    agent_stacks, agent_eps = generate_agent_variable_stacks()
    solved_values = np.zeros(6)
    for i in range(3):
        stack = agent_stacks[i]

        other_agent_vals = agents_prev_x[i].value
        other_agent_vals = cp.Parameter(6, value=other_agent_vals)
        agent_vals_my_change = other_agent_vals + stack

        J = cp.Parameter(n, value = 2*np.dot(Q, agents_prev_x[i].value))

        objective = cp.Minimize(-1 * (stack.T @ (J +  \
        # new term for variance of energy
        2/(n-1) * cp.sum(agents_prev_x[i] - cp.sum(agents_prev_x[i])/n))) + \
        alpha * cp.norm(agent_eps[i] - agents_prev_sol[i]) ** 2)
        #constraints = [cp.quad_form(agent_vals_my_change, A) <= c, agent_eps[i] <= eps_bounds, (-1 * eps_bounds) <= agent_eps[i]]
        constraints = [B @ agent_vals_my_change <= b, agent_eps[i] <= eps_bounds, (-1 * eps_bounds) <= agent_eps[i]]
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)
        if np.abs(prob.value) == np.inf:
                print('Agent {} Local Problem is {}'.format(i, prob.status))
                continue
        # print('Agent {} Local Solution'.format(i))
        # print(agent_eps[i].value)
        # print(agent_stacks[i].value)
        solved_values = solved_values + agent_stacks[i].value
        # print(solved_values)

    for i in range(3):
        agents_prev_x[i] = cp.Parameter(6, value=np.copy(solved_values))
        agents_prev_sol[i] = agent_eps[i]

    # print('Iter {} Global Solution'.format(iter))
    # print(solved_values)
    # print(np.dot(solved_values, np.dot(Q, solved_values)))
    global_solution.append(np.dot(solved_values, np.dot(Q, solved_values)))
    # print(np.dot(A, solved_values) <= b)

print('Final')
print(solved_values)
print(prob.value)
print(np.dot(solved_values, np.dot(Q, solved_values)))
print(np.dot(B, solved_values) <= b)

# Plot to see convergence
plt.plot(global_solution)
plt.show()
