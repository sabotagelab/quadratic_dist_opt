import cvxpy as cp
import numpy as np
from scipy.optimize import basinhopping
from generate_trajectories import generate_agent_states

EPS = 1e-8

class Objective():
    def __init__(self, N, H, system_model_config, init_states, obstacles, target,\
        Q, alpha, beta, gamma, kappa, eps_bounds, Ubox):
        self.N = N
        self.H = H
        self.system_model = system_model_config[0]
        self.control_input_size = system_model_config[1]
        self.init_states = init_states
        self.obstacles = obstacles  # only a single obstacle
        self.target = target  # only a single target
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa
        self.eps_bounds = eps_bounds
        self.Ubox = Ubox
        self.UBoxBounds = UBoxBounds(Ubox)
        self.TakeStep = TakeStep(eps_bounds)

    def solve_distributed(self, init_u, steps=10):
        control_input_size = self.control_input_size
        
        init_eps = []
        local_sols = {}
        for i in range(self.N):
            init_eps.append(np.zeros(self.H * control_input_size))
            local_sols[i] = []
        prev_eps = init_eps

        u = init_u
        fairness = []
        for s in range(steps):
            new_eps, sols = self.solve_local(u.flatten(), prev_eps)
            for i in range(self.N):
                u[i] += new_eps[i].reshape((self.H, control_input_size))
                local_sols[i].append(sols[i])

            fairness.append(self._fairness_central(u))
            prev_eps = new_eps

        return u, local_sols, fairness


    def solve_local(self, u, prev_eps):
        control_input_size = self.control_input_size
        state_size = self.init_states[0].shape
        F = self.N * self.H * control_input_size
        
        u_param = cp.Parameter(F, value=u)

        # Get the partial derivatives
        grad_quad = self.quad(u, grad=True).reshape(self.N, control_input_size * self.H)
        grad_fairness = self.fairness(u, grad=True)
        grad_obstacle = self.obstacle(u, grad=True)

        solved_values = []
        local_sols = []
        for i in range(self.N):
            init_state_param = cp.Parameter(state_size, value = self.init_states[i])
            grad = grad_quad[i] + self.alpha * grad_fairness[i] - self.beta * grad_obstacle[i]
            grad_param = cp.Parameter(self.H * control_input_size, value=grad)
            prev_eps_param = cp.Parameter(self.H * control_input_size, value=prev_eps[i])

            # create decision variable
            eps = cp.Variable(self.H * control_input_size)
            if i == 0:
                eps_zeros_after = cp.Parameter(F - (self.H * control_input_size), \
                    value=np.zeros(F - (self.H * control_input_size)))
                stack = cp.hstack([eps, eps_zeros_after])
            elif i < self.N - 1:
                eps_zeros_before = cp.Parameter(self.H * control_input_size*i, \
                    value=np.zeros(self.H * control_input_size * i))
                eps_zeros_after = cp.Parameter(self.H * control_input_size * (self.N-1-i), \
                    value=np.zeros(self.H * control_input_size * (self.N-1-i)))
                stack = cp.hstack([eps_zeros_before, eps, eps_zeros_after])
            else:
                eps_zeros_before = cp.Parameter(F - (self.H * control_input_size), \
                    value=np.zeros(F - (self.H * control_input_size)))
                stack = cp.hstack([eps_zeros_before, eps])

            # define local objective
            # objective = cp.Minimize(-1 * (stack.T @ grad_param) + \
            #     self.kappa * cp.norm(eps - prev_eps_param)**2)
            objective = cp.Minimize(-1 * (eps.T @ grad_param) + \
                self.kappa * cp.norm(eps - prev_eps_param)**2)

            # define local constraints
            constraints = [
                u_param + stack <= self.Ubox, \
                -self.Ubox <= u_param + stack,
                eps <= self.eps_bounds,
                -1 * self.eps_bounds <= eps
                ]

            # TODO: define constraints on position based on eps and init state param
            # for j in range(self.H):


            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=False)
            solved_values.append(eps.value)
            local_sols.append(prob.value)

        return solved_values, local_sols

    # TODO: add option to use custom-built simulated annealing
    def solve_central(self, init_u, steps=200):
        init_trajectories = []
        for i in range(self.N):
            traj = generate_agent_states(init_u[i], self.init_states[i], model=self.system_model)
            init_trajectories.append(traj)

        func = self.central_obj
        x0 = init_u.flatten()
        # res = basinhopping(func, x0, niter=steps, take_step=self.TakeStep, accept_test=self.UBoxBounds, callback=print_fun)
        # res = basinhopping(func, x0, niter=steps, accept_test=self.UBoxBounds, callback=print_fun)
        res = basinhopping(func, x0, niter=steps, take_step=self.TakeStep, accept_test=self.UBoxBounds)
        final_u = res.x
        final_obj = res.fun

        return final_obj, final_u
    
    def central_obj(self, u):
        return self.quad(u) + \
            self.alpha * self.fairness(u) - \
                 self.beta * self.obstacle(u)

    def quad(self, u, grad=False):
        if grad:
            return 2 * np.dot(self.Q, u)
        else:
            return np.dot(u, np.dot(self.Q, u))

    def fairness(self, u, grad=False):
        if grad:
            f = self._fairness_local(u)
            return f
        else:
            return self._fairness_central(u)
    
    def _fairness_central(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        
        init_trajectories = []
        for i in range(self.N):
            traj = generate_agent_states(u_reshape[i], self.init_states[i], model=self.system_model)
            init_trajectories.append(traj)

        u_reshape = u.reshape((self.N, control_input_size * self.H))
        mean_energy = np.mean(np.linalg.norm(u_reshape, axis=0))
        diffs = 0
        for i in range(self.N):
            diffs += (np.linalg.norm(u_reshape[i]) - mean_energy) ** 2
    
        fairness = 1/(self.N) * np.sum(diffs)
        return fairness

    def _fairness_local(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))        
        init_trajectories = []
        for i in range(self.N):
            traj = generate_agent_states(u_reshape[i], self.init_states[i], model=self.system_model)
            init_trajectories.append(traj[1:])

        u_reshape = u.reshape((self.N, control_input_size * self.H))
        mean_energy = np.mean(np.linalg.norm(u_reshape, axis=0))
        partials = []
        for i in range(self.N):
            grad = 2 * (1/self.N) * (np.linalg.norm(u_reshape[i]) - mean_energy)
            grad_positions = np.gradient(init_trajectories[i], axis=0)  # May have to take system derivative manually
            partials.append(grad * grad_positions.flatten())
            
        
        return partials

    def obstacle(self, u, grad=False):
        if grad:
            return self._obstacle_local(u)
        else:
            return self._obstacle_central(u)

    def _obstacle_central(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        c = self.obstacles['center']
        r = self.obstacles['radius']
        
        logsum = 0
        for i in range(self.N):
            positions = generate_agent_states(u_reshape[i], self.init_states[i], model=self.system_model)
            positions = positions[1:]
            distances_to_obstacle = np.linalg.norm(positions - c)
            logsum += np.exp(-1 * self.gamma * (
                distances_to_obstacle ** 2 - r**2
                ))
        
        return -1 * self.gamma * np.log(logsum + EPS)  # small EPS in case all distances to obstacle is very far, causing logsum to go to 0

    def _obstacle_local(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        c = self.obstacles['center']
        r = self.obstacles['radius']
        
        x = []
        logsum = 0
        for i in range(self.N):
            positions = generate_agent_states(u_reshape[i], self.init_states[i], model=self.system_model)
            positions = positions[1:]
            distances_to_obstacle = np.linalg.norm(positions - c)
            logsum += np.exp(-1 * self.gamma * (
                distances_to_obstacle ** 2 - r**2
                ))
            
            x.append(positions)

        x = np.array(x)

        partials = []
        for i in range(self.N):
            positions = x[i]
            distances_to_obstacle = np.linalg.norm(positions - c)
            partial_smoothmin = np.exp(-1 * self.gamma * distances_to_obstacle ** 2 - r**2) / logsum
            system_partial = np.gradient(positions, axis=0)  # May have to take system derivative manually
            p = partial_smoothmin * 2 * distances_to_obstacle * system_partial
            partials.append(p.flatten())

        return partials


class UBoxBounds():
    # Bounds for basinhopping, https://het.as.utexas.edu/HET/Software/Scipy/generated/scipy.optimize.basinhopping.html
    def __init__(self, Ubox):
        self.umax = Ubox
        self.umin = -1 * Ubox
    
    def __call__(self, **kwargs):
        x = kwargs['x_new']
        tmax = bool(np.all(x <= self.umax))
        tmin = bool(np.all(x >= self.umin))
        teps = bool(np.all(np.abs(x) >= 0.1))
        # print(tmax and tmin and teps)
        return tmax and tmin and teps


class TakeStep():
    def __init__(self, stepsize):
        self.stepsize = stepsize
    
    def __call__(self, x):
        s = self.stepsize
        random_change = np.random.uniform(low=s, high=1, size=x.shape)
        sign = np.random.choice([-1, 1])
        new_x = x + sign * random_change 
        # print(new_x)
        return new_x

def print_fun(x, f, accepted):
    print("at minima %.4f accepted %d" % (f, int(accepted)))
