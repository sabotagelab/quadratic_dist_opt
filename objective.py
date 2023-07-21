import cvxpy as cp
import numpy as np
from scipy.optimize import Bounds, basinhopping, minimize, NonlinearConstraint
from generate_trajectories import generate_agent_states

EPS = 1e-8

class Objective():
    def __init__(self, N, H, system_model_config, init_states, init_pos, obstacles, target,\
        Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=0.1, notion=0):
        self.N = N
        self.H = H
        self.system_model = system_model_config[0]
        self.control_input_size = system_model_config[1]
        self.init_states = init_states
        self.init_pos = init_pos
        self.obstacles = obstacles  # only a single obstacle
        self.target = target  # only a single target
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa
        self.eps_bounds = eps_bounds
        self.Ubox = Ubox
        self.safe_dist = 0.1
        self.dt = dt
        
        self.solo_energies = [1 for i in range(self.N)]
        
        self.stop_diff = 0.05
        self.stop = [0 for i in range(self.N)]
        self.notion = notion

    def solve_distributed(self, init_u, steps=10, dyn='simple'):
        control_input_size = self.control_input_size
        
        init_eps = []
        local_sols = {}
        for i in range(self.N):
            init_eps.append(np.zeros(self.H * control_input_size))
            local_sols[i] = []
        prev_eps = init_eps

        u = init_u
        fairness = []
        running_avgs = {i: [] for i in range(self.N)}
        stop_count = 0
        for s in range(steps):
            # print('Iter {}'.format(s))
            new_eps, sols = self.solve_local(u.flatten(), prev_eps, dyn=dyn)
            if len(new_eps) == 0:
                return [], [], []

            for i in range(self.N):
                u[i] += new_eps[i].reshape((self.H, control_input_size))
                local_sols[i].append(sols[i])

            fairness.append(self._fairness_central(u))
            prev_eps = new_eps

            for i in range(self.N):
                curr_avg = np.mean(local_sols[i])
                running_avgs[i].append(curr_avg)
                if s > 1:
                    last_avg = running_avgs[i][s-2]
                    if np.abs((curr_avg - last_avg)/last_avg) < self.stop_diff:
                        self.stop[i] = 1
            if np.sum(self.stop) > (0.75 * self.N):
                break

        return u, local_sols, fairness, s


    def solve_local(self, u, prev_eps, dyn='simple'):
        control_input_size = self.control_input_size
        state_size = self.init_states[0].shape
        F = self.N * self.H * control_input_size
        
        u_param = cp.Parameter(F, value=u)

        # Get the partial derivatives
        grad_quad = self.quad(u, grad=True).reshape(self.N, control_input_size * self.H)
        if self.notion in [3, 5]:
            grad_fairness = self.surge_fairness(u, grad=True)
        else:
            grad_fairness = self.fairness(u, grad=True)
        grad_obstacle = self.obstacle(u, grad=True, dyn=dyn)
        grad_avoid = self.avoid_constraint(u, grad=True, dyn=dyn)


        solved_values = []
        local_sols = []
        for i in range(self.N):
            curr_agent_u = u.reshape((self.N, self.H, control_input_size))[i].flatten()
            if self.notion == 0:  ## the basic fairness notion, uTQu + f1
                grad = self.alpha * grad_quad[i] + self.alpha * grad_fairness[i] - self.beta * grad_obstacle[i] - self.beta * grad_avoid[i]
            elif self.notion == 1:  # no fairness, uTQu only
                grad = self.alpha * grad_quad[i] + - self.beta * grad_obstacle[i] - self.beta * grad_avoid[i]
            elif self.notion == 2:  # no fairness, no uTQu term
                grad = - self.beta * grad_obstacle[i] - self.beta * grad_avoid[i]
            elif self.notion == 3:  # surge fairness
                grad = self.alpha * grad_quad[i] + self.alpha * grad_fairness[i] - self.beta * grad_obstacle[i] - self.beta * grad_avoid[i]
            else:  # f1 or f2 only
                grad = self.alpha * grad_fairness[i] - self.beta * grad_obstacle[i] - self.beta * grad_avoid[i]
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

            # define constraint on final position based on eps and init state param
            target_center = self.target['center']
            target_radius = self.target['radius']
            prev_state = self.init_states[i]
            if dyn =='quad':
                pos = prev_state[0:3]
                velo = prev_state[3:6]
                t = self.dt
                for j in range(self.H):
                    idx = j*control_input_size
                    prev_state = prev_state.flatten()
                
                    accel = curr_agent_u[idx:idx+control_input_size] + eps[idx:idx+control_input_size]

                    pos = pos + velo*t + (1.0/2.0)*accel*(t**2)
                    velo = velo + accel*t

                final_pos = pos
            else:
                # assuming simple dynamics
                for j in range(self.H):
                    idx = j*control_input_size
                    new_state = prev_state.flatten() + \
                        2*(curr_agent_u[idx:idx+control_input_size] + \
                            eps[idx:idx+control_input_size])
                    prev_state = new_state
                final_pos = prev_state

            # define local objective
            objective = cp.Minimize(-1 * (eps.T @ grad_param) + \
                self.kappa * cp.norm(eps - prev_eps_param)**2 )

            # define local constraints
            constraints = [
                u_param + stack <= self.Ubox, \
                -self.Ubox <= u_param + stack,
                eps <= self.eps_bounds,
                -1 * self.eps_bounds <= eps,
                cp.norm(final_pos - target_center) <= target_radius
                ]

            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=False)
            if prob.status == 'infeasible':
                print('Agent {} Problem Status {}'.format(i, prob.status))
                return [], []
            solved_values.append(eps.value)
            local_sols.append(prob.value)

        return solved_values, local_sols

    def solve_central(self, init_u, steps=200):
        func = self.central_obj
        x0 = init_u.flatten()

        constraints=[NonlinearConstraint(self.reach_constraint, -np.inf, 0)]

        if self.notion == 20:
            constraints.append(NonlinearConstraint(self.full_avoid_constraint, 0, np.inf))
        
        res = minimize(func, x0, bounds=Bounds(lb=-self.Ubox, ub=self.Ubox), 
                       constraints=constraints,
                       options={'maxiter':steps})
        if not res.success:
            print(res.message)
            return np.inf, []
        final_u = res.x
        final_obj = res.fun

        return final_obj, final_u

    def reach_constraint(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        target_center = self.target['center']
        target_radius = self.target['radius']
        num_agents = len(self.init_states)
        reach = -np.inf
        for i in range(num_agents):
            _, pos_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            final_pos = pos_i[len(pos_i)-1]
            reach = np.maximum(reach, np.linalg.norm(final_pos - target_center) - target_radius)
        return reach
    
    def full_avoid_constraint(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        obstacle_center = self.obstacles['center']
        obstacle_radius = self.obstacles['radius']
        num_agents = len(self.init_states)
        avoid = np.inf
        for i in range(num_agents):
            _, pos_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            final_pos = pos_i[len(pos_i)-1]
            avoid = np.minimum(avoid, np.linalg.norm(final_pos - obstacle_center) - obstacle_radius)
            pos_i = pos_i[1:]
            for j in range(i+1,num_agents):
                _, positions_j = generate_agent_states(u_reshape[j], self.init_states[j], self.init_pos[j], model=self.system_model, dt=self.dt)
                positions_j = positions_j[1:]
                distances = np.linalg.norm(pos_i - positions_j, axis=1)
                min_distance = np.min(distances)
                avoid = np.minimum(avoid, min_distance - self.safe_dist)
        return avoid

    
    def central_obj(self, u):
        if self.notion == 0:  ## the basic fairness notion, uTQu + f1
            return self.alpha * self.quad(u) + \
                self.alpha * self.fairness(u) - \
                self.beta * self.obstacle(u) - \
                self.beta * self.avoid_constraint(u)
        elif self.notion == 1:  ## no fairness, uTQu only
            return self.alpha * self.quad(u) - \
                self.beta * self.obstacle(u) - \
                self.beta * self.avoid_constraint(u)
        elif self.notion in [2, 20]:  # no fairness, no uTQu term
            # return 0
            return - self.beta * self.obstacle(u) - self.beta * self.avoid_constraint(u)
        elif self.notion == 3:  # use surge fairness 
            return self.alpha * self.quad(u) + \
                self.alpha * self.surge_fairness(u) - \
                self.beta * self.obstacle(u) - \
                self.beta * self.avoid_constraint(u)
        elif self.notion == 4:  #f1 only
            return self.alpha * self.fairness(u) - \
                self.beta * self.obstacle(u) - \
                self.beta * self.avoid_constraint(u)
        else:  # f2 only
            return self.alpha * self.surge_fairness(u) - \
                self.beta * self.obstacle(u) - \
                self.beta * self.avoid_constraint(u)

    def avoid_constraint(self, u, grad=False, dyn='simple'):
        if grad:
            return self._avoid_local(u, dyn=dyn)
        else:
            return self._avoid_central(u)

    def _avoid_central(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        
        logsum = 0
        for i in range(self.N):
            _, positions_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            positions_i = positions_i[1:]
            for j in range(i, self.N):
                _, positions_j = generate_agent_states(u_reshape[j], self.init_states[j], self.init_pos[j], model=self.system_model, dt=self.dt)
                positions_j = positions_j[1:]
            
                distances = np.linalg.norm(positions_i - positions_j, axis=1)
                logsum += np.sum(np.exp(-1 * self.gamma * (distances ** 2 - self.safe_dist**2)))
        
        return -1 * self.gamma * np.log(logsum + EPS)  # small EPS in case all distances to obstacle is very far, causing logsum to go to 0)

    def _avoid_local(self, u, dyn='simple'):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        
        x = []
        logsum = EPS
        for i in range(self.N):
            _, positions_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            positions_i = positions_i[1:]
            for j in range(i, self.N):
                _, positions_j = generate_agent_states(u_reshape[j], self.init_states[j], self.init_pos[j], model=self.system_model, dt=self.dt)
                positions_j = positions_j[1:]
            
                distances = np.linalg.norm(positions_i - positions_j, axis=1)
                logsum += np.sum(np.exp(-1 * self.gamma * (distances ** 2 - self.safe_dist**2)))
            
            x.append(positions_i)

        x = np.array(x)

        partials = []
        for i in range(self.N):
            positions_i = x[i]

            total_distances = 0
            for j in range(i, self.N):
                positions_j = x[j]
                distances_to_obstacle = np.linalg.norm(positions_i - positions_j, axis=1)
                total_distances += np.sum(distances_to_obstacle ** 2 - self.safe_dist**2)
            
            partial_smoothmin = np.exp(-1 * self.gamma * total_distances) / (logsum+EPS)
            
            if dyn == 'simple':
                system_partial = 2 * np.ones_like(u_reshape[1])
            else:
                system_partial = 2 * self.dt * u_reshape[i]
            p = np.multiply(partial_smoothmin * 2 * distances_to_obstacle, system_partial.T)
            partials.append(p.flatten())

        return partials


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
        agent_sum_energies = np.sum(np.linalg.norm(u_reshape, axis=2)**2, axis=1)
        mean_energy = np.mean(agent_sum_energies)
        diffs = 0
        for i in range(self.N):
            # NORMALIZE THE NORM BY AGENT SOLO ENERGY
            diffs += (np.linalg.norm(u_reshape[i])/self.solo_energies[i] - mean_energy/self.solo_energies[i]) ** 2
    
        fairness = 1/(self.N) * np.sum(diffs)
        return fairness

    def _fairness_local(self, u):
        control_input_size = self.control_input_size

        u_reshape = u.reshape((self.N, self.H, control_input_size))
        agent_sum_energies = np.sum(np.linalg.norm(u_reshape, axis=2)**2, axis=1)
        mean_energy = np.mean(agent_sum_energies)
        partials = []
        for i in range(self.N):
            # NORMALIZE THE NORM BY AGENT SOLO ENERGY
            grad = 2 * (1/self.N) * (np.linalg.norm(u_reshape[i])/self.solo_energies[i] - mean_energy/self.solo_energies[i])
            partials.append(grad)
            
        return partials

    def surge_fairness(self, u, grad=False):
        if grad:
            f = self._surge_fairness_local(u)
            return f
        else:
            return self._surge_fairness_central(u)

    def _surge_fairness_central(self, u):
        control_input_size = self.control_input_size

        u_reshape = u.reshape((self.N, self.H, control_input_size))
        energies = np.linalg.norm(u_reshape, axis=2)**2
        agent_mean_energies = np.mean(energies, axis=1)
        surges = np.diff(energies)
        surges = surges - np.min(surges) / (np.max(surges) - np.min(surges))
        surge_thresh = np.mean(surges) + np.std(surges)

        agent_total_over_surge = []
        for i in range(self.N):
            agent_total_over_surge.append(np.sum(surges[i] - surge_thresh))
    
        fairness = np.var(agent_total_over_surge)
        return fairness

    def _surge_fairness_local(self, u):
        control_input_size = self.control_input_size

        u_reshape = u.reshape((self.N, self.H, control_input_size))
        energies = np.linalg.norm(u_reshape, axis=2)**2
        agent_mean_energies = np.mean(energies, axis=1)
        surges = np.diff(energies)
        surges = surges - np.min(surges) / (np.max(surges) - np.min(surges))
        surge_thresh = np.mean(surges) + np.std(surges)

        agent_total_over_surge = []
        sk_div = []
        for i in range(self.N):
            agent_total_over_surge.append(np.sum(surges[i] - surge_thresh))
            sk_div.append(2*(np.linalg.norm(u_reshape[i][self.H-1]) - np.linalg.norm(u_reshape[i][0])))
        
        mean_agent_surge = np.mean(agent_total_over_surge)
        partials = []
        for i in range(self.N):
            grad = 2 * (1/self.N) * (agent_total_over_surge[i] - mean_agent_surge) * sk_div[i]
            partials.append(grad)   

        return partials 

    def obstacle(self, u, grad=False, dyn='simple'):
        if grad:
            return self._obstacle_local(u, dyn=dyn)
        else:
            return self._obstacle_central(u)

    def _obstacle_central(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        c = self.obstacles['center']
        r = self.obstacles['radius']
        
        logsum = 0
        for i in range(self.N):
            _, positions = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            positions = positions[1:]
            distances_to_obstacle = np.linalg.norm(positions - c, axis=1)
            logsum += np.sum(np.exp(-1 * self.gamma * (
                distances_to_obstacle ** 2 - r**2
                )))
        
        return -1 * self.gamma * np.log(logsum + EPS)  # small EPS in case all distances to obstacle is very far, causing logsum to go to 0

    def _obstacle_local(self, u, dyn='simple'):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        c = self.obstacles['center']
        r = self.obstacles['radius']
        
        x = []
        logsum = EPS
        for i in range(self.N):
            _, positions = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            positions = positions[1:]
            distances_to_obstacle = np.linalg.norm(positions - c, axis=1)
            logsum += np.sum(np.exp(-1 * self.gamma * (
                distances_to_obstacle ** 2 - r**2
                )))
            
            x.append(positions)

        x = np.array(x)

        partials = []
        for i in range(self.N):
            positions = x[i]
            distances_to_obstacle = np.linalg.norm(positions - c, axis=1)
            partial_smoothmin = np.sum(np.exp(-1 * self.gamma * (distances_to_obstacle ** 2 - r**2))) / (logsum + EPS)
            if dyn == 'simple':
                system_partial = 2 * np.ones_like(u_reshape[1])
            else:
                system_partial = 2 * self.dt * u_reshape[i]
            p = np.multiply(partial_smoothmin * 2 * distances_to_obstacle, system_partial.T)
            partials.append(p.flatten())

        return partials
    

    def _full_avoid_local(self, u, dyn='simple'):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        c = self.obstacles['center']
        r = self.obstacles['radius']
        
        x = []
        logsum = EPS
        for i in range(self.N):
            _, positions_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            positions_i = positions_i[1:]

            distances_to_obstacle = np.linalg.norm(positions_i - c, axis=1)
            logsum += np.sum(np.exp(-1 * self.gamma * (
                distances_to_obstacle ** 2 - r**2
                )))

            for j in range(i, self.N):
                _, positions_j = generate_agent_states(u_reshape[j], self.init_states[j], self.init_pos[j], model=self.system_model, dt=self.dt)
                positions_j = positions_j[1:]
            
                inter_distances = np.linalg.norm(positions_i - positions_j, axis=1)
                logsum += np.sum(np.exp(-1 * self.gamma * (inter_distances ** 2 - self.safe_dist**2)))
            
            x.append(positions_i)

        x = np.array(x)

        partials = []
        for i in range(self.N):
            positions_i = x[i]

            distances_to_obstacle = np.linalg.norm(positions_i - c, axis=1)

            total_distances = np.sum(np.exp(-1 * self.gamma * (distances_to_obstacle ** 2 - r**2)))
            for j in range(i, self.N):
                positions_j = x[j]
                inter_distances = np.linalg.norm(positions_i - positions_j, axis=1)
                total_distances += np.sum(inter_distances ** 2 - self.safe_dist**2)
            
            partial_smoothmin = np.exp(-1 * self.gamma * total_distances) / (logsum+EPS)
            
            if dyn == 'simple':
                system_partial = 2 * np.ones_like(u_reshape[1])
            else:
                system_partial = 2 * self.dt * u_reshape[i]
            p = np.multiply(partial_smoothmin * 2 * distances_to_obstacle, system_partial.T)
            partials.append(p.flatten())

        return partials

    def check_avoid_constraints(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))

        # Check That All agents avoid Obstacle AND REACH GOAL
        c = self.obstacles['center']
        r = self.obstacles['radius']
        cg = self.target['center']
        rg = self.target['radius']
        for i in range(self.N):
            _, positions = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            final_p = positions[self.H]
            positions = positions[1:]
            distances_to_obstacle = np.linalg.norm(positions - c, axis=1)
            if any(distances_to_obstacle < r):
                return False
            distance_to_target = np.linalg.norm(final_p - cg) - 0.001
            if distance_to_target > rg:
                print('doesnt reach')
                print(i, distance_to_target)
                return False

        # Check Collision Avoidance
        for i in range(self.N):
            _, positions_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            positions_i = positions_i[1:]
            for j in range(i+1, self.N):
                _, positions_j = generate_agent_states(u_reshape[j], self.init_states[j], self.init_pos[j], model=self.system_model, dt=self.dt)
                positions_j = positions_j[1:]
                distances_to_obstacle = np.linalg.norm(positions_i - positions_j, axis=1)
                if any(distances_to_obstacle < self.safe_dist):
                    print('collision')
                    print(distances_to_obstacle)
                    return False

        return True
    