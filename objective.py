import cvxpy as cp
import numpy as np
from scipy.optimize import Bounds, basinhopping, minimize, NonlinearConstraint
from generate_trajectories import generate_agent_states, generate_init_traj_quad

EPS = 1e-8
CP_SOLVER='ECOS'
SCIPY_SOLVER='SLSQP' #'L-BFGS-B' #

class Objective():
    def __init__(self, N, H, system_model_config, init_states, init_pos, obstacles, target,\
        Q, alpha, beta, gamma, kappa, eps_bounds, Ubox, dt=0.1, notion=0, safe_dist=0.1):
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
        self.safe_dist = safe_dist
        self.dt = dt

        self.heterogeneous = False
        self.with_penalty = True
        self.with_safety = False
        if self.heterogeneous:
            self.rn = []
            for r in range(N):
                if r % 2 == 0: 
                    self.rn.append(np.random.normal(0, 0.01))
                else:
                    self.rn.append(0)
        else:
            self.rn = [0 for r in range(N)]
        
        self.solo_energies = [1 for i in range(N)]
        
        self.stop_diff = 0.05
        self.stop = [0 for i in range(self.N)]
        self.notion = notion

    ##########################################################
    # Distributed Formulation
    ###########################################################

    def solve_distributed(self, init_u, steps=10, dyn='simple'):
        control_input_size = self.control_input_size
        
        init_eps = []
        local_sols = {}
        for i in range(self.N):
            init_eps.append(np.zeros(self.H * control_input_size))
            local_sols[i] = []
        prev_eps = init_eps
        prev_sols = list(local_sols.values())

        u = init_u
        fairness = []
        running_avgs = {i: [] for i in range(self.N)}
        stop_count = 0
        for s in range(steps):
            # print('Iter {}'.format(s))
            try:
                new_eps, new_sols = self.solve_local(u.flatten(), prev_eps, dyn=dyn)
                if len(new_eps) == 0:
                    new_eps = prev_eps
                    new_sols = prev_sols
                    # return [], [], [], []
                # print('Solutions', new_sols)
            except Exception as e:
                # print('Distributed Method Error at Iteration {}'.format(s))
                # print(e)
                # print(prev_eps)
                new_eps = prev_eps
                new_sols = prev_sols

            for i in range(self.N):
                u[i] += new_eps[i].reshape((self.H, control_input_size))
                local_sols[i].append(new_sols[i])

            fairness.append(self._fairness_central(u))
            prev_eps = new_eps
            prev_sols = new_sols

            # TODO: USE BETTER  CONVERGENCE CRITERIA
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
        F = self.N * self.H * control_input_size
        
        u_param = cp.Parameter(F, value=u)

        # Get the partial derivatives
        grad_quad = self.quad(u, grad=True).reshape(self.N, control_input_size * self.H)
        if self.notion in [3, 5]:
            grad_fairness = self.surge_fairness(u, grad=True)
        else:
            grad_fairness = self.fairness(u, grad=True)
        if self.with_safety:
            grad_obstacle = self.obstacle(u, grad=True, dyn=dyn)
            obst = self.obstacle(u, dyn=dyn)
            grad_avoid = self.avoid_constraint(u, grad=True, dyn=dyn)
            avoid = self.avoid_constraint(u, dyn=dyn)

        solved_values = []
        local_sols = []
        for i in range(self.N):
            curr_agent_u = u.reshape((self.N, self.H, control_input_size))[i].flatten()
            if self.notion in [0, 3]:  ## the basic fairness notion, uTQu + f1 (or uTQu + surge fairness)
                fairness_value = self.alpha * grad_quad[i] + self.alpha * grad_fairness[i]
            elif self.notion == 1:  # no fairness, uTQu only
                fairness_value = self.alpha * grad_quad[i]
            elif self.notion == 2:  # no fairness, no uTQu term
                fairness_value = np.zeros(self.H*control_input_size)
            else:  # f1 or f2 only  (ie self.notion in [4, 5])
                fairness_value = np.ones(self.H*control_input_size) * self.alpha * grad_fairness[i]
            
            if self.with_safety:
                if self.with_penalty:
                    grad = fairness_value + \
                        self.beta * (grad_obstacle[i] * self.penalty(obst, grad=True) + grad_avoid[i] * 2 * self.penalty(avoid, grad=True))
                else:
                    grad = fairness_value - self.beta * grad_obstacle[i] - self.beta * 2 * grad_avoid[i]
            else:
                grad = fairness_value

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
            # cg = np.append(target_center, np.array([0, 0, 0]))
            # rg = np.append(target_radius, np.array([0, 0, 0]))
            prev_state = self.init_states[i]
            if dyn =='quad':
                pos = prev_state[0:3]
                velo = prev_state[3:6]
                t = self.dt
                for j in range(self.H):
                    idx = j*control_input_size
                
                    accel = curr_agent_u[idx:idx+control_input_size] + eps[idx:idx+control_input_size]

                    pos = pos + velo*t + (1.0/2.0)*accel*(t**2)
                    velo = velo + accel*t
                # final_state = np.array([pos, velo]).flatten()
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
                cp.norm(final_pos - target_center) <= target_radius  # TODO: change this to include velocities
                # cp.norm(final_state - cg) <= rg
                ]

            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=False, solver=CP_SOLVER)
            # if prob.status == 'infeasible':
            #     # prob.solve(verbose=True, solver=CP_SOLVER)
            #     print('Agent {} Local Solution Infeasible'.format(i))
            #     # If a single agent's local solution is infeasible don't use any solution from this iteration, return empty team solution
            #     return [], []
            # solved_values.append(eps.value)
            # local_sols.append(prob.value)
            try:
                prob.solve(verbose=False, solver=CP_SOLVER)
                if prob.status == 'infeasible':
                    # prob.solve(verbose=True, solver=CP_SOLVER)
                    # print('Agent {} Local Solution Infeasible'.format(i))
                    # If a single agent's local solution is infeasible don't use any solution from this iteration, return empty team solution
                    return [], []
                solved_values.append(eps.value)
                local_sols.append(prob.value)
            except Exception as e:
                # Try to catch solver error. If an agent runs into solver error, catch and use previous agent's solution
                # print(e)
                # print('Agent {} Solver Error'.format(i))
                solved_values.append(prev_eps[i])
                local_sols.append(0)  # TODO: get solution from previous iteration

        return solved_values, local_sols
    
    ##########################################################
    # Central Formulation
    ###########################################################

    def solve_central(self, init_u, steps=200):
        func = self.central_obj
        x0 = init_u.flatten()

        constraints=[NonlinearConstraint(self.reach_constraint, -np.inf, 0)]

        if self.notion == 20:
            constraints.append(NonlinearConstraint(self.full_avoid_constraint, 0, np.inf))
        
        res = minimize(func, x0, bounds=Bounds(lb=-self.Ubox, ub=self.Ubox), 
                       constraints=constraints,
                       options={'maxiter':steps}, method=SCIPY_SOLVER)
        if not res.success:
            print(res.message)
            return np.inf, []
        final_u = res.x
        final_obj = res.fun
        # print(final_obj)

        return final_obj, final_u
    
    def central_obj(self, u):
        if self.notion == 0:  ## the basic fairness notion, uTQu + f1
            fairness_value = self.alpha * self.quad(u) + self.alpha * self.fairness(u)
        elif self.notion == 1:  ## no fairness, uTQu only
            fairness_value = self.alpha * self.quad(u)
        elif self.notion in [2, 20]:  # no fairness, no uTQu term
            fairness_value = 0
        elif self.notion == 3:  # use surge fairness 
            fairness_value =  self.alpha * self.quad(u) + self.alpha * self.surge_fairness(u)
        elif self.notion == 4:  #f1 only
            fairness_value = self.alpha * self.fairness(u)
        else:  # f2 only)
            fairness_value = self.alpha * self.surge_fairness(u)

        if self.with_safety:
            if self.with_penalty:
                return fairness_value + \
                    self.beta * (self.penalty(self.obstacle(u)) + 2*self.penalty(self.avoid_constraint(u)))
            else:
                return fairness_value - \
                    self.beta * self.obstacle(u) - \
                    self.beta * 2*self.avoid_constraint(u)
        else:
            return fairness_value

            
    ##########################################################
    # Reach Constraint
    ###########################################################

    def reach_constraint(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        target_center = self.target['center']
        target_radius = self.target['radius']
        num_agents = len(self.init_states)
        reach = -np.inf
        cg = np.append(target_center, np.array([0, 0, 0]))
        rg = np.append(target_radius, np.array([0, 0, 0]))  # TODO: wrong dim
        for i in range(num_agents):
            state_i, pos_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            final_state = state_i[len(state_i)-1]
            final_pos = pos_i[len(pos_i)-1]
            reach = np.maximum(reach, np.linalg.norm(final_pos - target_center) - target_radius)
            # reach = np.maximum(reach, np.linalg.norm(final_state - cg) - rg)
        return reach
        
    ##########################################################
    # Obstacle Avoidance Constraints
    ###########################################################

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
        
        return -1 / self.gamma * np.log(logsum + EPS)  # small EPS in case all distances to obstacle is very far, causing logsum to go to 0

    def _obstacle_local(self, u, dyn='simple'):
        g = np.array([
            [0.5*self.dt**2, 0, 0],
            [0, 0.5*self.dt**2, 0],
            [0, 0, 0.5*self.dt**2],
            [self.dt, 0, 0],
            [0, self.dt, 0],
            [0, 0, self.dt]])
        
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        c = self.obstacles['center']
        r = self.obstacles['radius']
        
        s = []
        p = []
        logsum = EPS
        for i in range(self.N):
            states, positions = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            states = states[1:]
            positions = positions[1:]
            distances_to_obstacle = np.linalg.norm(positions - c, axis=1)
            logsum += np.sum(np.exp(-1 * self.gamma * (
                distances_to_obstacle ** 2 - r**2
                )))
            s.append(states)
            p.append(positions)

        s = np.array(s)
        p = np.array(p)

        partials = []
        for i in range(self.N):
            states = s[i]
            positions = p[i]
            distances_to_obstacle = np.linalg.norm(positions - c, axis=1)
            partial_smoothmin = np.sum(np.exp(-1 * self.gamma * (distances_to_obstacle ** 2 - r**2))) / (logsum + EPS)
            partial_loss = 2 * np.linalg.norm(np.dot(g.T, states.T))
            partial = np.multiply(partial_smoothmin, partial_loss)
            partials.append(partial.flatten())

        return partials
        

    ##########################################################
    # Mutual Separation Constraints
    ###########################################################

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
            for j in range(i+1, self.N):
                if j == i:
                    continue
                _, positions_j = generate_agent_states(u_reshape[j], self.init_states[j], self.init_pos[j], model=self.system_model, dt=self.dt)
                positions_j = positions_j[1:]
            
                distances = np.linalg.norm(positions_i - positions_j, axis=1)
                logsum += np.sum(np.exp(-1 * self.gamma * (distances ** 2 - self.safe_dist**2)))
        
        return -1 / self.gamma * np.log(logsum + EPS)  # small EPS in case all distances to obstacle is very far, causing logsum to go to 0)

    def _avoid_local(self, u, dyn='simple'):
        g = np.array([
            [0.5*self.dt**2, 0, 0],
            [0, 0.5*self.dt**2, 0],
            [0, 0, 0.5*self.dt**2],
            [self.dt, 0, 0],
            [0, self.dt, 0],
            [0, 0, self.dt]])
        
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        
        s = []
        p = []
        logsum = EPS
        for i in range(self.N):
            states_i, positions_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            states_i = states_i[1:]
            positions_i = positions_i[1:]
            for j in range(i+1, self.N):
                if j == i:
                    continue
                _, positions_j = generate_agent_states(u_reshape[j], self.init_states[j], self.init_pos[j], model=self.system_model, dt=self.dt)
                positions_j = positions_j[1:]
            
                distances = np.linalg.norm(positions_i - positions_j, axis=1)
                logsum += np.sum(np.exp(-1 * self.gamma * (distances ** 2 - self.safe_dist**2)))
            
            s.append(states_i)
            p.append(positions_i)

        s = np.array(s)
        p = np.array(p)

        partials = []
        for i in range(self.N):
            states_i = s[i]
            positions_i = p[i]

            total_distances = 0
            for j in range(i+1, self.N):
                if j == i:
                    continue
                positions_j = p[j]
                distances_to_obstacle = np.linalg.norm(positions_i - positions_j, axis=1)
                total_distances += np.sum(distances_to_obstacle ** 2 - self.safe_dist**2)
            
            partial_smoothmin = np.exp(-1 * self.gamma * total_distances) / (logsum+EPS)
            
            partial_loss = 2 * np.linalg.norm(np.dot(g.T, states_i.T))
            partial = np.multiply(partial_smoothmin, partial_loss)
            partials.append(partial.flatten())

        return partials
    
    ##########################################################
    # Fairness Notions
    ###########################################################

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
        agent_sum_energies = np.sum(np.linalg.norm(u_reshape, axis=2)**2, axis=1) / np.array(self.solo_energies)
        mean_energy = np.mean(agent_sum_energies)
        diffs = 0
        for i in range(self.N):
            diffs += (np.linalg.norm(agent_sum_energies[i] - mean_energy)) ** 2
    
        fairness = 1/(self.N) * np.sum(diffs)
        return fairness

    def _fairness_local(self, u):
        control_input_size = self.control_input_size

        u_reshape = u.reshape((self.N, self.H, control_input_size))
        agent_sum_energies = np.sum(np.linalg.norm(u_reshape, axis=2)**2, axis=1) / np.array(self.solo_energies)
        mean_energy = np.mean(agent_sum_energies)
        partials = []
        for i in range(self.N):
            grad = 2 * (1/self.N) * (np.linalg.norm(agent_sum_energies[i] - mean_energy))
            partials.append(grad)
            
        return partials

    def surge_fairness(self, u, grad=False):
        if grad:
            f = self._surge_fairness_local(u)
            return f
        else:
            return self._surge_fairness_central(u)

    # TODO: NORMALIZE SURGE FAIRNESS
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
    
    ##########################################################
    # Combined Avoidance Constraints
    ###########################################################
    
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
    

    ##########################################################
    # Check that Mission Successful after Solving
    ###########################################################

    def check_avoid_constraints(self, u, avoid_only=False):
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
            distances_to_obstacle = np.linalg.norm(positions - c, axis=1) + 0.001
            if any(distances_to_obstacle < r):
                return 1
            if not avoid_only:
                distance_to_target = np.linalg.norm(final_p - cg) - 0.001
                if distance_to_target > rg:
                    return 3

        # Check Collision Avoidance
        for i in range(self.N):
            _, positions_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            positions_i = positions_i[1:]
            for j in range(i+1, self.N):
                _, positions_j = generate_agent_states(u_reshape[j], self.init_states[j], self.init_pos[j], model=self.system_model, dt=self.dt)
                positions_j = positions_j[1:]
                distances_to_obstacle = np.linalg.norm(positions_i - positions_j, axis=1) + 0.001
                if any(distances_to_obstacle < self.safe_dist):
                    return 2
        return 0
    

    ##########################################################
    # Seed Solution (Obstacle Avoidance and Mutual Separation Only)
    ###########################################################

    def solve_nbf(self, seed_u=None, final_pos=None, mpc=False):
        # Centrally Solve for Obstacle Avoidance
        f = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
        f_diag = []
        for i in range(self.N):
            f_diag_list = []
            for j in range(self.N):
                if i == j:
                    f_diag_list.append(f)
                else:
                    f_diag_list.append(np.zeros((6, 6)))
            f_diag.append(f_diag_list)
        f_diag = np.block(f_diag)
        g = np.array([
            [0.5*self.dt**2, 0, 0],
            [0, 0.5*self.dt**2, 0],
            [0, 0, 0.5*self.dt**2],
            [self.dt, 0, 0],
            [0, self.dt, 0],
            [0, 0, self.dt]])
        g_diag = []
        for i in range(self.N):
            g_diag_list = []
            for j in range(self.N):
                if i == j:
                    g_diag_list.append(g)
                else:
                    g_diag_list.append(np.zeros_like(g))
            g_diag.append(g_diag_list)
        g_diag = np.block(g_diag)
        
        c = self.obstacles['center']
        r = self.obstacles['radius']
        cg = self.target['center']
        rg = self.target['radius']

        u_t = cp.Variable(self.N*self.control_input_size)
        alpha = cp.Variable(1)
        
        # Create Quad systems for each 
        robots = []
        for r in range(self.N):
            rob = self.system_model(self.init_states[r], dt=self.dt)
            robots.append(rob)

        if seed_u is not None:
            u_fair_t = cp.Parameter((self.N*self.control_input_size), 
                                    value=np.zeros(self.N*self.control_input_size))
            objective = cp.Minimize(cp.sum_squares(u_t - u_fair_t) + alpha**2)
        else:
            u_ref_t = cp.Parameter((self.N*self.control_input_size), 
                                   value=np.zeros(self.N*self.control_input_size))
            objective = cp.Minimize(cp.sum_squares(u_t - u_ref_t))
        final_u = []
        for t in range(self.H):
            u_refs_fair = []
            u_refs = []
            target_pos = []
            state_collect = []
            constraints = []
            for r in range(self.N):
                state_collect.append(robots[r].state)
                if seed_u is not None:
                    u_refs_fair.append(seed_u[r, t, :])
                    continue
                kx = 0.7
                kv = 1.8 if self.N < 10 else 1.6 if self.N in [10, 15] else 1.5
                rn = self.rn[r]
                leftright = 1 if r % 2 == 0 else -1
                x_adj = 1 if r % 2 == 0 else 0
                z_adj = 1 if (r % 3 == 0) and self.N >= 10 else 0
                pos_adj = np.array([x_adj, 1, z_adj]) 
                velocity_desired = - kx * ( robots[r].state[0:3] - (self.target['center'] + leftright*r*self.safe_dist*pos_adj) )
                u_desired = - kv * ( robots[r].state[3:6] - velocity_desired )
                u_refs.append(u_desired)
                target = robots[r].state[0:3] + (rn + robots[r].state[3:6]*self.dt) + 0.5*u_desired*self.dt**2
                target_pos.append(target)

            if seed_u is not None:
                u_fair_t.value = np.array(u_refs_fair).flatten()
            else:
                u_ref_t.value = np.array(u_refs).flatten()
            
            # NBFs
            h_c_min = 9999
            h_cs = []
            A = []
            h_o_min = 9999
            h_os = []
            B = []
            V_max = -9999
            Vs = []
            C = []
            for r in range(self.N):
                # obstacle avoidance
                pos = robots[r].state[0:3]
                h_o = (pos[0] - c[0])**2 + (pos[1] - c[1])**2 + (pos[2] - c[2])**2 - r**2
                # h_os.append(1*h_o**3)
                h_os.append(h_o)
                h_o_min = np.min([h_o_min, h_o])
                Brow_start = [0 for i in range(r*6)]
                Brow_end = [0 for i in range((r+1)*6, self.N*6)]
                Brow = Brow_start + [2*(pos[0] - c[0]), 2*(pos[1] - c[1]), 2*(pos[2] - c[2]), 0, 0, 0] + Brow_end
                B.append(Brow)

                # reach goal  # TODO: CHANGE V TO INCLUDE RADIUS?
                V = (pos[0] - cg[0])**2 + (pos[1] - cg[1])**2 + (pos[2] - cg[2])**2 #+ \
                    #   robots[r].state[3]**2 + robots[r].state[4]**2 + robots[r].state[5]**2
                Vs.append(V)
                # Vs.append(V**3)
                V_max = np.max([V_max, V])
                Crow_start = [0 for i in range(r*6)]
                Crow_end = [0 for i in range((r+1)*6, self.N*6)]
                Crow = Crow_start + [2*(pos[0] - cg[0]), 2*(pos[1] - cg[1]), 2*(pos[2] - cg[2]), 0, 0, 0] + Crow_end
                                    #  2*robots[r].state[3], 2*robots[r].state[4], 2*robots[r].state[5]] + Crow_end
                C.append(Crow)

                # mutual separation
                for s in range(r+1, self.N):
                    pos1 = robots[s].state[0:3]
                    h_c = (pos[0] - pos1[0])**2 + (pos[1] - pos1[1])**2 + (pos[2] - pos1[2])**2 - self.safe_dist**2
                    # h_cs.append(1*h_c**3)
                    h_cs.append(h_c)
                    h_c_min = np.min([h_c_min, h_c])
                    deriv = [2*(pos[0] - pos1[0]), 2*(pos[1] - pos1[1]), 2*(pos[2] - pos1[2]), 0, 0, 0]
                    nderiv = [-2*(pos[0] - pos1[0]), -2*(pos[1] - pos1[1]), -2*(pos[2] - pos1[2]), 0, 0, 0]
                    Arow_start = [0 for i in range(r*6)]
                    Arow_mid = [0 for i in range((r+1)*6, s*6)]
                    Arow_end = [0 for i in range((s+1)*6, self.N*6)]
                    Arow = Arow_start + deriv + Arow_mid + nderiv + Arow_end
                    A.append(Arow)

            if self.N == 3:
                V_alpha = 1
                h_gamma = 2
                h_e = 1
            elif self.N == 5:
                V_alpha = 1
                h_gamma = 2 #5
                h_e = 1
            elif self.N == 7:
                V_alpha = 1
                h_gamma = 50 # 50
                h_e = 2
            elif self.N == 10:
                V_alpha = 1
                h_gamma = 20
                h_e = 2
            elif self.N == 15:
                V_alpha = 1
                h_gamma = 20
                h_e = 2
            else:
                V_alpha = 1
                h_gamma = 25
                h_e = 2

            h_min = np.min([h_c_min, h_o_min])
            B = np.array(B)
            Lfh1 = np.dot(np.dot(f_diag, np.array(state_collect).flatten()), B.T) 
            Lgh1_u = np.dot(B, g_diag) @ u_t
            constraints.append(Lfh1 + Lgh1_u + h_gamma*h_min**h_e >= 0) 

            C = np.array(C)
            LfV = np.dot(np.dot(f_diag, np.array(state_collect).flatten()), C.T) 
            LgV_u = np.dot(C, g_diag) @ u_t
            constraints.append(LfV + LgV_u + V_alpha*V_max <= alpha) 

            if self.N > 1:
                A = np.array(A)
                Lfh2 = np.dot(np.dot(f_diag, np.array(state_collect).flatten()), A.T) 
                Lgh2_u = np.dot(A, g_diag) @ u_t
                constraints.append(Lfh2 + Lgh2_u + h_gamma*h_min**h_e >= 0) 

            cbf_controller = cp.Problem(objective, constraints)
            cbf_controller.solve(solver=CP_SOLVER)

            if cbf_controller.status in ['infeasible', 'infeasible_inaccurate']:
                # print(f"QP infeasible")
                # cbf_controller.solve(solver=CP_SOLVER, verbose=True)
                # print(cbf_controller)
                return []
            
            for r in range(self.N):
                idx_start = r*self.control_input_size
                idx_end = idx_start + self.control_input_size
                _, _ = robots[r].forward(u_t.value[idx_start:idx_end])
            final_u.append(u_t.value.reshape(self.N, self.control_input_size))
            if mpc:
                break
            # print(alpha.value)
        return final_u, h_min, V_max
    
    ##########################################################
    # Penalty Function for Online Solution
    ###########################################################

    def penalty(self, x, grad=False):
        alpha = 1.
        if grad:
            # grad p(x) = exp(ax)/(e^ax + 1)
            return np.exp(alpha*x)/(np.exp(alpha*x) + 1)
        
        # p(x) = 1/smoothmax(0, x)
        # smoothmax(x1, ..., xn) = 1/a ln (e^ax1 + ... + e^axn)
        smax =  1/alpha * np.log(np.exp(alpha*x) + 1)
        return 1/(smax + EPS)
        