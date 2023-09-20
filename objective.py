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
        self.with_safety = False
        
        self.solo_energies = [1 for i in range(N)]
        
        self.stop_diff = 0.05
        self.stop = [0 for i in range(self.N)]
        self.notion = notion

        f = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
        self.f = f
        f_diag = []
        for i in range(self.N):
            f_diag_list = []
            for j in range(self.N):
                if i == j:
                    f_diag_list.append(f)
                else:
                    f_diag_list.append(np.zeros((6, 6)))
            f_diag.append(f_diag_list)
        self.f_diag = np.block(f_diag)

        g = np.array([
            [0.5*self.dt**2, 0, 0],
            [0, 0.5*self.dt**2, 0],
            [0, 0, 0.5*self.dt**2],
            [self.dt, 0, 0],
            [0, self.dt, 0],
            [0, 0, self.dt]])
        self.g = g
        g_diag = []
        for i in range(self.N):
            g_diag_list = []
            for j in range(self.N):
                if i == j:
                    g_diag_list.append(g)
                else:
                    g_diag_list.append(np.zeros_like(g))
            g_diag.append(g_diag_list)
        self.g_diag = np.block(g_diag)

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
            cg = np.append(target_center, np.array([0, 0, 0]))
            rg = np.array([target_radius, target_radius, target_radius, 0, 0, 0])
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
                # cp.abs(final_state - cg) <= rg
                ]

            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=False, solver=CP_SOLVER)
            try:
                prob.solve(verbose=False, solver=CP_SOLVER)
                if prob.status == 'infeasible':
                    # prob.solve(verbose=True, solver=CP_SOLVER)
                    print('Agent {} Local Solution Infeasible'.format(i))
                    # If a single agent's local solution is infeasible don't use any solution from this iteration, return empty team solution
                    return [], []
                solved_values.append(eps.value)
                local_sols.append(prob.value)
            except Exception as e:
                # Try to catch solver error. If an agent runs into solver error, catch and use previous agent's solution
                print(e)
                print('Agent {} Solver Error'.format(i))
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
    # Online Solution (Obstacle Avoidance and Mutual Separation Only)
    ###########################################################

    def solve_nbf(self, last_alpha=None, seed_u=None, mpc=False):
        # Centrally Solve for Obstacle Avoidance
        f = self.f
        f_diag = self.f_diag
        g_diag = self.g_diag
        
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

                # reach goal
                V = (pos[0] - cg[0])**2 + (pos[1] - cg[1])**2 + (pos[2] - cg[2])**2 - (rg/2)**2 #+ \
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
                h_gamma = 1
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

            if last_alpha is not None:
                constraints.append(alpha <= last_alpha)

            # Ubox constraint
            constraints.append(-1 * self.Ubox <= u_t)
            constraints.append(u_t <= self.Ubox)

            cbf_controller = cp.Problem(objective, constraints)
            cbf_controller.solve(solver=CP_SOLVER)

            if cbf_controller.status in ['infeasible', 'infeasible_inaccurate']:
                print(f"QP infeasible")
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
        return final_u, h_min, V_max, alpha.value[0]
        

    ##########################################################
    # Distributed Version of Online Solution (MPC only)
    ###########################################################
    def solve_distributed_nbf(self, seed_input, last_alphas, h_gamma=1, h_e=1, v_gamma=1, v_e=1):

        # Keep only first time step of seed_input
        seed_input = seed_input[:, 0, :]

        # CALCULATE TARGET POSITIONS FOR ALL AGENTS BASED ON SEED INPUT
        agent_target_pos = []
        for i in range(self.N):
            _, target_pos = generate_agent_states(seed_input[i], self.init_states[i], self.init_pos[i], 
                                                  self.system_model, dt=self.dt)
            agent_target_pos.append(target_pos[0])
        
        # INTIALIZE LOCAL CONSTRAINTS
        agent_gammas = []
        agent_Ais = []
        agent_bis = []
        for i in range(self.N):
            if last_alphas is not None:
                g, Ai, bi = self.init_local_constraints(i, agent_target_pos, last_alphas[i], h_gamma=1, h_e=1, v_gamma=1, v_e=1)
            else:
                g, Ai, bi = self.init_local_constraints(i, agent_target_pos, None, h_gamma=1, h_e=1, v_gamma=1, v_e=1)
            agent_gammas.append(g)
            agent_Ais.append(Ai)
            agent_bis.append(bi)
        
        m = Ai.shape[0]
        # SHARE GAMMAS and Ais BETWEEN AGENTS TO DO THE TRADES ALGORITHM
        uis = np.copy(seed_input)
        lambdas = [np.ones(m) for i in range(self.N)]
        zis = [0 for i in range(self.N)]
        yis = [0 for i in range(self.N)]

        # TODO: repeat below until convergence, auxiliary variables in ui will converge to match the neighbors control inputs in all ujs
        phi, agent_phi_i = self.trades_phi(uis, agent_target_pos, grad=False, return_all=True)  # lowercase phi
        for i in range(self.N):
            ui_p, lambda_p, zi_p, yi_p = self.trades(i, uis, seed_input, agent_target_pos[i], agent_gammas, agent_Ais, agent_bis, agent_phi_i, lambdas, zis, yis)
            # update uis, lambdas, zis, yis

    
    def init_local_constraints(self, agent_id, target_poses, last_alpha, h_gamma=1, h_e=1, v_gamma=1, v_e=1):
        # Calculate Target Position Coordinates Based on Seed Input
        # Compute Error Dynamics
        # Compute Auxiliary variables with for all other agents

        # Get Target Position Coordinates Based on Seed Input
        agent_actual_vel = self.init_states[agent_id][0:3]
        agent_target_pos = target_poses[agent_id][0]
        
        # Compute Error Dynamics
        agent_error_dyn = self.init_states[agent_id][0:3] - agent_target_pos

        # Compute Auxiliary variables with for all other agents
        gammas = []
        Ai = []
        for j in range(self.N):
            if j == agent_id:
                continue
            actual_dist = self.init_pos[agent_id] - self.init_pos[j]
            
            neighbor_actual_vel = self.init_states[j][0:3]
            neighbor_target_pos = target_poses[j][0]
            neighbor_error_dyn = self.init_states[j][0:3] - neighbor_target_pos
            # Compute Deltas
            delta_p = agent_target_pos - neighbor_target_pos
            delta_v = agent_actual_vel - neighbor_actual_vel
            
            # Compute tij 
            hij = np.sum(np.abs(agent_error_dyn - neighbor_error_dyn + delta_p) / self.safe_dist - 1)
            tij = 2/(self.dt**2) * h_gamma * (hij**h_e) + \
                2/self.dt*(1/self.safe_dist*np.sum(np.sign(actual_dist) * delta_v))
            gammas.append(tij)
            Ai.append(-1 * np.sign(actual_dist))

        # append tij for static obstacles as well (delta_v is 0 and self.safe_dist == obstacle radius)
        obj_actual_dist = self.init_pos[agent_id] - self.obstacles['center']
        delta_obj_p = agent_target_pos - self.obstacles['center']
        hi_obj = np.sum(np.abs(agent_error_dyn + delta_obj_p) / self.obstacles['radius'] - 1)
        ti_obj = 2/(self.dt**2) * h_gamma * (hi_obj**h_e) + \
            2/self.dt*(1/self.obstacles['radius']*np.sum(np.sign(obj_actual_dist) * agent_actual_vel))
        gammas.append(ti_obj)
        Ai.append(-1 * np.sign(obj_actual_dist))
        
        # append tij for the lyapunov function (similar to static obstacle case but negative because we want to go to target)
        target_actual_dist = self.init_pos[agent_id] - self.target['center']
        delta_target_p = agent_target_pos - self.target['center']
        if last_alpha is not None:
            vi_target = -1 * np.sum((np.abs(agent_error_dyn + delta_target_p) / self.target['radius'] - 1 - last_alpha))
        else:
            vi_target = -1 * np.sum((np.abs(agent_error_dyn + delta_target_p) / self.target['radius'] - 1))
        ti_target = 2/(self.dt**2) * v_gamma * (vi_target**v_e) + \
            2/self.dt*(1/self.target['radius']*np.sum(np.sign(target_actual_dist) * agent_actual_vel))
        gammas.append(ti_target)
        Ai.append(np.sign(target_actual_dist))

        Ai = np.array(Ai)
        
        n_p = np.array(gammas).size
        Ai = np.append(Ai, np.ones((Ai.shape[0], 1)), axis=1)
        Ai = np.append(Ai, np.zeros((Ai.shape[0], n_p - 1)), axis=1)
        bi = np.zeros((Ai.shape[0], 1))

        return gammas, Ai, bi

    def trades(self, agent_id, current_inputs, seed_input, agent_target_pos, agent_gammas, Ais, bis, phi,
               lambdas, zis, yis, step_size=0.01, trade_param=0.8):
        # Implement Algorithm 1 from paper 
        # (but for horizon = 1, that is, returned ui for each agent is what they should do for next time step, 
        # no future timesteps computed)

        # let network parameter wij be 1/N-1 (ie one over all neighbors) for all neighbors
        wij = 1 / (self.N - 1)
        rho = 1.1 * step_size * trade_param / wij

        ui = current_inputs[agent_id]
        
        # TODO: remove gammas from function and MOVE the ui_full creation to outside, let F_i just take the subset of ui
        # the now to create full decision variable (append the auxiliary variables)
        aux_vars = agent_gammas[agent_id]
        ui_full = np.hstack([ui, aux_vars])

        F_i = self.trades_F_i(agent_id, ui_full[0:3], seed_input[agent_id], agent_target_pos, phi[agent_id] + zis[agent_id])
        G_ui = self.trades_G_i_primal(ui_full, Ais[agent_id], bis[agent_id] + yis[agent_id], lambdas[agent_id], rho)
        G_lambi = self.trades_G_i_primal(ui_full, Ais[agent_id], bis[agent_id] + yis[agent_id], lambdas[agent_id], rho)
        ui_p = ui_full + trade_param * (self.projection(ui_full - step_size * F_i - step_size * G_ui) - ui_full)
        lambda_p = 0
        zi_p = 0
        yi_p = 0
        for i in range(self.N):
            if i == agent_id:
                continue
            uj = current_inputs[i]
            uj_full = np.hstack([uj, agent_gammas[i]])
            lambda_p += wij * lambdas[i] + trade_param * step_size * G_lambi
            zi_p += wij * zis[i] + wij * phi[i]
            yi_p += wij * yis[i] + wij * self.N * (np.dot(Ais[i], uj_full) - bis[i]) 

        zi_p = zi_p - phi[agent_id]
        yi_p = yi_p - self.N * (np.dot(Ais[i], ui_full) - bis[i]) 
        return ui_p, lambda_p, zi_p, yi_p



    def trades_phi_i(self, agent_id, agent_proposed_input, agent_target_pos, grad=False):
        # small_phi_i penalizes the agent's fraction of aggregate distance to target position for this timestep
        _, actual_new_pos = generate_agent_states(agent_proposed_input, 
                                                  self.init_states[agent_id], 
                                                  self.init_pos[agent_id], 
                                                  self.system_model, dt=self.dt)
        actual_new_pos = actual_new_pos[0]
        if grad:
            # return 2 * np.linalg.norm(actual_new_pos - agent_target_pos) * np.dot(self.g.T, self.init_states[agent_id].T)
            return 2 * np.linalg.norm(actual_new_pos - agent_target_pos) * np.linalg.norm(np.dot(self.g.T, self.init_states[agent_id].T))
        else:
            return np.linalg.norm(actual_new_pos - agent_target_pos)**2
        
    def trades_phi(self, proposed_inputs, target_positions, grad=False, return_all=False):
        if grad:
            partials = []
            for i in range(self.N):
                partials.append(1/self.N * self.trades_phi_i(i, proposed_inputs[i], target_positions[i], grad=True))
            return partials
        else:
            total_phi_i = 0
            all_phis = []
            for i in range(self.N):
                phi_i = self.trades_phi_i(i, proposed_inputs[i], target_positions[i], grad=False)
                total_phi_i += phi_i
                all_phis.append(phi_i)
            if not return_all:
                return 1/self.N * total_phi_i
            else:
                return 1/self.N * total_phi_i, all_phis

    def trades_J_i(self, agent_proposed_input, agent_seed_input, phi, grad=False):
        # OBJECTIVE FUNCTION, MINIMIZE DISTANCE TO SEED_INPUTS (THE FAIR INPUTS)
        if grad:
            partial_1 = 2 * np.linalg.norm(agent_proposed_input - agent_seed_input) + phi  # here phi is partial of phi for agent i
            partial_2 = 1
            return partial_1, partial_2
        else:
            return np.linalg.norm(agent_proposed_input - agent_seed_input)**2 + phi
        
    def trades_F_i(self, agent_id, agent_proposed_input, agent_seed_input, agent_target_pos, phi):
        # pseudo gradient of J_i
        partial_1_J_i, partial_2_j_i = self.trades_J_i(agent_proposed_input, agent_seed_input, phi, grad=True)  # here phi is partial of phi for agent i
        partial_phi_i = self.trades_phi_i(agent_id, agent_proposed_input, agent_target_pos, grad=True)
        return partial_1_J_i + (partial_phi_i / self.N) * partial_2_j_i

    def trades_G_i_primal(self, agent_proposed_input, Ai, bi, lambda_i, rho):
        # primal psuedo gradient of constraints
        m = Ai.shape[0]
        s1 = self.N * (np.dot(Ai, agent_proposed_input) - bi)
        s2 = lambda_i

        res = 0
        for i in range(m):
            res += np.maximum(np.dot(rho * s1[i] + s2[i], Ai.T[i]), 0)
        
        return res

    def trades_G_i_dual(self, agent_proposed_input, Ai, bi, lambda_i, rho):
        # dual psuedo gradient of constraints
        e = Ai.T
        m = Ai.shape[0]

        s1 = self.N * (np.dot(Ai, agent_proposed_input) - bi)
        s2 = lambda_i

        res = 0
        for i in range(m):
            res += np.maximum(np.dot(rho * s1[i] + s2[i], e[i]) - s2[i], - s2[i])
        return


    def projection(self, v):
        # TODO: project v into Ubox
        return v
