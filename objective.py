import cvxpy as cp
import numpy as np
from scipy.optimize import Bounds, basinhopping, minimize, NonlinearConstraint
from generate_trajectories import generate_agent_states, generate_init_traj_quad

EPS = 1e-8
CP_SOLVER='ECOS' # 'MOSEK' #
SCIPY_SOLVER='SLSQP' #'L-BFGS-B' #

class Objective():
    def __init__(self, N, H, system_model_config, init_states, init_pos, obstacles, targets, \
        starts, Q, alpha, kappa, eps_bounds, Ubox, dt=0.1, notion=0, safe_dist=0.1):
        self.N = N
        self.H = H
        self.system_model = system_model_config[0]
        self.control_input_size = system_model_config[1]
        self.init_states = init_states
        self.init_pos = init_pos
        self.obstacles = obstacles
        self.targets = targets
        self.Q = Q
        self.alpha = alpha
        self.kappa = kappa
        self.eps_bounds = eps_bounds
        self.Ubox = Ubox
        self.safe_dist = safe_dist
        self.dt = dt
        self.starts = starts
        self.goals_made = 0
        self.dist_to_goal = []

        self.heterogeneous = False
        
        self.solo_energies = [1 for i in range(N)]
        
        # Convergence criteria for distributed fair planner
        self.stop_diff = 0.05  
        self.stop = [0 for i in range(self.N)]
        
        self.notion = notion

        # Dynamics
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
                # fairness_value = self.alpha * grad_quad[i] + self.alpha * grad_fairness[i]
                fairness_value = self.alpha * grad_quad[i] + grad_fairness[i]
            elif self.notion == 1:  # no fairness, uTQu only
                fairness_value = self.alpha * grad_quad[i]
            elif self.notion == 2:  # no fairness, no uTQu term
                fairness_value = np.zeros(self.H*control_input_size)
            else:  # f1 or f2 only  (ie self.notion in [4, 5])
                # fairness_value = np.ones(self.H*control_input_size) * self.alpha * grad_fairness[i]
                fairness_value = np.ones(self.H*control_input_size) * grad_fairness[i]

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
            target_center = self.targets[i]['center']
            target_radius = self.targets[i]['radius']
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
                cp.norm(final_pos - target_center) <= target_radius  # TODO: change this to include target velocities 0?
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

        return final_obj, final_u
    
    def central_obj(self, u):
        if self.notion == 0:  ## the basic fairness notion, uTQu + f1
            # fairness_value = self.alpha * self.quad(u) + self.alpha * self.fairness(u)
            fairness_value = self.alpha * self.quad(u) + self.fairness(u)
        elif self.notion == 1:  ## no fairness, uTQu only
            fairness_value = self.alpha * self.quad(u)
        elif self.notion in [2, 20]:  # no fairness, no uTQu term
            fairness_value = 0
        elif self.notion == 3:  # use surge fairness 
            # fairness_value = self.alpha * self.quad(u) + self.alpha * self.surge_fairness(u)
            fairness_value =  self.alpha * self.quad(u) + self.surge_fairness(u)
        elif self.notion == 4:  #f1 only
            # fairness_value = self.alpha * self.fairness(u)
            fairness_value = self.fairness(u)
        else:  # f2 only)
            # fairness_value = self.alpha * self.surge_fairness(u)
            fairness_value = self.surge_fairness(u)

        return fairness_value

            
    ##########################################################
    # Reach Constraint
    ###########################################################

    def reach_constraint(self, u):
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        num_agents = len(self.init_states)
        reach = -np.inf
        for i in range(num_agents):
            state_i, pos_i = generate_agent_states(u_reshape[i], self.init_states[i], self.init_pos[i], model=self.system_model, dt=self.dt)
            final_pos = pos_i[len(pos_i)-1]
            target_center = self.targets[i]['center']
            target_radius = self.targets[i]['radius']
            # TODO: change this to include target velocities 0?
            reach = np.maximum(reach, np.linalg.norm(final_pos - target_center) - target_radius)
            
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
        # agent_norm_energies = np.sum(np.linalg.norm(u_reshape, axis=2)**2, axis=1) / (np.array(self.solo_energies) + EPS)
        agent_norm_energies = (np.linalg.norm(u_reshape, axis=(1,2))**2) / (np.array(self.solo_energies) + EPS)
        mean_energy = np.mean(agent_norm_energies)
        mean_energy_magnitude = int(np.floor(np.log10(mean_energy)))
        agent_norm_energies = agent_norm_energies * 10**(-mean_energy_magnitude)
            
        mean_energy = np.mean(agent_norm_energies)
        diffs = 0
        for i in range(self.N):
            diffs += (np.linalg.norm(agent_norm_energies[i] - mean_energy)) ** 2
    
        fairness = 1/(self.N) * np.sum(diffs)
        return fairness

    def _fairness_local(self, u):
        control_input_size = self.control_input_size

        u_reshape = u.reshape((self.N, self.H, control_input_size))
        # agent_norm_energies = np.sum(np.linalg.norm(u_reshape, axis=2)**2, axis=1) / (np.array(self.solo_energies) + EPS)
        agent_norm_energies = (np.linalg.norm(u_reshape, axis=(1,2))**2) / (np.array(self.solo_energies) + EPS)
        mean_energy = np.mean(agent_norm_energies)
        mean_energy_magnitude = int(np.floor(np.log10(mean_energy)))
        # scale down magnitude
        agent_norm_energies = agent_norm_energies * 10**(-mean_energy_magnitude)
        mean_energy = np.mean(agent_norm_energies)
        partials = []
        for i in range(self.N):
            grad = 2 * (1/self.N) * (np.linalg.norm(agent_norm_energies[i] - mean_energy))
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
        
        # scale down magnitude
        agent_mean_energies = np.mean(energies)
        mean_energy_magnitude = int(np.floor(np.log10(agent_mean_energies)))
        energies = energies * 10**(-mean_energy_magnitude)
        
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
        
        # scale down magnitude
        agent_mean_energies = np.mean(energies)
        mean_energy_magnitude = int(np.floor(np.log10(agent_mean_energies)))
        energies = energies * 10**(-mean_energy_magnitude)
        
        surges = np.diff(energies)
        surges = surges - np.min(surges) / (np.max(surges) - np.min(surges))
        surge_thresh = np.mean(surges) + np.std(surges)

        agent_total_over_surge = []
        for i in range(self.N):
            agent_total_over_surge.append(np.sum(surges[i] - surge_thresh))
        
        mean_agent_surge = np.mean(agent_total_over_surge)
        partials = []
        for i in range(self.N):
            grad = 2 * (1/self.N) * (agent_total_over_surge[i] - mean_agent_surge)
            partials.append(grad)   

        return partials
    

    ##########################################################
    # Check that Mission Successful after Solving
    ###########################################################

    def check_avoid_constraints(self, u, avoid_only=False):
        drone_results = []
        for i in range(self.N):
            drone_res = self.check_avoid_constraints_traj(u, i, avoid_only=avoid_only)
            drone_results.append(drone_res)
        return drone_results
        
    
    def check_avoid_constraints_traj(self, u, drone_id, avoid_only=False):
        # return for a single drone
        control_input_size = self.control_input_size
        u_reshape = u.reshape((self.N, self.H, control_input_size))
        
        drone_hit = 0
        _, positions = generate_agent_states(u_reshape[drone_id], self.init_states[drone_id], self.init_pos[drone_id], model=self.system_model, dt=self.dt)
        final_p = positions[self.H]
        positions = positions[1:]
        for obsId, obs in self.obstacles.items():
            c = obs['center']
            r = obs['radius']
            distances_to_obstacle = np.linalg.norm(positions - c, axis=1)
            if any(distances_to_obstacle < r):
                drone_hit = 2
                break
        drone_reach = 0
        if not avoid_only:
            cg = self.targets[drone_id]['center']
            rg = self.targets[drone_id]['radius'] + 0.4
            distance_to_target = np.linalg.norm(final_p - cg)
            if distance_to_target > rg:
                drone_reach = 1
                self.dist_to_goal.append(distance_to_target)
            else:
                self.goals_made += 1

        drone_hit_reach = max(drone_hit, drone_reach)

        positions_i = positions
        drone_collide = 0
        for j in range(drone_id+1, self.N):
            _, positions_j = generate_agent_states(u_reshape[j], self.init_states[j], self.init_pos[j], model=self.system_model, dt=self.dt)
            positions_j = positions_j[1:]
            distances_to_obstacle = np.linalg.norm(positions_i - positions_j, axis=1)
            if any(distances_to_obstacle < self.safe_dist):
                drone_collide = 3
                break
        return max(drone_hit_reach, drone_collide)
                       
    

    ##########################################################
    # Online Solution (Obstacle Avoidance and Mutual Separation Only)
    ###########################################################

    def solve_nbf(self, last_delta=None, seed_u=None, mpc=False, h_gamma=1, V_alpha=1):
        # Centrally Solve for Obstacle Avoidance
        f = self.f
        g = self.g
        f_diag = self.f_diag
        g_diag = self.g_diag
    
        u_t = cp.Variable(self.N*self.control_input_size)
        delta = cp.Variable(1)
        # w = cp.Variable(1)
        
        # Create Quad systems for each 
        robots = []
        for r in range(self.N): 
            rob = self.system_model(self.init_states[r], dt=self.dt)
            robots.append(rob)

        if seed_u is not None:
            u_fair_t = cp.Parameter((self.N*self.control_input_size), 
                                    value=np.zeros(self.N*self.control_input_size))
            # objective = cp.Minimize(cp.sum_squares(u_t - u_fair_t) + delta**2 + w)
            objective = cp.Minimize(cp.sum_squares(u_t - u_fair_t) + delta**2)
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
                # rn = self.rn[r]
                rn = 0
                leftright = 1 if r % 2 == 0 else -1
                x_adj = 1 if r % 2 == 0 else 0
                z_adj = 1 if (r % 3 == 0) and self.N >= 10 else 0
                pos_adj = np.array([x_adj, 1, z_adj]) 
                velocity_desired = - kx * ( robots[r].state[0:3] - (self.targets['center'] + leftright*r*self.safe_dist*pos_adj) )
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
                for obsId, obs in self.obstacles.items():
                    c = obs['center']
                    rad = obs['radius'] + self.safe_dist  # + 1
                    h_o = (pos[0] - c[0])**2 + (pos[1] - c[1])**2 + (pos[2] - c[2])**2 - rad**2
                    h_os.append(h_o)
                    h_o_min = np.min([h_o_min, h_o])
                    Brow_start = [0 for i in range(r*6)]
                    Brow_end = [0 for i in range((r+1)*6, self.N*6)]
                    Brow = Brow_start + [2*(pos[0] - c[0]), 2*(pos[1] - c[1]), 2*(pos[2] - c[2]), 0, 0, 0] + Brow_end
                    B.append(Brow)
                # ground avoidance
                # h_g = pos[2]**2
                # h_os.append(h_g)
                # h_o_min = np.min([h_o_min, h_g])
                # Brow_start = [0 for i in range(r*6)]
                # Brow_end = [0 for i in range((r+1)*6, self.N*6)]
                # Brow = Brow_start + [0, 0, 2*pos[2], 0, 0, 0] + Brow_end
                # B.append(Brow)

                # reach goal
                cg = self.targets[r]['center']
                rg = self.targets[r]['radius']
                V = (pos[0] - cg[0])**2 + (pos[1] - cg[1])**2 + (pos[2] - cg[2])**2 - (rg/2)**2   # TODO: include target velocities?
                Vs.append(V)
                V_max = np.max([V_max, V])
                Crow_start = [0 for i in range(r*6)]
                Crow_end = [0 for i in range((r+1)*6, self.N*6)]
                Crow = Crow_start + [2*(pos[0] - cg[0]), 2*(pos[1] - cg[1]), 2*(pos[2] - cg[2]), 0, 0, 0] + Crow_end
                C.append(Crow)
                
                # mutual separation
                drone_start = self.starts[r]
                dsc = drone_start['center']
                dsr = drone_start['radius']
                dist_from_start = (pos[0] - dsc[0])**2 + (pos[1] - dsc[1])**2 + (pos[2] - dsc[2])**2
                if (dist_from_start <= dsr**2 or V <= 0):
                    # print('still in start, continuing')
                    continue
                for s in range(r+1, self.N):
                    pos1 = robots[s].state[0:3]
                    h_c = (pos[0] - pos1[0])**2 + (pos[1] - pos1[1])**2 + (pos[2] - pos1[2])**2 - self.safe_dist**2
                    h_cs.append(h_c)
                    h_c_min = np.min([h_c_min, h_c])
                    deriv = [2*(pos[0] - pos1[0]), 2*(pos[1] - pos1[1]), 2*(pos[2] - pos1[2]), 0, 0, 0]
                    nderiv = [-2*(pos[0] - pos1[0]), -2*(pos[1] - pos1[1]), -2*(pos[2] - pos1[2]), 0, 0, 0]
                    Arow_start = [0 for i in range(r*6)]
                    Arow_mid = [0 for i in range((r+1)*6, s*6)]
                    Arow_end = [0 for i in range((s+1)*6, self.N*6)]
                    Arow = Arow_start + deriv + Arow_mid + nderiv + Arow_end
                    # A.append(Arow)
                    B.append(Arow)

            h_min = np.min([h_c_min, h_o_min])
            B = np.array(B)
            Lfh1 = np.dot(np.dot(f_diag, np.array(state_collect).flatten()), B.T) 
            Lgh1_u = np.dot(B, g_diag) @ u_t
            # constraints.append(Lfh1 + Lgh1_u + h_gamma*h_min - w >= 0) 
            constraints.append(Lfh1 + Lgh1_u + h_gamma*h_min >= 0) 

            C = np.array(C)
            LfV = np.dot(np.dot(f_diag, np.array(state_collect).flatten()), C.T) 
            LgV_u = np.dot(C, g_diag) @ u_t
            constraints.append(LfV + LgV_u + V_alpha*V_max <= delta) 

            # if self.N > 1:
            #     A = np.array(A)
            #     Lfh2 = np.dot(np.dot(f_diag, np.array(state_collect).flatten()), A.T) 
            #     Lgh2_u = np.dot(A, g_diag) @ u_t
            #     constraints.append(Lfh2 + Lgh2_u + h_gamma*h_min >= 0) 

            # constraints.append(w >= 0)
            # Ubox constraint
            constraints.append(-1 * self.Ubox <= u_t)
            constraints.append(u_t <= self.Ubox)

            if last_delta is not None:
                last_delta = max(0, last_delta)  # don't let last alpha be negative ?
                # last_delta = np.maximum(np.zeros(self.N), last_delta)
                constraints.append(delta <= last_delta)

            cbf_controller = cp.Problem(objective, constraints)
            cbf_controller.solve(solver=CP_SOLVER)

            relaxed = False
            if cbf_controller.status in ['infeasible', 'infeasible_inaccurate']:
                print(f"QP infeasible")
                if last_delta is not None:
                    print(f"attempting relaxation")
                    constraints.pop() #delta constraint
                    constraints.pop() #ubox constraint
                    constraints.pop() #ubox constraint
                    cbf_controller = cp.Problem(objective, constraints)
                    cbf_controller.solve(solver=CP_SOLVER)
                    relaxed = True
                    if cbf_controller.status in ['infeasible', 'infeasible_inaccurate']:
                        print(f"Relaxed problem also infeasible")
                        raise Exception('Central Safe Problem Infeasible after relaxation')
                else:
                    print(f"attempting relaxation")
                    constraints.pop() #ubox constraint
                    constraints.pop() #ubox constraint
                    cbf_controller = cp.Problem(objective, constraints)
                    cbf_controller.solve(solver=CP_SOLVER)
                    relaxed = True
                    if cbf_controller.status in ['infeasible', 'infeasible_inaccurate']:
                        print(f"Relaxed problem also infeasible")
                        raise Exception('Central Safe Problem Infeasible after relaxation')
                    # raise Exception('Central Safe Problem Infeasible')
            
            for r in range(self.N):
                idx_start = r*self.control_input_size
                idx_end = idx_start + self.control_input_size
                _, _ = robots[r].forward(u_t.value[idx_start:idx_end])
            final_u.append(u_t.value.reshape(self.N, self.control_input_size))
            if mpc:
                break
        return final_u, h_min, V_max, delta.value[0], h_os, h_cs, Vs, relaxed
        

    ##########################################################
    # Distributed Version of Online Solution (MPC only)
    ###########################################################
    def solve_distributed_nbf(self, seed_input, last_deltas, h_i=1, h_o=1, h_v=1, step_size=0.01, trade_param=0.8):

        # Keep only first time step of seed_input
        seed_input = seed_input[:, 0, :]

        # CALCULATE TARGET POSITIONS FOR ALL AGENTS BASED ON SEED INPUT
        agent_target_pos = []
        for i in range(self.N):
            _, target_pos = generate_agent_states(seed_input[i], self.init_states[i], self.init_pos[i], 
                                                  self.system_model, dt=self.dt)
            agent_target_pos.append(target_pos[1])

        # append 0 to end of seed_input for delta var
        seed_input = np.concatenate((seed_input, np.zeros((seed_input.shape[0], 1))), axis=1)
        
        # INTIALIZE LOCAL CONSTRAINTS
        agent_gammas = []
        agent_Ais = []
        agent_bis = []
        uis = []
        cbfs = []
        clfs = []
        for i in range(self.N):
            g, Ai, bi, cbf_val, clf_val = self.init_local_constraints(i, agent_target_pos, h_i=h_i, h_o=h_o, h_v=h_v)
            agent_gammas.append(g)
            agent_Ais.append(Ai)
            agent_bis.append(bi)
            cbfs.append(cbf_val)
            clfs.append(clf_val)
            ui = seed_input[i]
            ui_full = np.hstack([ui, g])
            uis.append(ui_full)
        
        m = Ai.shape[0]
        lambdas = [np.ones(m) for i in range(self.N)]
        zis = [0 for i in range(self.N)]
        yis = [0 for i in range(self.N)]

        # Repeat below until convergence, auxiliary variables in ui will converge to match the neighbors control inputs in all ujs
        last_deltas = [9999 for i in range(self.N)]
        agent_alg1_running = [True for i in range(self.N)]
        all_Js = []
        for p in range(100):
            if not any(agent_alg1_running):
                break
            # use to get all current phi_i for this iteration
            _, agent_phi_i = self.trades_sigma(uis, seed_input, grad=False, return_all=True)  
            new_uis = []
            new_lambdas = []
            new_zis = []
            new_yis = []
            for i in range(self.N):
                if agent_alg1_running[i]:
                    # Compute lines 3-6 of Alg 1
                    ui_p, lambda_p, zi_p, yi_p = self.trades(i, uis, seed_input, last_deltas[i], agent_Ais, agent_bis, agent_phi_i, 
                                                            lambdas, zis, yis, step_size=step_size, trade_param=trade_param)
                    # Check convergence criteria
                    cur_delta = np.maximum(np.linalg.norm(ui_p - uis[i]),
                                           np.linalg.norm(lambda_p - lambdas[i]))
                    last_deltas[i] = cur_delta
                    if cur_delta < 0.1:
                        agent_alg1_running[i] = False
                else:
                    ui_p = uis[i]
                    lambda_p = lambdas[i]
                    zi_p = zis[i]
                    yi_p = yis[i]
                # update uis, lambdas, zis, yis
                new_uis.append(ui_p)
                new_lambdas.append(lambda_p)
                new_zis.append(zi_p)
                new_yis.append(yi_p)
            
            uis = new_uis
            lambdas = new_lambdas
            zis = new_zis
            yis = new_yis
            
            # track and compute J as well
            sigma = self.trades_sigma(uis, seed_input, grad=False, return_all=False)  
            Ji = self.trades_J_i(uis, seed_input, sigma)
            all_Js.append(Ji)
        
        return np.array(uis)[:,0:4], all_Js, cbfs, clfs

    
    def init_local_constraints(self, agent_id, target_poses, h_i=1, h_o=1, h_v=1):
        # Computes Ai, ui, and all auxiliary variables (gamma_i) in compact form (see equation 8 of Margellos paper)

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
        h_ij_vals = []
        h_obj_vals = []
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
            ## ABS BARRIER FUNC
            hij = np.sum(np.abs(agent_error_dyn - neighbor_error_dyn + delta_p) / self.safe_dist - 1)
            tij = 2/(self.dt**2) * h_i * hij + \
                2/self.dt*(1/self.safe_dist*np.sum(np.sign(actual_dist) * delta_v))
            ## NORM BARRIER FUNC
            # hij = np.linalg.norm(agent_error_dyn - neighbor_error_dyn + delta_p)**2 - self.safe_dist**2
            # tij = 2/(self.dt**2) * h_i * hij + \
            #     2/self.dt*(1/self.safe_dist*np.sum(np.sign(actual_dist) * delta_v))
            gammas.append(tij)
            Ai.append(-1 * np.sign(actual_dist))
            h_ij_vals.append(hij)

        # append tij for static obstacles as well (delta_v is 0 and self.safe_dist == obstacle radius)
        for obsId, obs in self.obstacles.items():
            obj_actual_dist = self.init_pos[agent_id] - obs['center']
            delta_obj_p = agent_target_pos - obs['center']
            ## ABS BARRIER FUNC
            hi_obj = np.sum(np.abs(agent_error_dyn + delta_obj_p) / obs['radius'] - 1)
            ti_obj = 2/(self.dt**2) * h_o * hi_obj + \
                2/self.dt*(1/obs['radius']*np.sum(np.sign(obj_actual_dist) * agent_actual_vel))
            ## NORM BARRIER FUNC
            # hi_obj = np.linalg.norm(agent_error_dyn + delta_obj_p)**2 - obs['radius']**2
            # ti_obj = 2/(self.dt**2) * h_o * hi_obj + \
            #     2/self.dt*(1/self.obstacles['radius']*np.sum(np.sign(obj_actual_dist) * agent_actual_vel))
            gammas.append(ti_obj)
            Ai.append(-1 * np.sign(obj_actual_dist))
            h_obj_vals.append(hi_obj)
        
        # append tij for the lyapunov function (similar to static obstacle case but negative because we want to go to target)
        target_actual_dist = self.init_pos[agent_id] - self.targets[agent_id]['center']
        delta_target_p = agent_target_pos - self.targets[agent_id]['center']
        ## ABS BARRIER FUNC
        vi_target = np.sum((np.abs(agent_error_dyn + delta_target_p) / self.targets[agent_id]['radius'] - 1))
        ti_target = 2/(self.dt**2) * h_v * vi_target + \
            2/self.dt*(1/self.targets[agent_id]['radius']*np.sum(np.sign(target_actual_dist) * agent_actual_vel))
        ## NORM BARRIER FUNC
        # vi_target = np.linalg.norm(agent_error_dyn + delta_target_p)**2 - self.targets[agent_id]['radius']**2
        # ti_target = 2/(self.dt**2) * h_v * vi_target + \
        #     2/self.dt*(1/self.targets[agent_id]['radius']*np.sum(np.sign(target_actual_dist) * agent_actual_vel))
        gammas.append(ti_target)
        Ai.append(np.sign(target_actual_dist))

        Ai = np.array(Ai)
        
        n_p = np.array(gammas).size
        
        # for the additional delta decision variable 
        delta_col = np.zeros((Ai.shape[0], 1))
        delta_col[self.N] = -1
        Ai = np.append(Ai, delta_col, axis=1)  
        
        # -1 for each auxiliary variable (except CLF constraint)
        aux_col = -1 * np.ones((Ai.shape[0], 1))
        aux_col[self.N] = 1
        Ai = np.append(Ai, aux_col, axis=1)

        # append all the zeros
        Ai = np.append(Ai, np.zeros((Ai.shape[0], n_p - 1)), axis=1)
        bi = np.zeros((Ai.shape[0], 1))

        cbf_val = min(np.min(h_ij_vals), np.min(h_obj_vals))
        clf_val = vi_target

        return gammas, Ai, bi, cbf_val, clf_val
    
    # def trades_sigma(self, proposed_inputs, target_positions, grad=False, return_all=False):
    def trades_sigma(self, proposed_inputs, fair_inputs, grad=False, return_all=False):
        # the aggregative vector for the cost function J, which is the average distance to fair inputs
        if grad:
            partials = []
            for i in range(self.N):
                partials.append(1/self.N * self.trades_phi_i(proposed_inputs[i], fair_inputs[i], grad=True))
            return partials
        else:
            total_phi_i = 0
            all_phis = []
            for i in range(self.N):
                phi_i = self.trades_phi_i(proposed_inputs[i], fair_inputs[i], grad=False)
                total_phi_i += phi_i
                all_phis.append(phi_i)
            if not return_all:
                return 1/self.N * total_phi_i
            else:
                return 1/self.N * total_phi_i, all_phis
    
    def trades_phi_i(self, agent_proposed_input, agent_fair_input, grad=False):
        # phi_i penalizes the agent's fraction of aggregate distance to fair inputs for this timestep
        if grad:
            return 2 * np.linalg.norm(agent_proposed_input[0:4] - agent_fair_input[0:4])  # 0:4 to include delta var
        else:
            return np.linalg.norm(agent_proposed_input[0:4] - agent_fair_input[0:4])**2
        

    def trades(self, agent_id, current_inputs, seed_input, last_delta, Ais, bis, phi,
               lambdas, zis, yis, step_size=0.01, trade_param=0.8):
        # Implement lines 3-6 of Algorithm 1 from Margellos paper
        # (but for horizon = 1, that is, returned ui for each agent is what they should do for next time step, 
        # no future timesteps computed)

        # let network parameter wij be 1/N-1 (ie one over all neighbors) for all neighbors
        wij = 1 / (self.N - 1)
        rho = 1.5 * step_size * trade_param / wij

        ui_full = current_inputs[agent_id]

        # Compute Pseudo gradiant of Ji
        F_i = self.trades_F_i(ui_full, seed_input[agent_id], phi[agent_id] + zis[agent_id])
        
        # Compute Pseudo gradiant of constraints
        G_ui = self.trades_G_i_primal(ui_full, Ais[agent_id], bis[agent_id] + yis[agent_id], lambdas[agent_id], rho)
        G_lambi = self.trades_G_i_dual(ui_full, Ais[agent_id], bis[agent_id] + yis[agent_id], lambdas[agent_id], rho)
        
        # compute ui_p (line 3)
        ui_p = ui_full + trade_param * (self.projection(ui_full - step_size * F_i - step_size * G_ui, last_delta) - ui_full)
        
        # compute lambda, zi, yi (line 4-6)
        lambda_p = 0
        zi_p = 0
        yi_p = 0
        for i in range(self.N):
            if i == agent_id:
                continue
            uj_full = current_inputs[agent_id]
            lambda_p += wij * lambdas[i] + trade_param * step_size * G_lambi
            zi_p += wij * zis[i] + wij * phi[i]
            yi_p += wij * yis[i] + wij * self.N * (np.dot(Ais[i], uj_full) - bis[i]) 

        zi_p = zi_p - phi[agent_id]
        yi_p = yi_p - self.N * (np.dot(Ais[i], ui_full) - bis[i]) 
        return ui_p, lambda_p, zi_p, yi_p
        

    def trades_J_i(self, agent_proposed_input, agent_seed_input, second_arg, grad=False):
        if grad:
            # THE PSEUDO GRADIENT OF OBJECTIVE FUNCTION
            # first is gradient with respect to first argument, the agent's inputs, which is the derivative of agent's distance to seed inputs
            partial_1 = 2 * np.linalg.norm(agent_proposed_input[0:4] - agent_seed_input[0:4])
            # second is gradient with respect to second argument, the aggregate, which is just 1
            partial_2 = 1
            return partial_1, partial_2
        else:
            # OBJECTIVE FUNCTION, MINIMIZE AGENT'S OWN DISTANCE TO SEED_INPUTS (THE FAIR INPUTS) and THE AVERAGAE OF AGGREGATE DISTANCE SIGMA
            return np.linalg.norm(np.array(agent_proposed_input)[:,0:4] - np.array(agent_seed_input)[:,0:4])**2 + second_arg  # second arg is the aggregate vector sigma representing distance to all fair inputs
        
    def trades_F_i(self, agent_proposed_input, agent_seed_input, second_arg):
        # pseudo gradient of J_i
        partial_1_J_i, partial_2_j_i = self.trades_J_i(agent_proposed_input, agent_seed_input, second_arg, grad=True)  # here phi is partial of phi for agent i
        partial_phi_i = self.trades_phi_i(agent_proposed_input, agent_seed_input, grad=True)
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
        return res

    def projection(self, v, last_delta):
        # Project v into Ubox AND delta_prev constraint
        for i in [0, 1, 2]:
            if v[i] < -1 * self.Ubox:
                v[i] = -1 * self.Ubox
            elif v[i] > self.Ubox:
                v[i] = self.Ubox
            else:
                continue
        last_delta = max(0, last_delta)
        if v[3] > last_delta:
            v[3] = last_delta
        if v[3] < 0:
            v[3] = 0
        return v
