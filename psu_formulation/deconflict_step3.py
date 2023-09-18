import networkx as nx
import numpy as np
import picos as pc

# Create Spatio Temporal Grid
# each node is either a airport or a sector/cell at a time step
# each node has Capacity and Habitancy attributes

timesteps = [i for i in range(1, 11)]  # 10 timesteps [1, ... 10]
airportIds = ['A{}'.format(i) for i in [1, 2]]  # 2 airports
airportCapacities = [1 for i in [1, 2]]
airportHabitancies = [0 for i in [1, 2]]
sectorIds = ['S{}'.format(i) for i in [1, 2, 3, 4]]  # 4 airports
sectorCapacities = [1 for i in [1, 2, 3, 4]]
sectorHabitancies = [0 for i in [1, 2, 3, 4]]

GST = nx.DiGraph()  # spatio-temporal grid
GS = nx.Graph()  # spatial grid only
# Add Nodes
for t in timesteps:
    for i, a in enumerate(airportIds):
        nodeId = '{}-t{}'.format(a, t)
        GST.add_node(nodeId, name=a, cap=airportCapacities[i], hab=airportHabitancies[i], ts=t, resource='airport')
        # Add Edge pointing back to previous timestep
        if t > min(timesteps):
            prevNodeId = '{}-t{}'.format(a, t-1)
            GST.add_edge(prevNodeId, nodeId)
        if t == min(timesteps):
            GS.add_node(a, name=a, resource='airport')
    for i, s in enumerate(sectorIds):
        nodeId = '{}-t{}'.format(s, t)
        # FOR TESTING 
        # if t == 3 and s in ['S1']: 
        # if t in [3, 4, 5] and s in ['S1']: 
        if s in ['S1']: 
            GST.add_node(nodeId, name=s, cap=sectorCapacities[i], hab=1, ts=t, resource='sector')
        else:
            GST.add_node(nodeId, name=s, cap=sectorCapacities[i], hab=sectorHabitancies[i], ts=t, resource='sector')
        # GST.add_node(nodeId, name=s, cap=sectorCapacities[i], hab=sectorHabitancies[i], ts=t, resource='sector')
        # Add Edge pointing back to previous timestep
        if t > min(timesteps):
            prevNodeId = '{}-t{}'.format(s, t-1)
            GST.add_edge(prevNodeId, nodeId)
        if t == min(timesteps):
            GS.add_node(s, name=s, resource='sector')
ResourceDict = {n: i for i, n in enumerate(list(GS.nodes()))}

# Add Edges Between Sectors
for t in timesteps:
    if t == 10:
        break
    # A1 Connects to S3
    GST.add_edge('A1-t{}'.format(t), 'S3-t{}'.format(t+1))
    GST.add_edge('S3-t{}'.format(t), 'A1-t{}'.format(t+1))
    # S3 Connects to S1 and S4
    GST.add_edge('S3-t{}'.format(t), 'S1-t{}'.format(t+1))
    GST.add_edge('S1-t{}'.format(t), 'S3-t{}'.format(t+1))
    GST.add_edge('S3-t{}'.format(t), 'S4-t{}'.format(t+1))
    GST.add_edge('S4-t{}'.format(t), 'S3-t{}'.format(t+1))
    # S2 Connects to S1 and S4
    GST.add_edge('S1-t{}'.format(t), 'S2-t{}'.format(t+1))
    GST.add_edge('S2-t{}'.format(t), 'S1-t{}'.format(t+1))
    GST.add_edge('S4-t{}'.format(t), 'S2-t{}'.format(t+1))
    GST.add_edge('S2-t{}'.format(t), 'S4-t{}'.format(t+1))
    # A2 Connects to S2
    GST.add_edge('A2-t{}'.format(t), 'S2-t{}'.format(t+1))
    GST.add_edge('S2-t{}'.format(t), 'A2-t{}'.format(t+1))
    if t == 1:
        # A1 Connects to S3
        GS.add_edge('A1', 'S3')
        # S3 Connects to S1 and S4
        GS.add_edge('S3', 'S1')
        GS.add_edge('S3', 'S4')
        # S2 Connects to S1 and S4
        GS.add_edge('S2', 'S1')
        GS.add_edge('S2', 'S4')
        # A2 Connects to S2
        GS.add_edge('A2', 'S2')

# Create Flights
# f1 = {'departure': {'airport': 'A1', 'time': 1}, 'arrival': {'airport': 'A2', 'time': 5}, 'flex': 1, 'min_cell_time': 1, 
#       'O_bar': ['S3', 'S2', 'A1', 'A2'], 
#       'graph_constraint_set': ['A1-t1', 'A1-t2', 'A2-t5', 'A2-t6', 'S3-t2', 'S3-t3', 'S1-t3', 'S4-t3', 'S3-t4', 'S1-t4', 'S4-t4', 'S2-t4', 'S2-t5', 'S2-t3']}
# f2 = {'departure': {'airport': 'A2', 'time': 1}, 'arrival': {'airport': 'A1', 'time': 5}, 'flex': 1, 'min_cell_time': 1, 
#       'O_bar': ['S2', 'S3', 'A2', 'A1'], 
#       'graph_constraint_set': ['A2-t1', 'A2-t2', 'A1-t5', 'A1-t6', 'S2-t2', 'S2-t3', 'S1-t3', 'S4-t3', 'S2-t4', 'S1-t4', 'S4-t4', 'S3-t4', 'S3-t5', 'S3-t3']}
# IF SECTOR 1 BLOCKED
f1 = {'departure': {'airport': 'A1', 'time': 1}, 'arrival': {'airport': 'A2', 'time': 5}, 'flex': 1, 'min_cell_time': 1, 
      'O_bar': ['S3', 'S2', 'A1', 'A2'], 
      'graph_constraint_set': ['A1-t1', 'A1-t2', 'A2-t5', 'A2-t6', 'S3-t2', 'S3-t3', 'S1-t3', 'S4-t3', 'S3-t4', 'S1-t4', 'S4-t4', 'S2-t4', 'S2-t5', 'S2-t3']}
f2 = {'departure': {'airport': 'A2', 'time': 1}, 'arrival': {'airport': 'A1', 'time': 5}, 'flex': 1, 'min_cell_time': 1, 
      'O_bar': ['S2', 'S3', 'A2', 'A1'], 
      'graph_constraint_set': ['A2-t1', 'A2-t2', 'A1-t5', 'A1-t6', 'S2-t2', 'S2-t3', 'S1-t3', 'S4-t3', 'S2-t4', 'S1-t4', 'S4-t4', 'S3-t4', 'S3-t5', 'S3-t3']}
F = [f1, f2]


# Intermediate Sectors 
IntermediateSectors = {}
for r, i in ResourceDict.items():
    if r not in f1['O_bar'] or r not in f2['O_bar']:
        IntermediateSectors[r] = i
print(IntermediateSectors)

# PSU set constraints
c1 = np.array([[1., 0., 0., 0., 0., 0.],
               [1., 0., 0., 0., 1., 0.],
               [0., 0., 1., 0., 1., 1.],
               [0., 0., 1., 1., 0., 1.],
               [0., 1., 0., 1., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
c2 = np.array([[0., 1., 0., 0., 0., 0.],
               [0., 1., 0., 1., 0., 0.],
               [0., 0., 1., 1., 0., 1.],
               [0., 0., 1., 0., 1., 1.],
               [1., 0., 0., 0., 1., 0.],
               [1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])

# Operator Submitted Flight Plans
w1 = np.array([[1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0.],
               [0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
w2 = np.array([[0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 0., 1.],
               [0., 0., 0., 0., 1., 0.],
               [1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])

# IF SECTOR 1 BLOCKED ENTIRE TIME
# flight 1
c1 = np.array([[1., 0., 0., 0., 0., 0.],
               [1., 0., 0., 0., 1., 0.],
               [0., 0., 1., 0., 1., 1.],
               [0., 0., 1., 1., 0., 1.],
               [0., 1., 0., 1., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
# flight 2
c2 = np.array([[0., 1., 0., 0., 0., 0.],
               [0., 1., 0., 1., 0., 0.],
               [0., 0., 1., 1., 0., 1.],
               [0., 0., 1., 0., 1., 1.],
               [1., 0., 0., 0., 1., 0.],
               [1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
w1 = np.array([[1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0., 1.],
               [0., 0., 0., 1., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])
w2 = np.array([[0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 0., 1.],
               [0., 0., 0., 0., 1., 0.],
               [1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])


# Define Problem
P = pc.Problem()

# Parameters
w1 = pc.Constant('w1', w1)
w2 = pc.Constant('w2', w2)

t1 = pc.Constant('t1', timesteps)

# Decision Variables
v1 = pc.BinaryVariable("v1", (len(timesteps), len(GS.nodes())))
v2 = pc.BinaryVariable("v2", (len(timesteps), len(GS.nodes())))

# Objective, minimize energy variance
# P.set_objective('min', 
#                 0.5*((1-pc.sum(v1)/pc.sum(w1)) - 0.5*((1-pc.sum(v1)/pc.sum(w1)) + (1-pc.sum(v2)/pc.sum(w2))) + \
#                 (1-pc.sum(v2)/pc.sum(w2)) - 0.5*((1-pc.sum(v1)/pc.sum(w1)) + (1-pc.sum(v2)/pc.sum(w2))) ))

# Objective, minimize energy variance AND total delay time
f1_departure_idx = ResourceDict[f1['departure']['airport']]
f2_departure_idx = ResourceDict[f2['departure']['airport']]
f1_arrival_idx = ResourceDict[f1['arrival']['airport']]
f2_arrival_idx = ResourceDict[f2['arrival']['airport']]
P.set_objective('min', 
                # 0.5*( ((1-pc.sum(v1)/pc.sum(w1)) - 0.5*((1-pc.sum(v1)/pc.sum(w1)) + (1-pc.sum(v2)/pc.sum(w2))))**2 + \
                # ((1-pc.sum(v2)/pc.sum(w2)) - 0.5*((1-pc.sum(v1)/pc.sum(w1)) + (1-pc.sum(v2)/pc.sum(w2))))**2 ) + \
                (t1.T*v1[:, f1_arrival_idx] - f1['arrival']['time']) + (t1.T*v1[:, f1_departure_idx] - f1['departure']['time']) + \
                (t1.T*v2[:, f2_arrival_idx] - f2['arrival']['time']) + (t1.T*v2[:, f2_departure_idx] - f2['departure']['time'])
                )

# Constraints
for i in range(len(timesteps)):
    for f_num, flight in enumerate(F):
        if f_num == 0:
            flight_var = v1
            flight_con = w1
        else:
            flight_var = v2
            flight_con = w2
        
        ## Don't be in more than one cell at a time
        P.add_constraint(pc.sum(flight_var[i,:]) <= 1)

    for r, id in ResourceDict.items():
        adj_cells = list(GS.neighbors(r))
        adj_cells.append(r)
        adj_cells_idx = [ResourceDict[rs] for rs in adj_cells]

        if r in IntermediateSectors.keys():
            ## Capacity Constraints for intermediate sectors
            nodeId = '{}-t{}'.format(r, i+1)
            node_data = GST.nodes(data=True)[nodeId]
            P.add_constraint(v1[i, id] + v2[i, id] <= node_data['cap'] - node_data['hab'])

        for f_num, flight in enumerate(F):            
            if f_num == 0:
                flight_var = v1
                flight_con = w1
                psu_con = c1
            else:
                flight_var = v2
                flight_con = w2
                psu_con = c2
            arrival_idx = ResourceDict[flight['arrival']['airport']]
            flight_min_cell_time = flight['min_cell_time']

            # TODO: this constraint can lead to infeasibility if conflicts exist. need to relax this constraint.
            # TODO: instead departure/arrival sectors can change but remain within the PSU's previously set bounds 
            ## Chosen Departure/Arrival Sectors Don't Change, as well as timesteps outside of flight time
            # flight_sched_end_time = 5  # TODO: needs to be extracted from submitted flight plans
            # if r in flight['O_bar'] or i >= flight_sched_end_time:
            #     P.add_constraint(flight_var[i, id] == flight_con[i, id])
            
            ## Don't choose a restricted time slot
            P.add_constraint(flight_var[i, id] <= psu_con[i, id])

            ## Constrain path connectivity/cell adjacency
            # ie if in cell, then next cell in time is itself or an adjacent cell (unless cell is destination cell)
            # w1[i, id] - (w1[i+1, id] + sum(w1[i+1, all adjacent cells])  <= 0
            if i > 0 and r != flight['departure']['airport']:
                P.add_constraint(flight_var[i, id] - pc.sum([flight_var[i-1, rs] for rs in adj_cells_idx]) <= 0)
            
            ## Constrain minimum time spent in cell
            ## ie if in cell at time t, than still in cell at time < t+min_cell_time (aka time <= t+min_cell_time-1)
            ## w1[i, id] - wi[i+(min_cell_time-1), id] <= 0, (if min_cell_time = 1 then it's just w1[i, id] - w1[i, id] == 0
            P.add_constraint(flight_var[i, id] - flight_var[i+(flight_min_cell_time-1), id] <= 0)


for f_num, flight in enumerate(F):
    if f_num == 0:
        flight_var = v1
        flight_con = w1
        psu_con = c1
    else:
        flight_var = v2
        flight_con = w2
        psu_con = c2

    departure_idx = ResourceDict[flight['departure']['airport']]
    arrival_idx = ResourceDict[flight['arrival']['airport']]

    ## Only one departure airport slot chosen
    P.add_constraint(pc.sum(flight_var[:, departure_idx]) == 1)

    ## Only one arrival airport slot chosen
    P.add_constraint(pc.sum(flight_var[:, arrival_idx]) == 1)

    ## At least 1 departure sector chosen
    departure_sectors = list(GS[flight['departure']['airport']].keys())
    departure_sectors_ids = [ResourceDict[ds] for ds in departure_sectors]
    P.add_constraint(pc.sum(flight_var[:, departure_sectors_ids]) >= 1)  # will this work if multiple sectors?

    ## At least 1 arrival sector chosen
    arrival_sectors = list(GS[flight['arrival']['airport']].keys())
    arrival_sectors_ids = [ResourceDict[rs] for rs in arrival_sectors]
    P.add_constraint(pc.sum(flight_var[:, arrival_sectors_ids]) >= 1)  # will this work if multiple sectors?


# print(P)
try:
    P.options.solver = "mosek"
    solution = P.solve()
    print(P)
    print(solution)
    print(list(GS.nodes()))
    print(np.round(np.array(v1.value), 2))
    print(np.round(np.array(v2.value), 2))
    print(0.5*( ((1-pc.sum(v1)/pc.sum(w1)) - 0.5*((1-pc.sum(v1)/pc.sum(w1)) + (1-pc.sum(v2)/pc.sum(w2))))**2 + \
                ((1-pc.sum(v2)/pc.sum(w2)) - 0.5*((1-pc.sum(v1)/pc.sum(w1)) + (1-pc.sum(v2)/pc.sum(w2))))**2 ))
except pc.SolutionFailure as e:
    print('Infeasible')

    # TODO: REASONS FOR INFEASIBILITY
    # - BECAUSE GRID STATE CHANGED IN BETWEEN STEPS 2 AND 3 SUCH THAT STATE IS NO LONGER FEASIBLE, 
    #   SOLUTION 1 -> PSU NEEDS TO SOLVE ENTIRE PROBLEM WITH NEW STATE AND RELAXING CONSTRAINTS ON START AND END RESOURCES ? HOW TO FORMULATE?
    #   SOLUTION 2 -> APPROVE A SUBSET AND KICK BACK TO OTHERS TO STEP 1 WITH SHIFTED TIME)? HOW TO FIND WHO CAN BE APPROVED? HOW TO CHOOSE WHO TO APPROVE FAIRLY IF CONFLICTS?


import matplotlib.pyplot as plt

grid_lines = [[(1, 1), (1, 2)], 
              [(2, 1), (2, 2)], [(2, 2), (2, 3)],
              [(3, 1), (3, 2)], [(3, 2), (3, 3)],
              [(4, 1), (4, 2)], [(4, 2), (4, 3)],
              [(5, 2), (5, 3)],
              [(1, 1), (2, 1)], [(2, 1), (3, 1)], [(3, 1), (4, 1)],
              [(1, 2), (2, 2)], [(2, 2), (3, 2)], [(3, 2), (4, 2)], [(4, 2), (5, 2)],
              [(2, 3), (3, 3)], [(3, 3), (4, 3)], [(4, 3), (5, 3)],
              ]
def get_grid_point(p):
    if p == 0: 
        return (1.5, 1.5)
    if p == 1:
        return (4.5, 2.5)
    if p == 2:  ## 1
        return (2.5, 2.5)
    if p == 3:  ## 2
        return (3.5, 2.5)
    if p == 4: ## 3
        return (2.5, 1.5)
    else:  ## 5
        return (3.5, 1.5)

f1_res = np.array(v1.value)
f2_res = np.array(v2.value)

fig, axes = plt.subplots(6)
for t in range(6):
    for l in grid_lines:
        p1 = l[0]
        p2 = l[1]
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        axes[t].plot(xs, ys, color='black')
    axes[t].set_xlim(0, 6)
    axes[t].set_ylim(0, 4)

    f1_points_xs = []
    f1_points_ys = []

    f2_points_xs = []
    f2_points_ys = []
    
    for j in range(f1_res.shape[1]):
        if f1_res[t, j] >= 1:
            p = get_grid_point(j)
            f1_points_xs.append(p[0])
            f1_points_ys.append(p[1])
    
    for j in range(f2_res.shape[1]):
        if f2_res[t, j] >= 1:
            p = get_grid_point(j)
            f2_points_xs.append(p[0])
            f2_points_ys.append(p[1])
    
    axes[t].scatter(f1_points_xs, f1_points_ys, color='blue')
    axes[t].scatter(f2_points_xs, f2_points_ys, color='red')
    axes[t].set_title('t={}'.format(t))

fig.suptitle('PSU Step 3 Deconfliction, S1 Blocked, no fairness term')
plt.subplot_tool()
plt.show()