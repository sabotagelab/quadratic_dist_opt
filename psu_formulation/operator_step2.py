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
        # FOR TESTING
        # if t == 5:
        #     GST.add_node(nodeId, name=a, cap=airportCapacities[i], hab=1, ts=t, resource='airport')
        # else:
        #     GST.add_node(nodeId, name=a, cap=airportCapacities[i], hab=airportHabitancies[i], ts=t, resource='airport')
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
print(ResourceDict)

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

# Set Flight
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
flight = f2
flight_min_start_time = flight['departure']['time']
flight_max_end_time = flight['arrival']['time'] + flight['flex']
subset_times = timesteps

departure_idx = ResourceDict[flight['departure']['airport']]
arrival_idx = ResourceDict[flight['arrival']['airport']]

# Flight Choices from PSU
# ['A1', 'A2', 'S1', 'S2', 'S3', 'S4']
# flight 1
# c1 = np.array([[1., 0., 0., 0., 0., 0.],
#                [1., 0., 0., 0., 1., 0.],
#                [0., 0., 1., 0., 1., 1.],
#                [0., 0., 1., 1., 0., 1.],
#                [0., 1., 0., 1., 0., 0.],
#                [0., 1., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0.]])
# flight 2
c1 = np.array([[0., 1., 0., 0., 0., 0.],
               [0., 1., 0., 1., 0., 0.],
               [0., 0., 1., 1., 0., 1.],
               [0., 0., 1., 0., 1., 1.],
               [1., 0., 0., 0., 1., 0.],
               [1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])

# If conflict in sector 1 for entire period
# flight 1
# c1 = np.array([[1., 0., 0., 0., 0., 0.],
#                [1., 0., 0., 0., 1., 0.],
#                [0., 0., 1., 0., 1., 1.],
#                [0., 0., 1., 1., 0., 1.],
#                [0., 1., 0., 1., 0., 0.],
#                [0., 1., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0.],
#                [0., 0., 0., 0., 0., 0.]])
# flight 2
c1 = np.array([[0., 1., 0., 0., 0., 0.],
               [0., 1., 0., 1., 0., 0.],
               [0., 0., 1., 1., 0., 1.],
               [0., 0., 1., 0., 1., 1.],
               [1., 0., 0., 0., 1., 0.],
               [1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]])

# Define Problem
P = pc.Problem()

# Decision Variables
w1 = pc.BinaryVariable("w1", (len(subset_times), len(GS.nodes())))

# Parameters
t1 = np.array([1+i for i in range(len(subset_times))])
t1 = pc.Constant('t1', t1)

# Objective, Minimize energy AND delay (difference from actual arrival time and proposed arrival time)
P.set_objective('min', pc.sum(w1) + (t1.T*w1[:, arrival_idx] - flight['arrival']['time']) + (t1.T*w1[:, departure_idx] - flight['departure']['time']))
# P.set_objective('min', pc.norm(w1) + (t1.T*w1[:, arrival_idx] - flight['arrival']['time']) + (t1.T*w1[:, departure_idx] - flight['departure']['time']))

# Constraints
for i in range(len(subset_times)):
    ## Don't be in more than one cell at a time
    P.add_constraint(pc.sum(w1[i,:]) <= 1)

    for r, id in ResourceDict.items():
        nodeId = '{}-t{}'.format(r, subset_times[i])
        node_data = GST.nodes(data=True)[nodeId]
        adj_cells = list(GS.neighbors(r))
        adj_cells.append(r)
        adj_cells_idx = [ResourceDict[rs] for rs in adj_cells]
        
        ## Don't choose a restricted time slot
        P.add_constraint(w1[i, id] <= c1[i, id])

        ## Volume constraints
        P.add_constraint(w1[i, id] <= node_data['cap'] - node_data['hab'])

        ## TODO: Flight Capability Constraints (weather/altitude/separation constraints)
        ## for now, everything is permissible

        ## Constrain path connectivity/cell adjacency
        # ie if in cell, then next cell in time is itself or an adjacent cell (unless cell is destination cell)
        # w1[i, id] - (w1[i+1, id] + sum(w1[i+1, all adjacent cells])  <= 0
        # if (i < flight_max_end_time - 1) and id != arrival_idx:
        if i > 0 and r != flight['departure']['airport']:
            # P.add_constraint(w1[i, id] - pc.sum([w1[i+1, rs] for rs in adj_cells_idx]) <= 0)
            P.add_constraint(w1[i, id] - pc.sum([w1[i-1, rs] for rs in adj_cells_idx]) <= 0)

        ## Constrain minimum time spent in cell
        ## ie if in cell at time t, than still in cell at time < t+min_cell_time (aka time <= t+min_cell_time-1)
        ## w1[i, id] - wi[i+(min_cell_time-1), id] <= 0, (if min_cell_time = 1 then it's just w1[i, id] - w1[i, id] == 0
        P.add_constraint(w1[i, id] - w1[i+(flight['min_cell_time']-1), id] <= 0)

## Only one departure airport slot chosen
P.add_constraint(pc.sum(w1[:, departure_idx]) == 1)

## Only one arrival airport slot chosen
P.add_constraint(pc.sum(w1[:, arrival_idx]) == 1)

## At least 1 departure sector chosen
departure_sectors = list(GS[flight['departure']['airport']].keys())
departure_sectors_ids = [ResourceDict[ds] for ds in departure_sectors]
P.add_constraint(pc.sum(w1[:, departure_sectors_ids]) >= 1)  # will this work if multiple sectors?

## At least 1 arrival sector chosen
arrival_sectors = list(GS[flight['arrival']['airport']].keys())
arrival_sectors_ids = [ResourceDict[rs] for rs in arrival_sectors]
P.add_constraint(pc.sum(w1[:, arrival_sectors_ids]) >= 1)  # will this work if multiple sectors?

try:
    print(P)
    P.options.solver = "mosek"
    solution = P.solve()
    print(solution)
    print(ResourceDict)
    print(np.array(w1.value))

except pc.SolutionFailure as e:
    print('Infeasible')

    # REASONS FOR INFEASIBILITY AT THIS STEP
    # - GRID STATE CHANGED BETWEEN STEP 1 AND 2 SUCH ENOUGH INTERMEDIATE SECTORS ARE BLOCKED THAT NO PATH CAN BE PLANNED, 
    #   - SOLUTION 1 -> TOO BAD, SUBMIT REQUEST AT ANOTHER TIME (OR AUTO-SUBMIT TO STEP 1 WITH SHIFTED TIME))
    #   - SOLUTION 2 -> TRY TO SOLVE PROBLEM BY SHIFTING TIME CONSTRAINTS ON START AND END RESOURCES, 
    #       BUT THESE CAN RENDER PROBLEM INFEASIBLE AT NEXT STEP, BETTER TO RESUBMIT REQUEST
    
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

f1_res = np.array(w1.value)

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

    for j in range(f1_res.shape[1]):
        if f1_res[t, j] == 1:
            p = get_grid_point(j)
            f1_points_xs.append(p[0])
            f1_points_ys.append(p[1])
    
    # axes[t].scatter(f1_points_xs, f1_points_ys, color='blue')
    axes[t].scatter(f1_points_xs, f1_points_ys, color='red')
    axes[t].set_title('t={}'.format(t))

fig.suptitle('Operator Step 2, Flight 2, S1 Blocked')
plt.subplot_tool()
plt.show()