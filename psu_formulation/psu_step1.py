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
        # if s in ['S1']: 
        #     GST.add_node(nodeId, name=s, cap=sectorCapacities[i], hab=1, ts=t, resource='sector')
        # else:
        #     GST.add_node(nodeId, name=s, cap=sectorCapacities[i], hab=sectorHabitancies[i], ts=t, resource='sector')
        GST.add_node(nodeId, name=s, cap=sectorCapacities[i], hab=sectorHabitancies[i], ts=t, resource='sector')
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

# Create Flights
f1 = {'departure': {'airport': 'A1', 'time': 1}, 'arrival': {'airport': 'A2', 'time': 5}, 'flex': 1, 'min_cell_time': 1}
# f1 = {'departure': {'airport': 'A1', 'time': 1}, 'arrival': {'airport': 'A1', 'time': 5}, 'flex': 1, 'min_cell_time': 1}
f2 = {'departure': {'airport': 'A2', 'time': 1}, 'arrival': {'airport': 'A1', 'time': 5}, 'flex': 1, 'min_cell_time': 1}
F = [f1, f2]
# F = [f1]

# Create O_bar set, aka the set of resources to be chosen for each flight, aka departure and arrival airports and their adjacent sectors
# TODO: i actually don't need O_bar_all
O_bar_all = []
for flight in F:
    O_bar = list(GS[flight['departure']['airport']].keys()) + list(GS[flight['arrival']['airport']].keys())  # gets the adjacent sectors of departure/arrival airports
    O_bar.append(flight['departure']['airport'])  # add departure airport
    O_bar.append(flight['arrival']['airport'])  # add arrival airport
    flight['O_bar'] = O_bar
    for r in O_bar:
        O_bar_all.append(r)

O_bar_all = list(set(O_bar_all))  # combine sets of all flights


# Create the 'graph_constraint_set'
# for each flight
# add airport departure nodes inside of flex time
# add each child edge of the selected airport departure nodes
# add airport arrival nodes inside of flex time
# add each parent edge of the selected airport departure nodes
# for each added child, add their children up to arrival flex time 
# for each added parent, add their parents up to departure time (unless node is already in set)
O_bar_set_all = []
for flight in F:
    flight['graph_constraint_set'] = []
    queue_children = []

    O_bar_set = []
    flight_min_start_time = flight['departure']['time'] #- flight['flex']  # assume flights can't start earlier? and so can't arrive earlier?
    flight_max_start_time = flight['departure']['time'] + flight['flex']
    departure_nodes = []
    for node, data in GST.nodes(data=True):
        if data['name'] == flight['departure']['airport']:
            if flight_min_start_time <= data['ts'] <= flight_max_start_time:
                flight['graph_constraint_set'].append(node)
                queue_children.append(node)
    queue_parents = []

    flight_min_end_time = flight['arrival']['time'] #- flight['flex']  # assume flights can't start earlier? and so can't arrive earlier?
    flight_max_end_time = flight['arrival']['time'] + flight['flex']
    arrival_nodes = []
    for node, data in GST.nodes(data=True):
        if data['name'] == flight['arrival']['airport']:
            if flight_min_end_time <= data['ts'] <= flight_max_end_time:
                if node not in flight['graph_constraint_set']:
                    flight['graph_constraint_set'].append(node)
                queue_parents.append(node)

    while len(queue_children) > 0:
        node = queue_children.pop(0)
        if node not in flight['graph_constraint_set']:
            flight['graph_constraint_set'].append(node)
        for child in GST.successors(node):
            if (GST.nodes(data=True)[child]['resource'] == 'sector') and \
                (child not in queue_children) and \
                (flight_max_start_time <= GST.nodes(data=True)[child]['ts'] < flight_min_end_time):
                    queue_children.append(child)

    while len(queue_parents) > 0:
        node = queue_parents.pop(0)
        if node not in flight['graph_constraint_set']:
            flight['graph_constraint_set'].append(node)
        for parent in GST.predecessors(node):
            if (GST.nodes(data=True)[parent]['resource'] == 'sector') and \
                (parent not in queue_parents) and \
                (flight_max_start_time < GST.nodes(data=True)[parent]['ts'] <= flight_min_end_time):
                queue_parents.append(parent)

O_bar_set_all = list(set(O_bar_set_all))

print(f1)
print(f2)

# Define Problem
P = pc.Problem()

# Decision Variables
c1 = pc.BinaryVariable("c1", (len(timesteps), len(GS.nodes())))
c2 = pc.BinaryVariable("c2", (len(timesteps), len(GS.nodes())))

# Objective, maximize average choices provided for a flight
P.set_objective('max', 0.5 * (pc.sum(c1) + pc.sum(c2)))
# P.set_objective('max', (pc.sum(c1)))

# Constraints

## Capacity Constraints for decision set 
for i in range(len(timesteps)):
    for r, id in ResourceDict.items():
        nodeId = '{}-t{}'.format(r, i+1)
        # if r in O_bar_all:  # if resource is in decision set, flight can be in resource subject to capacity - expected occupancy
        node_data = GST.nodes(data=True)[nodeId]
        P.add_constraint(c1[i, id] + c2[i, id] <= node_data['cap'] - node_data['hab'])
        # P.add_constraint(c1[i, id] <= node_data['cap'] - node_data['hab'])

        for f_num, flight in enumerate(F):
            if nodeId not in flight['graph_constraint_set']:
                if f_num == 0:
                    P.add_constraint(c1[i, id] <= 0)
                else:
                    P.add_constraint(c2[i, id] <= 0)
            
            ## Constrain path connectivity/cell adjacency
            # ie if in cell, then next cell in time is itself or an adjacent cell (unless cell is destination cell)
            # w1[i, id] - (w1[i+1, id] + sum(w1[i+1, all adjacent cells])  <= 0
            adj_cells = list(GS.neighbors(r))
            adj_cells.append(r)
            adj_cells_idx = [ResourceDict[rs] for rs in adj_cells]
            if i > 0 and r != flight['departure']['airport']:
                if f_num == 0:
                    P.add_constraint(c1[i, id] - pc.sum([c1[i-1, rs] for rs in adj_cells_idx]) <= 0)
                else:
                    P.add_constraint(c2[i, id] - pc.sum([c2[i-1, rs] for rs in adj_cells_idx]) <= 0)

## At Least 1 Slot Chosen for Departure Airport, Arrival Airport, Departure Sectors, Arrival Sectors
for f_num, flight in enumerate(F):
    departure_airport = flight['departure']['airport']
    departure_idx = ResourceDict[departure_airport]
    arrival_airport = flight['arrival']['airport']
    arrival_idx = ResourceDict[arrival_airport]
    
    departure_sectors = list(GS[flight['departure']['airport']].keys())
    departure_sectors_ids = [ResourceDict[ds] for ds in departure_sectors]
    
    arrival_sectors = list(GS[flight['arrival']['airport']].keys())
    arrival_sectors_ids = [ResourceDict[rs] for rs in arrival_sectors]

    if f_num == 0:
        P.add_constraint(pc.sum(c1[:, departure_idx]) >= 1)
        P.add_constraint(pc.sum(c1[:, arrival_idx]) >= 1)
        P.add_constraint(pc.sum(c1[:, departure_sectors_ids]) >= 1)  # will this work if multiple sectors?
        P.add_constraint(pc.sum(c1[:, arrival_sectors_ids]) >= 1)  # will this work if multiple sectors?
    else:
        P.add_constraint(pc.sum(c2[:, departure_idx]) >= 1)
        P.add_constraint(pc.sum(c2[:, arrival_idx]) >= 1)
        P.add_constraint(pc.sum(c2[:, departure_sectors_ids]) >= 1)  # will this work if multiple sectors?
        P.add_constraint(pc.sum(c2[:, arrival_sectors_ids]) >= 1)  # will this work if multiple sectors?

## TODO: Flight Capability Constraints (weather/altitude/separation constraints)
## for now, everything is permissible
try:
    print(P)
    P.options.solver = "mosek"
    solution = P.solve()
    print(solution)
    print(list(GS.nodes()))
    print(np.array(c1.value))
    print(np.array(c2.value))

    # TODO: extract choices as numpy array
except pc.SolutionFailure as e:
    print('Infeasible')

    # REASONS FOR INFEASIBILITY AT THIS STEP
    # - GRID STATE IS SUCH THAT NO AVAILABLE RESOURCES CAN BE ASSIGNED, SOLUTION -> TOO BAD, SUBMIT REQUEST AT ANOTHER TIME (OR AUTO-SOLVE WITH SHIFTED TIME)

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

f1_res = np.array(c1.value)
f2_res = np.array(c2.value)

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
        if f1_res[t, j] == 1:
            p = get_grid_point(j)
            f1_points_xs.append(p[0])
            f1_points_ys.append(p[1])
    
    for j in range(f2_res.shape[1]):
        if f2_res[t, j] == 1:
            p = get_grid_point(j)
            f2_points_xs.append(p[0])
            f2_points_ys.append(p[1])
    
    axes[t].scatter(f1_points_xs, f1_points_ys, color='blue')
    axes[t].scatter(f2_points_xs, f2_points_ys, color='red')
    axes[t].set_title('t={}'.format(t))

fig.suptitle('PSU Step 1, S1 Blocked')
plt.subplot_tool()
plt.show()