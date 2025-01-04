import numpy as np
import queue as q
import matplotlib.pyplot as plt

def loadNodes(file):
    
    # Initialize the empty nodes list
    N = []
    
    with open(file, encoding='utf-8') as f:
        next(f) # Skip the header
        
        # Convert attributes into list
        for row in f:
            try:
                x, y = row.split(sep=",")

                N.append((float(x), float(y)))
                
            except ValueError as e:
                print(f"Invalid row: {row.strip()} -> {e}")
                
    return N

def findClosestNode(x, y, nodes, tolerance):
    closest_node = None
    min_distance = float('inf')
    for idx, (nx, ny) in enumerate(nodes):
        distance = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
        if distance < min_distance and distance < tolerance:
            min_distance = distance
            closest_node = idx
    return closest_node

def loadEdges(file, nodes):

    # Initialize the edge dictionaries for all nodes
    E = {}
    ET = {}
    for i in range(len(nodes)):
        E[i] = []
        ET[i] = []
    
    with open(file, encoding='utf-8') as f:
        next(f)  # Skip the header
        
        try:
            for row in f:
                row = row.strip()
                x1, y1, x2, y2, order, lenght = row.split(sep=",")
                
                # Calculate weight parameters
                euklidean_distance = np.sqrt(((float(x2) - float(x1))**2 + (float(y2) - float(y1))**2))
                curvature = float(lenght)/euklidean_distance
                
                # Max speed assumption
                if int(order) == 1:
                    max_speed = 130
                elif int(order) == 2:
                    max_speed = 110
                elif int(order) == 3:
                    max_speed = 90
                elif int(order) == 4:
                    max_speed = 70
                else:
                    max_speed = 50
                    
                # Total weight conversion into travel time in seconds
                straight_time = float(lenght)/(max_speed/3.6)
                travel_time = straight_time*curvature
                
                # Find closest nodes for start and end points
                start_node = findClosestNode(float(x1), float(y1), nodes, 0.01)
                end_node = findClosestNode(float(x2), float(y2), nodes, 0.01)
                
                # Convert into dictionary if correct
                if start_node is not None and end_node is not None:  
                                  
                    E[start_node].append((start_node, end_node, float(lenght)))
                    E[end_node].append((end_node, start_node, float(lenght)))
                
                    ET[start_node].append((start_node, end_node, travel_time))
                    ET[end_node].append((end_node, start_node, travel_time))            

        except ValueError as e:
            print(f"Invalid edge row: {row.strip()} -> {e}")
        except IndexError as e:
            print(f"Edge points not found in nodes list: {row.strip()} -> {e}") 
            
    return E, ET

def cleanGraph(E):
    # Remove nodes with no edges from the graph
    return {node: neighbors for node, neighbors in E.items() if neighbors}

def fixNegativeValues(E):
    # Detect negative weights
    min_weight = float('inf')
    for edges in E.values():
        for _, _, w in edges:
            if w < min_weight:
                min_weight = w

    if min_weight < 0:
        total_offset = abs(min_weight)
        # Add offset to all edges
        for edges in E.values():
            for i in range(len(edges)):
                start, end, weight = edges[i]
                edges[i] = (start, end, weight + total_offset)
        return E, total_offset
    else:
        return E, 0

def cancelOffset(result, offset, edges):
    # Cancel out fixNegativeValues trick
    original = result - (offset * len(edges))    
    
    return original

def dijkstra(N, E, u, v):
    # List of predecessors and list of minimal distances
    P = [-1] * len(N)
    D = [np.inf] * len(N)
    
    # Create priority queue
    PQ = q.PriorityQueue()
    
    # Starting node
    D[u] = 0
    PQ.put((D[u], u))
    
    # Repeat until PQ is empty
    while not PQ.empty():
        # Remove node with lowest value d[]
        _, u = PQ.get()
        
        # Check if node u has edges
        if u not in E:
            continue
            
        # Browse adjacent nodes
        for _, v, wuv in E.get(u, []):
            # Relaxation
            if D[v] > D[u] + wuv:
                D[v] = D[u] + wuv
                P[v] = u
                # Add updated node to PQ
                PQ.put((D[v], v))
                
    return P, D[v]

def reconstruct_path(P, u, v):
    path = []
    
    # Path shortening
    while v != u and v !=-1:
        path.append(v)
        v = P[v]
        
    path.append(v)
    if v != u:
        print('Incorrect path')
    return (path)

def jarnik(N, E):
    
    min_span_tree = []
    total_weight = 0
    P = [False] * len(N)  # Track visited nodes
    
    # Create the priority queue to store edges with their weights
    PQ = q.PriorityQueue()
    
    # Start with node 0
    u = 0
    P[u] = True
    
    # Add all edges from the start node to the priority queue
    for _, v, weight in E[u]:
        PQ.put((weight, u, v))
    
    while not PQ.empty() and len(min_span_tree) < len(N) - 1:
        weight, u, v = PQ.get()
        
        # If the end node has already been visited, skip this edge
        if P[v]:
            continue
        
        # Add this edge to the MST
        min_span_tree.append((u, v, weight))
        total_weight += weight
        P[v] = True
        
        # Add all edges from the newly added node to the priority queue
        for _, next_node, next_weight in E[v]:
            if not P[next_node]:
                PQ.put((next_weight, v, next_node))
    
    return min_span_tree, total_weight

def graphVisualization(N, E, path):

    plt.figure(figsize=(10, 8))
    
    # Extract x and y coordinates for nodes
    node_x = [n[0] for n in N]
    node_y = [n[1] for n in N]
    
    # Plot edges
    for start_node, neighbors in E.items():
        for edge in neighbors:
            end_node = edge[1]
            start_coords = N[start_node]
            end_coords = N[end_node]
            plt.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], 'k-', lw=0.5)
            
    # Plot nodes
    plt.scatter(node_x, node_y, c='blue', s=2, zorder=2)
    
    # Plot path
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        x_coords = [N[start][0], N[end][0]]
        y_coords = [N[start][1], N[end][1]]
        plt.plot(x_coords, y_coords, color='red', linewidth=2, zorder=3)
    
    plt.show()

import matplotlib.pyplot as plt

def minimalSpanningTree(N, E, tree):
    
    plt.figure(figsize=(10,8))
    
    # Plot edges
    for start_node, neighbors in E.items():
        for edge in neighbors:
            end_node = edge[1]
            start_coords = N[start_node]
            end_coords = N[end_node]
            plt.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], 'k-', lw=0.5)
    
    # Plot minimal spanning tree
    for start, end, _ in tree:
        x_start, y_start = N[start]
        x_end, y_end = N[end]
        plt.plot([x_start, x_end], [y_start, y_end], 'r-', lw=2)

    plt.show()

# Load and prepare data
file_nodes = 'C:/Coding/GEOINFO/Graf/vertex.txt'
file_edges = 'C:/Coding/GEOINFO/Graf/line.txt'

N = loadNodes(file_nodes)
E, ET = loadEdges(file_edges, N)

# Negative edges fix
E, offset = fixNegativeValues(E)
ET, offsetT = fixNegativeValues(ET)

# Dijkstra (Euclidean distance and travel time)
# Start (u) and end (v)
u = 1
v = 2000

# Dijkstra 
P, dmin = dijkstra(N, E, u, v)
path = reconstruct_path(P, u, v)
P_T, time = dijkstra(N, ET, u, v)
path_T = reconstruct_path(P_T, u, v)

# Fix if graph contains negative weights
if offset != 0:
    dmin = cancelOffset(dmin, offset, P)
    time = cancelOffset(time, offset, P_T)

# Results
print(f"Traveling distance: {dmin/1000} km.")
print(f"Traveling time: {time/60} minutes.")
graphVisualization(N, E, path)
graphVisualization(N, ET, path_T)

# Jarnik/Prim (minimal spanning tree)
tree, w = jarnik(N, E)
tree_T, time_j = jarnik(N, ET)

# Fix if graph contains negative weights
if offset != 0:
    w = cancelOffset(w, offset, tree)
    time_j = cancelOffset(time_j, offset, tree_T)

# Results
print(f"Minimum spanning tree lenght: {w/1000} km.")
print(f"Minimum spanning tree travel time: {time_j/60} minutes.")
minimalSpanningTree(N, E, tree)
minimalSpanningTree(N, ET, tree_T)