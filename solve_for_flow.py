'''
This module has been totaly reconstructed.
--By Jinliang
'''

import numpy as np

def solve_for_flow(G, Pin, Pout, H=None):
    """
    Solve for flow in the bifurcating vessel network (Loop Structure).
    Toplogy:
    - Node 0: Inlet
    - Node 5: Bifurcation (Bottom path & Top path start)
    - Node 15: Reunion (Bottom path & Top path meet)
    - Node 20: Outlet
    - Nodes 21-39: Internal nodes of the Top Path
    """
    Nn = 40  # Total nodes
    Nseg = 40 # Total segments
    
    # Eposilon to prevent zero conductance
    G[G == 0] = 1e-25
    
    P = np.zeros(Nn)
    Q = np.zeros(Nseg)
    C = np.zeros((Nn, Nn))
    B = np.zeros(Nn)

    # Constract Linear equations C * P = B

    # 1. Boundary Conditions
    # Node 0 (Inlet)
    C[0, 0] = 1.0
    B[0] = Pin
    
    # Node 20 (Outlet)
    C[20, 20] = 1.0
    B[20] = Pout

    # 2. Linear connections in Lower Loop
    # Seg i connects Node i -> Node i+1
    # Nodes 1-4, 6-14, 16-19
    linear_nodes_lower = list(range(1, 5)) + list(range(6, 15)) + list(range(16, 20))
    for node in linear_nodes_lower:
        # Conservation: Q_in = Q_out => G[node-1]*(P[node-1]-P[node]) = G[node]*(P[node]-P[node+1])
        C[node, node-1] = -G[node-1]
        C[node, node]   =  G[node-1] + G[node]
        C[node, node+1] = -G[node]

    # 3. Upper Loop internals
    # Nodes 21-39.
    # Seg 20 connects Node 5 -> Node 21
    # Seg 21 connects Node 21 -> Node 22
    # ...
    # Seg 38 connects Node 38 -> Node 39
    # Seg 39 connects Node 39 -> Node 15 (Reunion)
    
    # First node in upper loop (Node 21)
    # In: Seg 20 (from Node 5), Out: Seg 21 (to Node 22)
    C[21, 5]  = -G[20]
    C[21, 21] =  G[20] + G[21]
    C[21, 22] = -G[21]
    
    # Middle nodes in upper loop (22 to 38)
    for node in range(22, 39):
        # In: Seg node-1 (from node-1), Out: Seg node (to node+1)
        C[node, node-1] = -G[node-1]
        C[node, node]   =  G[node-1] + G[node]
        C[node, node+1] = -G[node]
        
    # Last node in upper loop (Node 39)
    # In: Seg 38 (from Node 38), Out: Seg 39 (to Node 15)
    C[39, 38] = -G[38]
    C[39, 39] =  G[38] + G[39]
    C[39, 15] = -G[39]

    # 4. Bifurcation Node 5
    # In: Seg 4 (from Node 4)
    # Out 1: Seg 5 (to Node 6)
    # Out 2: Seg 20 (to Node 21)
    # Eq: Q4 = Q5 + Q20
    # G[4]*(P4-P5) = G[5]*(P5-P6) + G[20]*(P5-P21)
    C[5, 4]  = -G[4]
    C[5, 5]  =  G[4] + G[5] + G[20]
    C[5, 6]  = -G[5]
    C[5, 21] = -G[20]

    # 5. Reunion Node 15
    # In 1: Seg 14 (from Node 14)
    # In 2: Seg 39 (from Node 39)
    # Out: Seg 15 (to Node 16)
    # Eq: Q14 + Q39 = Q15
    # G[14]*(P14-P15) + G[39]*(P39-P15) = G[15]*(P15-P16)
    C[15, 14] = -G[14]
    C[15, 39] = -G[39]
    C[15, 15] =  G[14] + G[39] + G[15]
    C[15, 16] = -G[15]

    P = np.linalg.solve(C, B)
    
    # Calculate flow Q
    # Also follow the topology
    
    # Lower Path (Seg 0-19)
    # Seg 0-4
    for seg in range(5):
        Q[seg] = G[seg] * (P[seg] - P[seg+1])
    
    # Seg 5-14
    for seg in range(5, 15):
        Q[seg] = G[seg] * (P[seg] - P[seg+1])
        
    # Seg 15-19
    for seg in range(15, 20):
        if seg == 19:
            # Last segment P20 is fixed boundary Pout
            Q[seg] = G[seg] * (P[seg] - P[20])
        else:
            Q[seg] = G[seg] * (P[seg] - P[seg+1])
            
    # Upper Path (Seg 20-39)
    # Seg 20: Node 5 -> 21
    Q[20] = G[20] * (P[5] - P[21])
    
    # Seg 21-38: Node seg -> seg+1
    for seg in range(21, 39):
        Q[seg] = G[seg] * (P[seg] - P[seg+1])
        
    # Seg 39: Node 39 -> 15
    Q[39] = G[39] * (P[39] - P[15])
    
    # Compute shear stress if H is provided
    if H is not None:
        tau = H * Q
        return P, Q, tau
    else:
        return P, Q
