'''
This file has been minimal updated.
--By Jinliang
'''

import numpy as np

def make_segments(L):
    # You do not need 6 direction vectors
    v_up = np.array([0, 1])
    v_right = np.array([1, 0])
    v_down = np.array([0, -1])

    # Lower Path
    # Seg 0-4: Inlet (Vertical Up)
    vessel1 = np.zeros((6, 2))
    for i in range(5):
        vessel1[i+1] = vessel1[i] + v_up * L[i] * 1e6

    # Seg 5-14: Lower Channel (Horizontal Right)
    vessel2 = np.zeros((11, 2))
    vessel2[0] = vessel1[5] # Start where vessel1 ends
    for i in range(10):
        vessel2[i+1] = vessel2[i] + v_right * L[5+i] * 1e6

    # Seg 15-19: Outlet (Vertical Down)
    vessel3 = np.zeros((6, 2))
    vessel3[0] = vessel2[10] # Start where vessel2 ends
    for i in range(5):
        vessel3[i+1] = vessel3[i] + v_down * L[15+i] * 1e6

    # Combine lower part points (Seg 0-19)
    # Use [1:] to remove duplicate connection points, so 20 segments correspond to 21 points
    pts_lower = np.vstack((vessel1, vessel2[1:], vessel3[1:]))

    # Upper Path
    # Seg 20-24: Upper Branch Up (Vertical Up from Bifurcation)
    vessel4 = np.zeros((6, 2))
    vessel4[0] = vessel1[5] # Start at Node 5 (Bifurcation point)
    for i in range(5):
        vessel4[i+1] = vessel4[i] + v_up * L[20+i] * 1e6
        
    # Seg 25-34: Upper Branch Across (Horizontal Right)
    vessel5 = np.zeros((11, 2))
    vessel5[0] = vessel4[5]
    for i in range(10):
        vessel5[i+1] = vessel5[i] + v_right * L[25+i] * 1e6
        
    # Seg 35-39: Upper Branch Down (Vertical Down to Reunion)
    vessel6 = np.zeros((6, 2))
    vessel6[0] = vessel5[10]
    for i in range(5):
        vessel6[i+1] = vessel6[i] + v_down * L[35+i] * 1e6

    # Combine upper part points (Seg 20-39)
    # remove duplicate connection points
    pts_upper = np.vstack((vessel4, vessel5[1:], vessel6[1:]))

    # Final merge
    segments = np.vstack((pts_lower, pts_upper))

    return segments
