import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba_kdtree import KDTree
from tqdm import tqdm, trange
from datetime import datetime
import sys

@jit(nopython=True)
def cube_points(N):
    samples = np.random.random(size=(N,4))
    return samples

@jit(nopython=True)
def sample_points(N):
    samples = np.random.random(size=(N,4))
    for i in range(N):
        while np.sum(samples[i]) > 1:
            samples[i] = np.random.random(size=(4))
    return samples
    
@jit(nopython=True)
def test_volume(samples_tree, centre, r):
    # r2 = r*r
    # total = 0
    # for s in samples:
        # v = centre - s
        # if np.dot(v,v) <= r2:
            # total += 1
    # return total
    in_ball = samples_tree.query_radius(centre,r=r)[0]
    return len(in_ball)
    
if __name__=='__main__':
    """
    Polytope options are:
    - the 2d square [0,1]^2 (we take k=1, so beta = 0, and so there is no phase transition for narrow/wide angles in two dimensions).
    - the 3d cube [0,1]^3
    - a tetrahedron  (sharpest edge < pi/2)
    - a dodecahedron (sharpest edge > pi/2)
    - a 4d simplex
    """
    N = 5000000
    
    # description = "Interior point."
    # centre = np.array( [1/8, 1/8, 1/8, 1/8] ) # interior point
    # dim = 4
    
    # description = "3d face NOT touching the origin"
    # centre = np.array( [1/4,1/4,1/4,1/4] )  # 3d face without origin
    # dim = 3
    
    # description = "3d face touching the origin"
    # centre = np.array( [0, 1/4,1/4,1/4] ) # 3d face touching origin
    # dim = 3
    
    # description = "2d face touching the origin"
    # centre = np.array( [0, 0, 1/3, 1/3] ) # 2d face, touching origin
    # dim = 2
    
    # description = "2d face NOT touching the origin"
    # centre = np.array( [0, 1/3, 1/3, 1/3] ) # 2d face, not touching origin
    # dim = 2
    
    # description = "1d face touching the origin"
    # centre = np.array( [0, 0, 0, 1/2] ) # 1d face, touching origin
    # dim = 1
    
    description = "1d face NOT touching the origin"
    centre = np.array( [0,0,1/2,1/2] ) # 1d face, not touching origin
    dim = 1
    
    r = 1/32
    
    ms = []
    
    count = 1
    while True:
        X = KDTree(sample_points(N))
        # X = sample_points(N)
        m = test_volume(X, centre, r)
        #print(f'We have {m} points out of {N}.')
        ms.append(m)
        print(f'\n{count} experiments')
        print(description)
        print(f'On average we had {np.mean(ms)} points out of {N}.')
        rho = np.mean(ms) / (24*N*r**4)
        print(f'Our estimate for rho_phi is therefore {rho}')
        print(f'The limit term is therefore estimated as {0.25*dim / (24*rho)}')
        count += 1
