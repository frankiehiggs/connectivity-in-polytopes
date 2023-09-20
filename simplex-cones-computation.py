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

def box_is_in_ball(corner,edge_length):
    """
    Given the first corner in lexicographic ordering of a box,
    """

@jit(nopython=True)
def angular_volume_lower_bound(centre, radius, spacing):
    """
    Finds a lower bound on the angular volume of a face
    by a "box counting" lower bound on the volume of a small ball
    placed at the midpoint of the face.
    midpoint and radius have to be chosen sensibly.
    
    This is a very crude method and could be made much faster,
    but we don't need a very small grid spacing to obtain
    the bounds we need to confirm which face attains the maximum
    over D(phi)/(d f_phi rho_phi),
    so I've not made much effort to optimise the speed.
    """
    r2 = radius*radius
    r2_on_s2 = (radius/spacing)**2
    N = int(radius/spacing) + 1
    unit_cube_points = np.array( [
        (0,0,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1),
        (0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),
        (1,0,0,0),(1,0,0,1),(1,0,1,0),(1,0,1,1),
        (1,1,0,0),(1,1,0,1),(1,1,1,0),(1,1,1,1)
    ])
    count = 0
    for x1 in range(-N,N+1):
        for x2 in range(-N,N+1):
            for x3 in range(-N,N+1):
                for x4 in range(-N,N+1):
                    x = np.array([x1,x2,x3,x4])*spacing
                    # First we check if the cube is in the simplex.
                    # (We only need to check the last point
                    #  in lexicographic order.)
                    lower_left = x + centre
                    if np.sum(lower_left) > 1 - 4*spacing or np.min(lower_left) < 0:
                        continue
                    # Next we check if the cube is in the ball.
                    is_in_ball = True
                    for v in unit_cube_points*spacing:
                        if np.dot(x+v,x+v) > r2:
                            is_in_ball = False
                            break
                    if not is_in_ball:
                        continue
                    # If we get this far, then the cube is in the
                    # intersection of the simplex and the small ball
                    # around the midpoint of the edge.
                    count += 1
    volume_lower_bound = count * (spacing**4)
    return volume_lower_bound / (radius**4)
    
@jit(nopython=True)
def angular_volume_upper_bound(centre, radius, spacing):
    """
    Finds an upper bound on the angular volume of a face
    by a "box counting" upper bound on the volume of a small ball
    placed at the midpoint of the face.
    midpoint and radius have to be chosen sensibly.
    """
    r2 = radius*radius
    r2_on_s2 = (radius/spacing)**2
    N = int(radius/spacing) + 2
    unit_cube_points = np.array( [
        (0,0,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1),
        (0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),
        (1,0,0,0),(1,0,0,1),(1,0,1,0),(1,0,1,1),
        (1,1,0,0),(1,1,0,1),(1,1,1,0),(1,1,1,1)
    ])
    count = 0
    for x1 in range(-N,N+1):
        for x2 in range(-N,N+1):
            for x3 in range(-N,N+1):
                for x4 in range(-N,N+1):
                    x = np.array([x1,x2,x3,x4])*spacing
                    # First we check if the cube touches the simplex.
                    # (We only need to check the first point
                    #  in lexicographic order.)
                    lower_left = x + centre
                    if np.sum(lower_left) > 1 or np.min(lower_left) < -spacing:
                        continue
                    # Next we check if the cube touches.
                    is_in_ball = False
                    for v in unit_cube_points*spacing:
                        if np.dot(x+v,x+v) <= r2:
                            is_in_ball = True
                            break
                    if not is_in_ball:
                        continue
                    # If we get this far, then the cube is in the
                    # intersection of the simplex and the small ball
                    # around the midpoint of the edge.
                    count += 1
    volume_upper_bound = count * (spacing**4)
    return volume_upper_bound / (radius**4)
        
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
    
    faces = [
        {
            'description':"Interior point.",
            'centre':np.array( [1/8, 1/8, 1/8, 1/8] ),
            'dim':4
        },{
            'description':"3d face touching the origin",
            'centre':np.array( [0, 1/4,1/4,1/4] ),
            'dim':3
        },{
            'description':"3d face NOT touching the origin",
            'centre':np.array( [1/4,1/4,1/4,1/4] ),
            'dim':3
        },{
            'description':"2d face touching the origin",
            'centre':np.array( [0, 0, 1/3, 1/3] ),
            'dim':2
        },{
            'description':"2d face NOT touching the origin",
            'centre':np.array( [0, 1/3, 1/3, 1/3] ),
            'dim':2
        },{
            'description':"1d face touching the origin",
            'centre':np.array( [0, 0, 0, 1/2] ),
            'dim':1
        },{
            'description':"1d face NOT touching the origin",
            'centre':np.array( [0,0,1/2,1/2] ),
            'dim':1
        },{
            'description':"vertex at the origin",
            'centre':np.array([0,0,0,0]),
            'dim':0
        },{
            'description':"vertex NOT at the origin",
            'centre':np.array([0,0,0,1]),
            'dim':0
        }
    ]
    
    r = 1/32
    
    ### First experiment:
    ### demonstrate that the maximum in the right-hand side of (2.6)
    ### is attained by a edge which is not incident to the origin,
    ### i.e. by a 1-d face which is the convex hull of two standard
    ### unit vectors.
    for face in faces:
        print(face['description'])
        lb = angular_volume_lower_bound(face['centre'],0.01,0.0002)
        print(f'Our lower bound on the angular volume is {lb},')
        dim = face['dim']
        if dim > 0:
            print(f'therefore an upper bound on the limit term is {0.25*dim/(24*lb)}\n')
        else:
            print(f'For a vertex, the limit term when k=1 is 0.\n')
    
    ### Next, the lower bound to prove our favourite edge is better than the other faces.
    face = faces[6]
    desc = face['description']
    dim = face['dim']
    print(f'To see that the maximum in (2.6) is attained at a {desc},')
    ub = angular_volume_upper_bound(face['centre'],0.01,0.0001)
    print(f'we calculate an upper bound on the angular volume {ub},')
    print(f'which gives us a lower bound on the limit term of {0.25*dim/(24*ub)},')
    print(f'more than any of the upper bounds for other types of face.')

    ### Second experiment:
    ### estimate the angular volume and hence limiting term
    ### at a 1-d edge not touching the origin
    ### (i.e. at a convex combination of two standard unit vectors).
    # ms = []
    # count = 1
    # while True:
        # X = KDTree(sample_points(N))
        # # X = sample_points(N)
        # m = test_volume(X, centre, r)
        # #print(f'We have {m} points out of {N}.')
        # ms.append(m)
        # print(f'\n{count} experiments')
        # print(description)
        # print(f'On average we had {np.mean(ms)} points out of {N}.')
        # rho = np.mean(ms) / (24*N*r**4)
        # print(f'Our estimate for rho_phi is therefore {rho}')
        # print(f'The limit term is therefore estimated as {0.25*dim / (24*rho)}')
        # count += 1
