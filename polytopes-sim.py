import numpy as np
import matplotlib.pyplot as plt
import polytope
from numba import jit
from numba_kdtree import KDTree
from tqdm import tqdm, trange
from datetime import datetime
import sys
from matplotlib.ticker import FormatStrFormatter

def limit_and_dim(shape):
    if shape == 'tetrahedron':
        constant = 1 / (12 * np.sqrt(2) * np.arccos(1/3))
        dim = 3
    elif shape == 'square':
        constant = 1/np.pi
        dim = 2
    elif shape == 'cube':
        constant = 1/np.pi
        dim = 3
    elif shape == 'dodecahedron':
        constant = (15 + 7*np.sqrt(5))/(4*np.pi)
        dim = 3
    elif shape == '4d-simplex':
        constant = 0 # I haven't worked this out yet. I suspect it occurs one of the 2d faces which do not intersect the origin
        # i.e. on the convex hull of three of the standard unit vectors.
        dim = 4
    elif shape=='cut-simplex':
        constant = 0.04998774884163379 # Estimated numerically
        dim = 4
    return constant, dim

def sample_points(N,domain):
    """
    Samples N independent uniform random points in the given domain.
    Returns a numpy array of shape (N,dim), where dim is the dimension
    of the domain.
    For most domains it just uses rejection sampling,
    i.e. picks a uniform point in a box and re-samples a point
    if it lies outside the target domain.
    """
    if domain == 'square':
        samples = np.random.random(size=(N,2))
    elif domain == 'cube':
        samples = np.random.random(size=(N,3))
    elif domain == 'tetrahedron':
        """
        The convex hull of the points
        (0,0,0), (0,1,1), (1,0,1), (1,1,0)
        is a tetrahedron which is a subset of [0,1]^3
        and has side length sqrt(2)
        The angle between faces is arccos(1/3).
        Gardner, M. "Tetrahedrons." Ch. 19 in The Sixth Book of Mathematical Games from Scientific American. Chicago, IL: University of Chicago Press, pp. 183-194, 1984.
        We divide all the coordinates by sqrt(2) at the end
        so that we have a tetrahedron with side length 1.
        """
        tetra = polytope.qhull(np.array([(0,0,0),(0,1,1),(1,0,1),(1,1,0)]))
        samples = np.random.random(size=(N,3))
        for i in range(N):
            while not samples[i] in tetra:
                samples[i] = np.random.random(size=3)
        samples /= np.sqrt(2)
    elif domain == 'dodecahedron':
        """
        For convenience we use a dodecahedron with side lengths
        sqrt(5) - 1, then rescale after sampling the points
        so that they are inside a dodecahedron of side length 1.
        """
        phi = 0.5*(1 + np.sqrt(5))
        dodeca = polytope.qhull(np.array([
            ( 1, 1, 1), ( 1, 1,-1), ( 1,-1, 1), ( 1,-1,-1),
            (-1, 1, 1), (-1, 1,-1), (-1,-1, 1), (-1,-1,-1),
            (0,phi,1/phi), (0,phi,-1/phi),
            (0,-phi,1/phi), (0,-phi,-1/phi),
            (1/phi,0,phi),(1/phi,0,-phi),
            (-1/phi,0,phi),(-1/phi,0,-phi),
            (phi,1/phi,0),(phi,-1/phi,0),
            (-phi,1/phi,0),(-phi,-1/phi,0)
        ]))
        samples = 2*np.sqrt(3)*(np.random.random(size=(N,3))-0.5)
        for i in range(N):
            while not samples[i] in dodeca:
                samples[i] = 2*np.sqrt(3)*(np.random.random(size=3)-0.5)
        samples /= np.sqrt(5) - 1
    elif domain == '4d-simplex':
        samples = 2*np.random.random(size=(N,4)) - 1
        for i in range(N):
            while np.sum(np.abs(samples[i])) > 1:
                samples[i] = 2*np.random.random(size=(4)) - 1
    elif domain == 'cut-simplex':
        samples = np.random.random(size=(N,4))
        for i in range(N):
            while np.sum(samples[i]) > 1:
                samples[i] = np.random.random(size=(4))
    else:
        raise Exception(f'Domain "{domain}" not supported')
    return samples
    
@jit(nopython=True)
def lnnl(points_tree):
    distances = points_tree.query(points_tree.data,k=2)[0][:,1]
    return max(distances)
    
@jit(nopython=True)
def find_connectivity_threshold(points, lower_bound, precision=0.00000001):
    """
    Given a numpy array of shape (N,d)
    (N points in R^d),
    and a radius r,
    tests if the graph formed by connecting any two points
    at distance <= r
    is connected.
    If it's connected, it returns 0,
    and if it's not connected it returns the smallest radius
    greater than r needed to connect an additional vertex to 
    the component containing points[0].
    """
    N = points.shape[0]
    ## Start with vertex 0
    # component                 = [0]
    # unchecked_active_vertices = [0]
    
    ## Start with a random vertex
    s = np.random.randint(0,N)
    component                 = [s]
    
    r2 = lower_bound * lower_bound
    
    while True:
        unchecked_active_vertices = component.copy()
        while unchecked_active_vertices:
            v = points[unchecked_active_vertices.pop()]
            for i in range(N):
                if not i in component:
                    increment = v - points[i]
                    if np.dot(increment,increment) <= r2 + precision:
                        component.append(i)
                        unchecked_active_vertices.append(i)
        if len(component) == N:
            break
        else:
            running_min = np.inf
            for i in component:
                v = points[i]
                for j in range(N):
                    if not j in component:
                        increment = v - points[j]
                        running_min = min(running_min, np.dot(increment,increment))
            if running_min <= r2:
                raise Exception("There shouldn't be a point this close to the component that isn't already in the component.")
            r2 = running_min
    return np.sqrt(r2)

# def produce_diagram(Nmax,shape):
    # X = sample_points(Nmax,shape)
    # lnnls = []
    # connectivity_thresholds = []
    
    # for n in trange(2,Nmax):
        # L = lnnl(KDTree(X[:n]))
        # lnnls.append(L)
        # R = find_connectivity_threshold(X[:n],L)
        # connectivity_thresholds.append(R)
    
    # fig, ax = plt.subplots()
    # ax.plot(range(2,Nmax),lnnls,'k--',linewidth=1.5,label="LNNL")
    # ax.plot(range(2,Nmax),connectivity_thresholds,'r--',linewidth=1.5,label="Connectivity threshold")
    # def f(x):
        # return np.sqrt(np.pi**(-1)*np.log(x) / x)
    # ax.plot(range(2,Nmax),f(range(2,Nmax)),'b--',linewidth=1.0,label="Theoretical curve")
    # ax.legend(loc='upper right')
    # plt.show()
    # plt.close()

def produce_normalised_diagram(Nmax,shape,outname=None):
    # X = sample_points(Nmax,shape)
    # ns = [int(10**(x/30)) for x in range(90,121)]
    OUTLENGTH = 500
    step = max(int( (Nmax+1)/OUTLENGTH ), 1)
    ns = range(2,Nmax+1, step)
    X = sample_points(max(ns),shape)
    constant, dim = limit_and_dim(shape)
    lnnls = []
    connectivity_thresholds = []
    
    #for n in trange(100,Nmax):
    progress = tqdm(ns)
    for n in progress:
        progress.set_description(f'n = {n}')
        L = lnnl(KDTree(X[:n]))
        R = find_connectivity_threshold(X[:n],L)
        lnnls.append(n*L**dim/np.log(n))
        connectivity_thresholds.append(n*R**dim / np.log(n))
    
    fig, ax = plt.subplots()
    ax.plot(ns,lnnls,'k',linewidth=1.5,label="LNNL")
    ax.plot(ns,connectivity_thresholds,'r',linewidth=1.5,label="Connectivity threshold")
    ax.plot(ns,np.ones(len(ns))*constant,'b--',linewidth=1.0)
    ax.legend(loc='upper right')
    ax.set_ylim(0,max(connectivity_thresholds)+0.1)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('$n$')
    fig.tight_layout()
    if not outname:
        plt.show()
    else:
        fig.savefig(outname)
    plt.close()

# def lnnl_diagram(Nmax,shape):
    # X = sample_points(Nmax,shape)
    # lnnls = []
    # if shape == 'tetrahedron':
        # constant = 1 / (12 * np.sqrt(2) * np.arccos(1/3))
        # dim = 3
    # elif shape == 'square':
        # constant = 1/np.pi
        # dim = 2
    
    # for n in trange(2,Nmax):
        # L = lnnl(KDTree(X[:n]))
        # lnnls.append(L)
            
    # fig, ax = plt.subplots()
    # ax.plot(range(2,Nmax),lnnls,'k--',linewidth=1.5,label="LNNL")
    # def f(x):
        # return (constant*np.log(x) / x)**(1/dim)
    # ax.plot(range(2,Nmax),f(range(2,Nmax)),'b--',linewidth=1.0,label="Theoretical curve")
    # ax.legend(loc='upper right')
    # plt.show()
    # plt.close()

# def normalised_lnnl_diagram(Nmax,shape):
    # X = sample_points(Nmax,shape)
    # lnnls = []
    # if shape == 'tetrahedron':
        # constant = 1 / (12 * np.sqrt(2) * np.arccos(1/3))
        # dim = 3
    # elif shape == 'square':
        # constant = 1/np.pi
        # dim = 2
    
    # for n in trange(2,Nmax):
        # L = lnnl(KDTree(X[:n]))
        # lnnls.append(n*L**dim / np.log(n))
            
    # fig, ax = plt.subplots()
    # ax.plot(range(2,Nmax),lnnls,'k--',linewidth=1.5,label="LNNL")
    # def f(x):
        # return (constant*np.log(x) / x)**(1/dim)
    # ax.plot(range(2,Nmax),np.ones(len(range(2,Nmax)))*constant,'b--',linewidth=1.0,label="Theoretical curve")
    # ax.legend(loc='upper right')
    # plt.show()
    # plt.close()


if __name__=='__main__':
    """
    Polytope options are:
    - the 2d square [0,1]^2 (we take k=1, so beta = 0, and so there is no phase transition for narrow/wide angles in two dimensions).
    - the 3d cube [0,1]^3 (critical shape)
    - a tetrahedron  (sharpest edge < pi/2)
    - a dodecahedron (sharpest edge > pi/2)
    - a 4d simplex
    """
    shapes = ['square', 'cube', 'tetrahedron', 'dodecahedron','4d-simplex', 'cut-simplex']
    # choice = np.random.randint(0,6)
    # shape = shapes[choice]
    # shape = 'cut-simplex'
    
    if len(sys.argv) >= 2:
        Nmax = int(sys.argv[1])
    else:
        Nmax = 1000
    if len(sys.argv) >= 3:
        shape = sys.argv[2]
    else:
        choice = np.random.randint(0,6)
        shape = shapes[choice]
    
    print(f'{Nmax} points in the {shape}.')
    
    now = datetime.now()
    seed = int(now.strftime("%d%H%M%S"))
    np.random.seed(seed)
    
    # produce_normalised_diagram(shape)
    # produce_diagram(1000,shape)
    produce_normalised_diagram(Nmax,shape,outname=f'diagrams/pdfs/{shape}-{now.strftime("%y-%m-%d-%H:%M:%S")}.pdf')
