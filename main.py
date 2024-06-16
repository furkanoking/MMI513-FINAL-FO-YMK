import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

def undirectedconnectiongraph(xnum=30, ynum=30):
    G = {'V':[], 'E':[]} # We will use a dictionary for simplicity
    for xind in range(xnum):
      for yind in range(ynum):
        G['V'].append((xind, yind))

    # Traverse north first
    for pt in G['V']:
      vtn = north(pt[0], pt[1])
      if isvertex(vtn, G['V']):
        G['E'].append((pt, vtn))

    # Traverse east second
    for pt in G['V']:
      vte = east(pt[0], pt[1])
      if isvertex(vte, G['V']):
        G['E'].append((pt, vte))
    return G

def north(xind, yind):
    node = (xind, yind + 1)
    return node
def south(xind, yind):
    node = (xind, yind - 1)
    return node

def east(xind, yind):
    node = (xind + 1, yind)
    return node

def west(xind, yind):
    node = (xind - 1, yind)
    return node

def isvertex(node, vertices):
    return node in vertices

def plotgraph(G, vertexflag=True):
    for e in G['E']:
      vec = np.array([e[1][0]-e[0][0], e[1][1]-e[0][1]])
      ort = np.array([-vec[1], vec[0]])
      olen = np.linalg.norm(ort)
      ort = ort / olen
      sum = np.array([(e[1][0]+e[0][0])/2, (e[1][1]+e[0][1])/2])
      startp = sum - ort / 2
      endp = sum + ort / 2
      plt.plot((startp[0], endp[0]), (startp[1], endp[1]), 'k', linewidth=10)
      if vertexflag:
        for v in G['V']:
          plt.plot(float(v[0]), float(v[1]), 'ro')
    plt.axis('square')
    plt.show()

def neighbourhood(node, vertices):
    pneighbours = [north(node[0], node[1]), south(node[0], node[1]), east(node[0], node[1]), west(node[0], node[1])]
    neighbours = []
    for ind in range(len(pneighbours)):
      n = pneighbours[ind]
      if(isvertex(n, vertices) == True):
        neighbours.append(n)
    return set(neighbours)

def randomnode(vertices):
    vertices = list(vertices)
    randind = np.random.randint(0, len(vertices))
    return vertices[randind]

def subsetinset(query, settosearch): # Used also in Kruskal's method
    subset = set()
    for element in settosearch:
        if len(query.difference(element))==0:
          subset = subset.union(set(element))
    return frozenset(subset)

def frozenunion(S1, S2):
    Sunion = set()
    for element in S1:
      Sunion = Sunion.union(set([element]))
    for element in S2:
      Sunion = Sunion.union(set([element]))
    return frozenset(Sunion)

# For default argument
H=undirectedconnectiongraph(30, 30)

def pimmsmaze(G=H):


  #assert part idea is taken from ChatGPT
  # Check if G is None
  assert G is not None, "Input graph G cannot be None"

  # Check if G has keys 'V' and 'E'
  assert 'V' in G and 'E' in G, "Input graph G must have keys 'V' and 'E'"

  # create a set of vertices
  vertices_set=set(G['V'].copy())

  # copy the edges of the graph
  W=set(G['E'].copy())

  # all the cells are unvisited
  C=set()

  # Create an empty set to sote candiate walls
  L=set()

  # Create an empty set for the maze
  M=set()

  # Choose a random starting cell
  c=randomnode(vertices_set)

  # # Add walls adjacent to the starting cell to the candidate set
  for wall in W.copy():
    if c == wall[0] or c == wall[1]:
      L.add(wall)

  while len(L)!=0:

    # Choose a random candidate wall
    l=random.choice(list(L))

    # Find common elements between the candidate wall and visited cells
    common_elements=set(l).intersection(C)
    if len(common_elements)<= 1:
      # Mark the cells of the candidate wall as visited
      C.add(l[0])
      C.add(l[1])

      # Remove the candidate wall from the maze
      W.remove(l)

      # Add new candidate walls adjacent to the visited cells
      for wall in W.copy():
        if l[0] in wall or l[1] in wall:
          if wall not in L:
            L.add(wall)

    # Remove the current candidate wall from consideration
    L.remove(l)

  # Create the maze dictionary with vertices and edges
  M=dict()
  M['V'] = G['V'].copy()
  M['E'] = list(W)
  return M


# Set figure size once at the beginning
plt.rcParams['figure.figsize'] = [15, 15]

# First plot
plt.clf()
plt.close()

G = undirectedconnectiongraph(30, 30)
#Testing the maze
hop = pimmsmaze(G)
plt.title('pimmsmaze maze')
plotgraph(hop, False)


