graphDict = {} # maps names of graphs to graph objects.

class Graph(object):
    '''
    Creates a framework for storing nodes.
    '''
    def __init__(self, name, nodes = []):
        '''
    	Initializes graph with name name. Optional: pass in nodes.
        '''
        self.name = name
        self.nodes = nodes
        graphDict[self.name] = self

    def getName(self):
    	'''
    	Returns name of graph.
    	'''
        return self.name

    def getNodes(self):
    	'''
    	Returns list of nodes in this graph.
    	'''
        return self.nodes

    def addNode(self, node):
        '''
        Adds a node to the graph by name.
        '''
        if node not in self.getNodes():
            self.nodes.append(node)

    def __str__(self):
        '''
        Returns string representation of graph.
        '''
        return str(self.name)

    def __eq__(self, other):
        '''
        Returns True if graphs are equal.
        '''
        if self.name == other.name:
            return True
        return False

def getGraph(name):
    '''
    Returns graph object with name name. False if this graph does not exist.
    '''
    for key in graphDict.keys():
        if key == name:
            return graphDict[key]
    return False

distance = {} # maps 2-tuples of nodes to the distance between those two nodes on a link.

class Node(object):
    '''
    Creates a node object stored in a graph with a name and a list of linked nodes.
    '''
    def __init__(self, name, graph, links = []):
        '''
        Initializes node with name name under graph graph. Optional: pass in linked nodes.
        '''
        self.name = name
        self.graph = graph
        self.graph.addNode(self)
        self.links = []
        for node in links:
            self.addLink(node)

    def getName(self):
        '''
        Returns name of node.
        '''
        return self.name

    def getGraph(self):
        '''
        Returns graph object node is stored under.
        '''
        return self.graph

    def getLinks(self):
        '''
        Returns list of nodes linked to this node.
        '''
        return self.links
        
    def isLinked(self, other):
        '''
        Returns true if nodes are linked, False otherwise. Error if only one way link.
        '''
        if self.graph != other.getGraph():
	    return False
        dirs = 0
        if self in other.getLinks():
            dirs = 1
        if other in self.links:
            dirs += 1
        assert not dirs == 1
        if dirs == 2:
            return True
        else:
            return False

    def addLink(self, other, dist = 1):
        '''
        Adds link between this node and other node, distance given by optional argument.
        If nodes are already linked, the distance is changed to reflect the new argument passed in.
        '''
        assert self.graph == other.getGraph()
        if not self.isLinked(other):
            self.links.append(other)
            other.links.append(self)
        distance[(self, other)] = dist
        distance[(other, self)] = dist

    def __str__(self):
        '''
        Returns a string representation of this node.
        '''
        return self.name

    def __repr__(self):
        '''
        Returns a representation of this node.
        '''
        return self.name
        
    def __eq__(self, other):
        '''
        Returns true if two node objects are the same.
        '''
        if self.name == other.getName() and self.graph == other.getGraph():
            return True
        else:
	    return False

def getNode(name, graphname):
    '''
    Returns node object with name name. False if there is no such node in graph graphname.
    '''
    graph = getGraph(graphname)
    for node in graph.getNodes():
	if node.getName() == name:
	    return node
    return False
    
def addLink(node1, node2, dist = 1):
    '''
    Adds link between two nodes.
    '''
    node1.addLink(node2, dist)
    
def areLinked(node1, node2):
    return node1.isLinked(node2)
    
def dist(node1, node2):
    '''
    Returns distance between two nodes according to distance dictionary.
    '''
    assert node1.isLinked(node2)
    return distance[(node1, node2)]

def dijkstra(origin, destination, distf=dist):
    '''
    Uses Dijkstra's algorithm to find the shortest path of nodes between origin and destination.
    origin: origin node
    destination: destination node
    distf: function to measure distance; accepts two nodes as arguments.
    Returns: list of nodes to get to destination (including origin and destination)
    '''
    assert origin.getGraph() == destination.getGraph()
    graph = origin.getGraph()
    tendist = {}         # maps each node to the tentative distance to that node from the origin
    prev = {}            # maps each node to the previous node in shortest path to origin
    nodesleft = []       # list of nodes that have not been visited yet
    for node in graph.getNodes():
        tendist[node] = float('inf') # set the tentative distance to infinity for each node
        prev[node] = None            # leave the prev node undefined
        nodesleft.append(node)       # node has not been visited 
    tendist[origin] = 0
    current = None
    while len(nodesleft) != 0:
        mindist = float('inf')
        for node in nodesleft: # find the unvisited node with the shortest tentative distance
            if tendist[node] < mindist:
	        current = node
		mindist = tendist[node]
	for neighbor in current.getLinks():
	    if neighbor in nodesleft: # consider all unvisited neighbors of the current node
	        tempdist = tendist[current] + distf(current,neighbor) # the distance to the neighbor is distance to current node + distance between current and neighbor
		if tempdist < tendist[neighbor]: # if the path through currentnode is shorter than all previously considered paths
		    tendist[neighbor] = tempdist
		    prev[neighbor] = current
	nodesleft.remove(current) # current node is now considered visited
	if not destination in nodesleft:
	    break # the destination has been visited, and there is a shortest pathway
	connections = False
	for node in nodesleft:
            if not tendist[node] == float('inf'):
	        connections = True
		break
	if not connections:
	    break # there is no connection between the origin and destination
    current = destination
    pathway = [destination] # list to store path of nodes
    while prev[current] != None:
	pathway.append(prev[current])
	current = prev[current]
    if not connections:
	return False
    pathway.reverse() # pathway was constructed in opposite order, so reverse it
    return pathway
    
from random import randint

def gen():
    
    graph1 = Graph('graph1')

    a = Node('a', graph1)
    b = Node('b', graph1)
    c = Node('c', graph1)
    d = Node('d', graph1)
    e = Node('e', graph1)
    f = Node('f', graph1)
    g = Node('g', graph1)
    h = Node('h', graph1)
    i = Node('i', graph1)
    j = Node('j', graph1)
    k = Node('k', graph1)
    l = Node('l', graph1)
    m = Node('m', graph1)
    n = Node('n', graph1)
    o = Node('o', graph1)
    p = Node('p', graph1)
    q = Node('q', graph1)
    r = Node('r', graph1)
    s = Node('s', graph1)
    t = Node('t', graph1)
    u = Node('u', graph1)
    v = Node('v', graph1)
    w = Node('w', graph1)
    x = Node('x', graph1)
    y = Node('y', graph1)
    z = Node('z', graph1)
    
    for node1 in graph1.getNodes():
        for node2 in graph1.getNodes():
            if node1 != node2:
                if randint(0,10) < 2:
                    addLink(node1, node2, randint(1,10))
                    
    for node1 in graph1.getNodes():
        dict1 = {}
        for node2 in node1.getLinks():
            dict1[node2] = distance[(node1,node2)]
        print node1, dict1
    
    print 'dijkstra(a,z)'
    print dijkstra(a,z)
    return len(dijkstra(a,z))
    

    





