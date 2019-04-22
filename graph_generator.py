import random
from collections import defaultdict
import numpy as np

from io_parser import DotParser, PickleParser
from models import Graph, StackLinearLayout


def generate_maximal_bipartite_graph(num_vertices_1, num_vertices_2):
    graph = Graph()

    for i in range(1, num_vertices_1 + 1):
        for j in range(1, num_vertices_2 + 1):
            graph.add_edge('A' + str(i), 'B' + str(j))

    DotParser().write(graph)
    return graph


def generate_planer_bipartite_graph(num_vertices, edge_density=0.5,
                                    max_degree=None):
    graph = Graph()
    stack1 = []
    stack2 = []
    vertices_degree = defaultdict(int)
    for i in range(1, num_vertices):
        graph.add_vertex(i)

    max_edges = 2 * num_vertices - 4
    while len(stack1) + len(stack2) < max_edges * edge_density:
        v1 = random.choice(range(1, num_vertices + 1, 2))
        v2 = random.choice(range(2, num_vertices + 1, 2))
        if v1 > v2:
            v1, v2 = v2, v1
        edge = (v1, v2)

        if max_degree and (vertices_degree[v1] >= max_degree or
                           vertices_degree[v2] >= max_degree):
            continue

        if edge in stack1 or edge in stack2:
            continue

        stacks = [stack1, stack2]
        random.shuffle(stacks)
        for stack in stacks:
            fits_to_stack = True
            for w1, w2 in stack:
                if v1 < w1 < v2 < w2 or w1 < v1 < w2 < v2:
                    fits_to_stack = False
                    break

            if fits_to_stack:
                graph.add_edge(v1, v2)
                stack.append(edge)
                vertices_degree[v1] += 1
                vertices_degree[v2] += 1
                break

    layout = StackLinearLayout(graph)
    layout.order = list(range(1, num_vertices + 1))
    layout.stack1 = stack1
    layout.stack2 = stack2

    DotParser().write(graph)
    PickleParser().write(layout)

    return graph


def generate_random_graph(num_vertices, num_edges):
    graph = Graph()

    for i in range(1, num_vertices + 1):
        graph.add_vertex(i)

    #max_edges = (num_vertices * (num_vertices - 1)) // 2
    while len(graph.edges) < num_edges:
        v1 = random.choice(range(1, num_vertices + 1))
        v2 = random.choice(range(1, num_vertices + 1))

        if v1 != v2 and not graph.has_edge(v1, v2):
            graph.add_edge(v1, v2)

    return graph


def generate_k_tree(num_vertices, k):
    if k < 0 or k + 1 > num_vertices:
        return

    graph = Graph()
    for i in range(num_vertices):
        graph.add_vertex(i)

    # get first(k + 1) - clique
    for i in range(k):
        for j in range(i + 1, k + 1):
            graph.add_edge(i, j)

    # store k - cliques
    cliques = []
    for i in range(k + 1):
        clique = []
        for j in range(k + 1):
            if j != i:
                clique.append(j)
        cliques.append(clique)

    # Add vertices
    next_vertex = k + 1
    while next_vertex < num_vertices:
        clique = random.choice(cliques)
        for i in range(len(clique)):
            graph.add_edge(clique[i], next_vertex)

        # add new cliques
        for i in range(k):
            new_clique = clique.copy()
            new_clique[i] = next_vertex
            cliques.append(new_clique)

        next_vertex += 1

    return graph


def generate_complete_graph(num_vertices):
    graph = Graph()
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            graph.add_edge(i, j)

    return graph


def generate_planar_graph(num_vertices):
    # Create a random set of points
    points = np.random.random((num_vertices, 2))

    # Create Delaunay Triangulation
    dt = Delaunay2d()
    for s in points:
        dt.addPoint(s)

    triangles = dt.exportTriangles()
    graph = Graph()
    for v1, v2, v3 in triangles:
        # add_edges() does not add a edge twice if it already exists
        graph.add_edge(v1, v2)
        graph.add_edge(v1, v3)
        graph.add_edge(v2, v3)

    return graph


# https://github.com/jmespadero/pyDelaunay2D
class Delaunay2d:
    """
    Class to compute a Delaunay triangulation in 2D
    ref: http://en.wikipedia.org/wiki/Bowyer-Watson_algorithm
    ref: http://www.geom.uiuc.edu/~samuelp/del_project.html
    """

    def __init__(self, center=(0, 0), radius=9999):
        """ Init and create a new frame to contain the triangulation
        center -- Optional position for the center of the frame. Default (0,0)
        radius -- Optional distance from corners to the center.
        """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [center+radius*np.array((-1, -1)),
                       center+radius*np.array((+1, -1)),
                       center+radius*np.array((+1, +1)),
                       center+radius*np.array((-1, +1))]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

    def circumcenter(self, tri):
        """Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                      [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance
        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)

    def inCircleFast(self, tri, p):
        """Check if point p is inside of precomputed circumcircle of tri.
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def inCircleRobust(self, tri, p):
        """Check if point p is inside of circumcircle around the triangle tri.
        This is a robust predicate, slower than compare distance to centers
        ref: http://www.cs.cmu.edu/~quake/robust.html
        """
        m1 = np.asarray([self.coords[v] - p for v in tri])
        m2 = np.sum(np.square(m1), axis=1).reshape((3, 1))
        m = np.hstack((m1, m2))    # The 3x3 matrix to check
        return np.linalg.det(m) <= 0

    def addPoint(self, p):
        """Add a point to the current DT, and refine it using Bowyer-Watson.
        """
        p = np.asarray(p)
        idx = len(self.coords)
        # print("coords[", idx,"] ->",p)
        self.coords.append(p)

        # Search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            # Choose one method: inCircleRobust(T, p) or inCircleFast(T, p)
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        boundary = []
        # Choose a "random" triangle and edge
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # Check if edge of triangle T is on the boundary...
            # if opposite triangle of this edge is external to the list
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Insert edge and external triangle into boundary list
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))

                # Move to next CCW edge in this triangle
                edge = (edge + 1) % 3

                # Check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Move to next CCW edge in opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Remove triangles too near of point p of our solution
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # Retriangle the hole left by bad_triangles
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            # Create a new triangle using point p and edge extremes
            T = (idx, e0, e1)

            # Store circumcenter and circumradius of the triangle
            self.circles[T] = self.circumcenter(T)

            # Set opposite triangle of the edge as neighbour of T
            self.triangles[T] = [tri_op, None, None]

            # Try to set T as neighbour of the opposite triangle
            if tri_op:
                # search the neighbour of tri_op that use edge (e1, e0)
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            # change link to use our new triangle
                            self.triangles[tri_op][i] = T

            # Add triangle to a temporal list
            new_triangles.append(T)

        # Link the new triangles each another
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]   # next
            self.triangles[T][2] = new_triangles[(i-1) % N]   # previous

    def exportTriangles(self):
        """Export the current list of Delaunay triangles
        """
        # Filter out triangles with any vertex in the extended BBox
        return [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportCircles(self):
        """Export the circumcircles as a list of (center, radius)
        """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)

        # Filter out triangles with any vertex in the extended BBox
        # Do sqrt of radius before of return
        return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportDT(self):
        """Export the current set of Delaunay coordinates and triangles.
        """
        # Filter out coordinates in the extended BBox
        coord = self.coords[4:]

        # Filter out triangles with any vertex in the extended BBox
        tris = [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]
        return coord, tris

    def exportExtendedDT(self):
        """Export the Extended Delaunay Triangulation (with the frame vertex).
        """
        return self.coords, list(self.triangles)

    def exportVoronoiRegions(self):
        """Export coordinates and regions of Voronoi diagram as indexed data.
        """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)
        useVertex = {i: [] for i in range(len(self.coords))}
        vor_coors = []
        index = {}
        # Build a list of coordinates and a index per triangle/region
        for tidx, (a, b, c) in enumerate(self.triangles):
            vor_coors.append(self.circles[(a, b, c)][0])
            # Insert triangle, rotating it so the key is the "last" vertex
            useVertex[a] += [(b, c, a)]
            useVertex[b] += [(c, a, b)]
            useVertex[c] += [(a, b, c)]
            # Set tidx as the index to use with this triangles
            index[(a, b, c)] = tidx
            index[(c, a, b)] = tidx
            index[(b, c, a)] = tidx

        # init regions per coordinate dictionary
        regions = {}
        # Sort each region in a coherent order, and substitude each triangle
        # by its index
        for i in range(4, len(self.coords)):
            v = useVertex[i][0][0]  # Get a vertex of a triangle
            r = []
            for _ in range(len(useVertex[i])):
                # Search the triangle beginning with vertex v
                t = [t for t in useVertex[i] if t[0] == v][0]
                r.append(index[t])  # Add the index of this triangle to region
                v = t[1]            # Choose the next vertex to search
            regions[i-4] = r        # Store region.

        return vor_coors, regions
