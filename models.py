import itertools
import sys
from collections import defaultdict
from random import shuffle

from utils import check_layout


class Graph:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.adjacency = defaultdict(list)
        self._vertex_mapping = {}  # External name to internal representation
        self.edge_order = {}  # circular edge order around a vertex

    def __repr__(self):
        return 'Graph: vertices: %s, '  \
               'edges: %s' % (len(self.vertices), len(self.edges))

    def add_edge(self, vertex_1, vertex_2):
        if vertex_1 not in self._vertex_mapping:
            v1 = len(self.vertices) + 1
            self._vertex_mapping[vertex_1] = v1
            self.vertices.append(v1)
        else:
            v1 = self._vertex_mapping[vertex_1]
        if vertex_2 not in self._vertex_mapping:
            v2 = len(self.vertices) + 1
            self._vertex_mapping[vertex_2] = v2
            self.vertices.append(v2)
        else:
            v2 = self._vertex_mapping[vertex_2]

        if v1 > v2:
            # Edges are always ordered like (lower_vertex_id, higher_vertex_id)
            v1, v2 = v2, v1
        if (v1, v2) not in self.edges:
            self.edges.append((v1, v2))
            self.adjacency[v1].append(v2)
            self.adjacency[v2].append(v1)

    def add_vertex(self, vertex):
        if vertex not in self._vertex_mapping:
            v1 = len(self.vertices) + 1
            self._vertex_mapping[vertex] = v1
            self.vertices.append(v1)

    def add_circular_edge_order(self, vertex, edges):
        vertex = self._vertex_mapping[vertex]
        edges = [self._vertex_mapping[e] for e in edges]
        self.edge_order[vertex] = edges

    def has_edge(self, vertex_1, vertex_2):
        v1 = self._vertex_mapping[vertex_1]
        v2 = self._vertex_mapping[vertex_2]
        if v1 > v2:
            v1, v2 = v2, v1
        return True if (v1, v2) in self.edges else False

    def get_other_vertex_of_edge(self, vertex, edge):
        return edge[0] if vertex == edge[1] else edge[1]

    def get_edges_of_vertex(self, vertex):
        return [edge for edge in self.edges if vertex in edge]

    def get_vertices(self):
        return self.get_vertices_labeled(self.vertices)

    def get_edges(self):
        return self.get_edges_labeled(self.edges)

    def get_vertex_label(self, vertex):
        inverse_map = {v: k for k, v in self._vertex_mapping.items()}
        return inverse_map[vertex]

    def get_vertices_labeled(self, vertices):
        inverse_map = {v: k for k, v in self._vertex_mapping.items()}
        return [inverse_map[x] for x in vertices]

    def _get_internal_vertices(self, vertices):
        return [self._vertex_mapping[str(x)] for x in vertices]

    def get_edges_labeled(self, edges):
        inverse_map = {v: k for k, v in self._vertex_mapping.items()}
        return [(inverse_map[x], inverse_map[y]) for x, y in edges]


class LinearLayout:
    def __init__(self, stacks, queues, graph=None, order=None):
        self._num_stacks = stacks
        self._num_queues = queues
        self.stacks = []
        self.queues = []
        for i in range(stacks):
            self.stacks.append([])
        for i in range(queues):
            self.queues.append([])
        self.graph = graph
        self.order = order if order else []

    @property
    def number_of_stacks(self):
        return len(self.stacks)

    @property
    def number_of_queues(self):
        return len(self.queues)

    def get_page(self, number):
        return (self.stacks + self.queues)[number - 1]

    def is_valid(self):
        if not self.is_order_and_edges_valid():
            return False
        for stack in self.stacks:
            if not self.is_stack_valid(stack):
                return False
        for queue in self.queues:
            if not self.is_queue_valid(queue):
                return False
        return True

    def is_order_and_edges_valid(self):
        if set(self.order) != set(self.graph.vertices) \
                or len(self.order) != len(self.graph.vertices)\
                or len(self.stacks) != self._num_stacks\
                or len(self.queues) != self._num_queues:
            return False
        edges = set()
        number_of_edges = 0
        for page in self.stacks + self.queues:
            edges.update(page)
            number_of_edges += len(page)
        if edges != set(self.graph.edges) \
                or number_of_edges != len(self.graph.edges):
            return False
        return True

    def is_stack_valid(self, stack_edges):
        vertex_indices = self.get_vertex_indices()
        stack = Stack()
        for vertex in self.order:
            vertex_index = vertex_indices[vertex]
            edges = [edge for edge in stack_edges if vertex in edge]
            start_edges = [e for e in edges if vertex_index < vertex_indices[e[0]] or vertex_index < vertex_indices[e[1]]]
            end_edges = [e for e in edges if vertex_index > vertex_indices[e[0]] or vertex_index > vertex_indices[e[1]]]

            for edge in sorted(end_edges, key=lambda e: vertex_indices[e[0]] if vertex_index != vertex_indices[e[0]] else vertex_indices[e[1]], reverse=True):
                if stack.pop() != edge:
                    return False

            for edge in sorted(start_edges, key=lambda e: vertex_indices[e[0]] if vertex_index != vertex_indices[e[0]] else vertex_indices[e[1]], reverse=True):
                stack.push(edge)
        return True

    def is_queue_valid(self, queue_edges):
        vertex_indices = self.get_vertex_indices()
        queue = Queue()
        for vertex in self.order:
            vertex_index = vertex_indices[vertex]
            edges = [edge for edge in queue_edges if vertex in edge]
            start_edges = [e for e in edges if vertex_index < vertex_indices[e[0]] or vertex_index < vertex_indices[e[1]]]
            end_edges = [e for e in edges if vertex_index > vertex_indices[e[0]] or vertex_index > vertex_indices[e[1]]]

            for edge in sorted(end_edges, key=lambda e: vertex_indices[e[0]] if vertex_index != vertex_indices[e[0]] else vertex_indices[e[1]]):
                if queue.dequeue() != edge:
                    return False

            for edge in sorted(start_edges, key=lambda e: vertex_indices[e[0]] if vertex_index != vertex_indices[e[0]] else vertex_indices[e[1]]):
                queue.enqueue(edge)
        return True

    def get_vertex_indices(self, order=None):
        order = order if order else self.order
        return {vertex: i for i, vertex in enumerate(order)}

    def get_edges_in_order(self, edges=None):
        """
        Sort vertices of edges and the edges by start vertex index
        as the first criteria and the end vertex index as the second criteria
        """
        if not edges:
            edges = self.graph.edges
        indices = self.get_vertex_indices()
        _edges = []
        for v1, v2 in edges:
            if indices[v1] > indices[v2]:
                v1, v2 = v2, v1
            _edges.append((v1, v2))
        return sorted(_edges, key=lambda x: (indices[x[0]], indices[x[1]]))

    def total_crossings(self):
        crossings = 0
        for page in self.stacks:
            for edge in page:
                crossings += self.crossings_of_edge_on_page(edge, page)
        return crossings // 2

    def total_nestings(self):
        nestings = 0
        for page in self.queues:
            for edge in page:
                nestings += self.nestings_of_edge_on_page(edge, page)
        return nestings // 2

    def total_conflicts(self):
        return self.total_crossings() + self.total_nestings()

    def crossings_of_edge_on_page(self, edge, page):
        crossings = 0
        indices = self.get_vertex_indices()
        a1 = indices[edge[0]]
        a2 = indices[edge[1]]
        if a1 > a2:
            a1, a2 = a2, a1
        for edge_2 in page:
            b1 = indices[edge_2[0]]
            b2 = indices[edge_2[1]]
            if b1 > b2:
                b1, b2 = b2, b1
            if a1 < b1 < a2 < b2 or b1 < a1 < b2 < a2:
                crossings += 1
        return crossings

    def nestings_of_edge_on_page(self, edge, page):
        nestings = 0
        indices = self.get_vertex_indices()
        a1 = indices[edge[0]]
        a2 = indices[edge[1]]
        if a1 > a2:
            a1, a2 = a2, a1
        for edge_2 in page:
            b1 = indices[edge_2[0]]
            b2 = indices[edge_2[1]]
            if b1 > b2:
                b1, b2 = b2, b1
            if a1 < b1 < b2 < a2 or b1 < a1 < a2 < b2:
                nestings += 1
        return nestings

    def conflicts_of_vertex(self, vertex):
        conflicts = 0
        for stack in self.stacks:
            for edge in stack:
                if vertex in edge:
                    conflicts += self.crossings_of_edge_on_page(edge, stack)
        for queue in self.queues:
            for edge in queue:
                if vertex in edge:
                    conflicts += self.nestings_of_edge_on_page(edge, queue)
        return conflicts

    def best_page_for_new_edge(self, edge):
        """
        In case that a queue and stack page would have the same number of
        conflicts, the stack page will be preferred.
        """
        best_page = None
        best_page_conflicts = sys.maxsize
        for stack in self.stacks:
            crossings = self.crossings_of_edge_on_page(edge, stack)
            if best_page_conflicts > crossings:
                best_page = stack
                best_page_conflicts = crossings
        for queue in self.queues:
            nestings = self.nestings_of_edge_on_page(edge, queue)
            if best_page_conflicts > nestings:
                best_page = queue
                best_page_conflicts = nestings
        return best_page


class MixedLinearLayout(LinearLayout):
    def __init__(self, graph=None, order=None):
        super().__init__(stacks=1, queues=1, graph=graph, order=order)

    def __repr__(self):
        return 'MixedLinearLayout: order %s, stack size: %s, queue size: %s' \
               % (self.graph.get_vertices_labeled(self.order),
                  len(self.stack),
                  len(self.queue))

    @property
    def stack(self):
        return self.stacks[0]

    @stack.setter
    def stack(self, stack):
        self.stacks[0] = stack

    @property
    def queue(self):
        return self.queues[0]

    @queue.setter
    def queue(self, queue):
        self.queues[0] = queue

    @classmethod
    @check_layout
    def create(cls, graph):
        for order in itertools.permutations(graph.vertices):
            layout = cls.create_fixed_order(graph, list(order))
            if layout:
                return layout
        return MixedLinearLayout(graph)

    @classmethod
    @check_layout
    def create_random(cls, graph):
        i = 0
        while True:
            i += 1
            order = graph.vertices
            shuffle(order)
            layout = cls.create_fixed_order(graph, order)
            if layout:
                return layout
            if i % 100 == 0:
                print('%s orders tried' % i)

    @classmethod
    @check_layout
    def create_fixed_order(cls, graph, order):
        """
        Create mixed layout in linear time if it exists
        """
        # https://kartikkukreja.wordpress.com/2013/05/16/solving-2-sat-in-linear-time/
        # https://github.com/kartikkukreja/blog-codes/blob/master/src/2SAT%20in%20linear%20time.cpp
        mll = MixedLinearLayout(graph, order)
        indices = mll.get_vertex_indices()
        edges = mll.get_edges_in_order()
        edge_ids = {e: i for i, e in enumerate(edges, start=1)}

        num_edges = len(edges)
        g = defaultdict(list)
        g_rev = defaultdict(list)
        for i, e1 in enumerate(edges):
            a1 = indices[e1[0]]
            a2 = indices[e1[1]]
            for e2 in edges[i + 1:]:
                b1 = indices[e2[0]]
                b2 = indices[e2[1]]
                if a1 < b1 < a2 < b2:
                    g[edge_ids[e1]].append(num_edges + edge_ids[e2])
                    g[edge_ids[e2]].append(num_edges + edge_ids[e1])
                    g_rev[num_edges + edge_ids[e2]].append(edge_ids[e1])
                    g_rev[num_edges + edge_ids[e1]].append(edge_ids[e2])
                elif a1 < b1 < b2 < a2:
                    g[num_edges + edge_ids[e1]].append(edge_ids[e2])
                    g[num_edges + edge_ids[e2]].append(edge_ids[e1])
                    g_rev[edge_ids[e2]].append(num_edges + edge_ids[e1])
                    g_rev[edge_ids[e1]].append(num_edges + edge_ids[e2])
                elif a2 <= b1:
                    break

        def dfs_reverse(i):
            nonlocal t
            explored[i] = True
            for it in g_rev[i]:
                if not explored[it]:
                    dfs_reverse(it)
            t += 1
            finish[i] = t

        def dfs(i):
            explored[i] = True
            leader[i] = parent
            for it in g[i]:
                if not explored[it]:
                    dfs(it)

        explored = defaultdict(lambda: False)
        order = defaultdict(int)
        leader = defaultdict(int)
        finish = defaultdict(int)
        t = 0
        parent = 0
        i = 2 * num_edges
        while i > 0:
            if not explored[i]:
                dfs_reverse(i)
            order[finish[i]] = i
            i -= 1

        explored = defaultdict(lambda: False)
        i = 2 * num_edges
        while i > 0:
            if not explored[order[i]]:
                parent = order[i]
                dfs(order[i])
            i -= 1

        def strongly_connected(u, v):
            return leader[u] == leader[v]

        truth_assignment = {}
        i = 2 * num_edges
        while i > 0:
            u = order[i]
            if u > num_edges:
                if strongly_connected(u, u - num_edges):
                    break
                if leader[u] not in truth_assignment:
                    truth_assignment[leader[u]] = True
                    truth_assignment[leader[u - num_edges]] = False

            else:
                if strongly_connected(u, u + num_edges):
                    break
                if leader[u] not in truth_assignment:
                    truth_assignment[leader[u]] = True
                    truth_assignment[leader[u + num_edges]] = False
            i -= 1

        if i > 0:
            return None
        for e in edges:
            edge = e
            if e not in graph.edges:
                # Vertices can be switches due to get_edges_in_order(). Restore
                # original order
                edge = (e[1], e[0])
            if truth_assignment[leader[edge_ids[e]]]:
                mll.stack.append(edge)
            else:
                mll.queue.append(edge)
        return mll


class StackLinearLayout(LinearLayout):
    def __init__(self, graph=None, order=None):
        super().__init__(stacks=2, queues=0, graph=graph, order=order)

    @property
    def stack1(self):
        return self.get_page(1)

    @stack1.setter
    def stack1(self, stack):
        self.stacks[0] = stack

    @property
    def stack2(self):
        return self.get_page(2)

    @stack2.setter
    def stack2(self, stack):
        self.stacks[1] = stack


class AbstractDataStructure:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def size(self):
        return len(self.items)


class Stack(AbstractDataStructure):
    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]


class Queue(AbstractDataStructure):
    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        return self.items.pop(0)

    def peek(self):
        return self.items[0]
