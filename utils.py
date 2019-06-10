import numpy as np


def check_layout(func):
    def function_wrapper(*args, **kwargs):
        layout = func(*args, **kwargs)
        if layout is not None and not layout.is_valid():
            raise Exception('Layout ist not valid!')
        return layout
    return function_wrapper


class Tree:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
        obj.parent = self

    def get_values_dfs(self):
        values = [self.value]
        for child in self.children:
            values.extend(child.get_values_dfs())
        return values


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


def is_graph_connected(graph):
    """
    Check if the graph is connected with a breadth-first search
    """
    visited = []
    queue = Queue()

    start_vertex = graph.vertices[0]
    visited.append(start_vertex)
    queue.enqueue(start_vertex)
    while queue.size() > 0:
        v = queue.dequeue()
        neighbours = graph.adjacency[v]
        for neighbour in neighbours:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.enqueue(neighbour)

    return len(visited) == len(graph.vertices)


def triangles_of_graph(graph):
    a = np.array(graph.get_adjacency_matrix())
    a_3 = np.matmul(a, np.matmul(a, a))
    return np.trace(a_3) / 6
