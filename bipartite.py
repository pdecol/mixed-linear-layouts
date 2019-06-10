import sys
from collections import defaultdict

from models import Graph
from utils import Queue


class LayeredGraph(Graph):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.edge_colors = {}
        self.vertex_added_by = {}  # maps a vertex we to its parent that introduced him
        self.vertices_added = defaultdict(list)  # all vertices that got introduced by a vertex

    def get_order(self):
        return [x for layer in self.layers for x in layer]

    def get_stack(self):
        return [e for e, color in self.edge_colors.items() if color == 'red']

    def get_queue(self):
        return [e for e, color in self.edge_colors.items() if color == 'green']

    @classmethod
    def create_layers_bfs(cls, graph, start_vertex_id=1):
        layered_graph = LayeredGraph()
        layered_graph.vertices = graph.vertices
        layered_graph.edges = graph.edges
        layered_graph._vertex_mapping = graph._vertex_mapping
        layered_graph.edge_order = graph.edge_order

        order = []
        queue = Queue()

        start_vertex = graph.vertices[start_vertex_id - 1]
        layered_graph.vertex_added_by[start_vertex] = None
        last_layer = []
        current_layer = [start_vertex]

        order.append(start_vertex)
        queue.enqueue(start_vertex)
        while queue.size() > 0:
            v = queue.dequeue()
            if v == start_vertex or layered_graph.vertex_added_by[v] not in last_layer:
                layered_graph.layers.append(current_layer)
                last_layer = current_layer
                current_layer = []
            if v != start_vertex:
                current_layer.append(v)

            neighbours = graph.edge_order[v]
            if v != start_vertex:
                while neighbours[0] != layered_graph.vertex_added_by[v]:
                    neighbours.append(neighbours.pop(0))
            for neighbour in neighbours:
                if neighbour not in order:
                    order.append(neighbour)
                    queue.enqueue(neighbour)
                    layered_graph.vertex_added_by[neighbour] = v
                    layered_graph.vertices_added[v].append(neighbours)
        layered_graph.layers.append(current_layer)

        return layered_graph

    def colour_edges(self):
        """
        Colours all edges that can be put on the queue in green and the
        Remaining edges in red
        """
        for edge in self.edges:
            self.edge_colors[edge] = 'green'
        pre_layer = []
        for layer in self.layers:
            if not pre_layer:
                pre_layer = layer
                continue

            vertex_to_match = layer[0]
            for vertex in pre_layer:
                neighbours = self.edge_order[vertex]

                # shift neighbours list until the first edge to the next layer is in front
                if self.vertex_added_by[vertex]:  # vertex in first layer has no parent
                    while neighbours[-1] != self.vertex_added_by[vertex]:
                        neighbours.append(neighbours.pop(0))

                for neighbour in neighbours:
                    if neighbour not in layer:
                        continue
                    if layer.index(neighbour) < layer.index(vertex_to_match):
                        edge = (vertex, neighbour)
                        if edge not in self.edges:
                            edge = (neighbour, vertex)
                        self.edge_colors[edge] = 'red'
                    else:
                        vertex_to_match = neighbour

            pre_layer = layer

        # Colour edges on the same layer red
        for layer in self.layers:
            for v1 in layer:
                for edge in self.get_edges_of_vertex(v1):
                    if self.get_other_vertex_of_edge(v1, edge) in layer:
                        self.edge_colors[edge] = 'red'

    def move_ending_vertex_up(self):
        """
        If a vertex v is on layer L_i and has only green edges to Layer L_i-1
        and all of the neighbour vertices are neighbours on L_i-1 then we can
        move v up to Li-1 and put the edges onto the stack.
        """
        moves = []
        previous_layer = None
        for layer in self.layers:
            if not previous_layer:
                previous_layer = layer
                continue

            vertices_already_moved = 0
            for vertex in layer:
                is_eligible = True
                neighbours = self.edge_order[vertex]
                # check if all neighbours are in the upper layer:
                for neighbour in neighbours:
                    if neighbour not in previous_layer:
                        is_eligible = False
                # check if all edges are green:
                for edge in self.get_edges_of_vertex(vertex):
                    if not self.edge_colors[edge] == 'green':
                        is_eligible = False

                if not is_eligible:
                    continue

                # check if all neighbours of vertex are next to each other in the upper layer
                highest_index = None
                lowest_index = None
                for neighbour in neighbours:
                    index = previous_layer.index(neighbour)
                    if lowest_index is None or index < lowest_index:
                        lowest_index = index
                    if highest_index is None or index > highest_index:
                        highest_index = index
                if highest_index - lowest_index == len(neighbours) - 1:
                    # (vertex to move, current layer, index at new layer, new layer)
                    moves.append([vertex, layer, lowest_index + 1 + vertices_already_moved, previous_layer])
                    vertices_already_moved += 1

            previous_layer = layer
            for move in moves:
                move[1].remove(move[0])
                move[3].insert(move[2], move[0])
                # remove layer if empty
                if not move[1]:
                    self.layers.remove(move[1])
            moves = []

