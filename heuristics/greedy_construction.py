import random
import sys
from collections import defaultdict

from models import MixedLinearLayout, StackLinearLayout, LinearLayout, Queue

from utils import Tree


#
# Vertex order heuristics
#

def random_dfs(graph):
    def dfs(current_vertex):
        order.append(current_vertex)
        neighbours = graph.adjacency[current_vertex].copy()
        random.shuffle(neighbours)
        for neighbour in neighbours:
            if neighbour not in order:
                dfs(neighbour)

    order = []
    while len(order) < len(graph.vertices):  # for unconnected graphs
        start_vertex = random.choice(graph.vertices)
        if start_vertex not in order:
            dfs(start_vertex)
    return order


def smallest_degree_dfs(graph):
    def dfs(current_vertex):
        order.append(current_vertex)
        neighbours = graph.adjacency[current_vertex]
        neighbours = sorted(neighbours, key=lambda n: len(graph.adjacency[n]))
        for neighbour in neighbours:
            if neighbour not in order:
                dfs(neighbour)

    order = []
    while len(order) < len(graph.vertices):  # for unconnected graphs
        remaining_vertices = set(graph.vertices).difference(order)
        start_vertex = min(remaining_vertices, key=lambda v: len(graph.adjacency[v]))
        if start_vertex not in order:
            dfs(start_vertex)
    return order


def highest_degree_dfs(graph):
    def dfs(current_vertex):
        order.append(current_vertex)
        neighbours = graph.adjacency[current_vertex]
        neighbours = sorted(neighbours, key=lambda n: len(graph.adjacency[n]), reverse=True)
        for neighbour in neighbours:
            if neighbour not in order:
                dfs(neighbour)

    order = []
    while len(order) < len(graph.vertices):  # for unconnected graphs
        remaining_vertices = set(graph.vertices).difference(order)
        start_vertex = max(remaining_vertices, key=lambda v: len(graph.adjacency[v]))
        if start_vertex not in order:
            dfs(start_vertex)
    return order


def random_bfs(graph):
    order = []
    queue = Queue()

    while len(order) < len(graph.vertices):
        start_vertex = random.choice(graph.vertices)
        if start_vertex not in order:
            order.append(start_vertex)
            queue.enqueue(start_vertex)
            while queue.size() > 0:
                v = queue.dequeue()
                neighbours = graph.adjacency[v]
                random.shuffle(neighbours)
                for neighbour in neighbours:
                    if neighbour not in order:
                        order.append(neighbour)
                        queue.enqueue(neighbour)

    return order


def tree_bfs(graph):
    visited = []
    queue = Queue()
    trees = []
    tree_dict = {}  # from vertex to its tree node
    i = 0
    while len(visited) < len(graph.vertices):
        start_vertex = random.choice(graph.vertices)
        if start_vertex not in visited:
            visited.append(start_vertex)
            queue.enqueue(start_vertex)
            while queue.size() > 0:
                next_vertex = queue.dequeue()
                if next_vertex in tree_dict:
                    tree = tree_dict[next_vertex]
                else:
                    tree = Tree(next_vertex)
                    tree_dict[next_vertex] = tree
                    trees.append(tree)

                neighbours = graph.adjacency[next_vertex]
                random.shuffle(neighbours)
                for neighbour in neighbours:
                    if neighbour not in visited:
                        child = Tree(neighbour)
                        tree_dict[neighbour] = child
                        tree.add_child(child)

                        visited.append(neighbour)
                        queue.enqueue(neighbour)

    order = []
    for tree in trees:
        order.extend(tree.get_values_dfs())
    return order


def con_greedy(graph):
    order = []
    remaining_vertices = graph.vertices.copy()
    while len(order) < len(graph.vertices):
        # select best next vertex
        best_v = None
        best_v_score = (0, 0)
        for v in remaining_vertices:
            placed_neighbours = 0
            unplaced_neighbours = 0
            for adjacent in graph.adjacency[v]:
                if adjacent in order:
                    placed_neighbours += 1
                else:
                    unplaced_neighbours += 1
            if not best_v or placed_neighbours > best_v_score[0] or \
                    (placed_neighbours == best_v_score[0] and
                     unplaced_neighbours < best_v_score[1]):
                best_v = v
                best_v_score = (placed_neighbours, unplaced_neighbours)

        # mark bad positions for the new vertex
        stack_marks = defaultdict(int)
        queue_marks = defaultdict(int)
        for open_edge in graph.get_edges_of_vertex(best_v):
            u = graph.get_other_vertex_of_edge(best_v, open_edge)
            if u in remaining_vertices:
                continue

            for x, y in graph.edges:
                if u in (x, y) or x in remaining_vertices or y in remaining_vertices:
                    continue

                u_index = order.index(u)
                x_index = order.index(x)
                y_index = order.index(y)
                if x_index > y_index:
                    x_index, y_index = y_index, x_index

                if x_index < u_index < y_index:
                    for i in range(0, x_index + 1):
                        stack_marks[i] += 1
                    for i in range(y_index, len(order) + 1):
                        stack_marks[i] += 1
                else:
                    for i in range(x_index, y_index + 1):
                        stack_marks[i] += 1

                if u_index < x_index:
                    for i in range(y_index, len(order) + 1):
                        queue_marks[i] += 1
                elif x_index < u_index < y_index:
                    for i in range(x_index, y_index + 1):
                        queue_marks[i] += 1
                else:
                    for i in range(0, x_index + 1):
                        queue_marks[i] += 1

        # find and place the vertex to the best position
        best_index = None
        best_index_marks = None
        for index in range(0, len(order) + 1):
            marks = 0
            marks += stack_marks[index]
            marks += queue_marks[index]
            if best_index is None or marks < best_index_marks:
                best_index = index
                best_index_marks = marks
        if best_index is None:
            best_index = 0

        order.insert(best_index, best_v)
        remaining_vertices.remove(best_v)

    return order


def con_greedy_plus_vertex_order(graph, stacks, queues):
    layout = con_greedy_plus(graph, stacks, queues)
    return layout.order


def unchanged_vertex_order(graph):
    return graph.vertices


#
# Edge order heuristics
#

def edge_length(graph, vertex_order, stacks=1, queues=1):
    position = {v: i for i, v in enumerate(vertex_order)}
    edges = sorted(graph.edges,
                   key=lambda e: abs(position[e[0]] - position[e[1]]),
                   reverse=True)
    return _best_fit_stack_first(graph, vertex_order, edges, stacks, queues)


def ceil_floor(graph, vertex_order, stacks=1, queues=1):
    position = {v: i for i, v in enumerate(vertex_order)}
    length = {}
    for edge in graph.edges:
        v1 = position[edge[0]]
        v2 = position[edge[1]]
        if v1 > v2:
            v1, v2 = v2, v1
        length[edge] = min(v1 - v2, v1 + (v2 - len(graph.edges)))
    edges = sorted(graph.edges, key=lambda e: length[edge], reverse=True)
    return _best_fit_stack_first(graph, vertex_order, edges, stacks, queues)


def connectivity(graph, vertex_order, stacks=1, queues=1):
    """
    Work the way from one end of the linear layout to the other
    """
    position = {v: i for i, v in enumerate(vertex_order)}
    edge_end = {}
    edge_start = {}
    for edge in graph.edges:
        v1 = position[edge[0]]
        v2 = position[edge[1]]
        if v1 > v2:
            v1, v2 = v2, v1
        edge_start[edge] = v1
        edge_end[edge] = v2
    edges = sorted(graph.edges, key=lambda x: (edge_end[x], edge_start[x]))
    return _best_fit_stack_first(graph, vertex_order, edges, stacks, queues)


def data_structure(graph, vertex_order, stacks=1, queues=1, alpha=1):
    alpha_max = (1 / (stacks + queues))  # 2 pages = 50% chance to land on this page
    alpha_min = 0.01  # at least something very small
    alpha = alpha_max * (1.15 - (0.1 * (stacks + queues)))  # the more pages we have, the lower alpha performs better
    alpha = max(min(alpha_max, alpha), alpha_min)  # must be between min and max

    layout = LinearLayout(stacks, queues, graph=graph, order=vertex_order)
    stacks = {i: [] for i in range(0, len(layout.stacks))}
    queues = {i: [] for i in range(0, len(layout.queues))}
    conflicts_on_stack = defaultdict(int)
    conflicts_on_queue = defaultdict(int)

    vertex_indices = layout.get_vertex_indices()
    for vertex in vertex_order:
        vertex_index = vertex_indices[vertex]
        edges = graph.get_edges_of_vertex(vertex)
        start_edges = [e for e in edges if vertex_index < vertex_indices[e[0]] or vertex_index < vertex_indices[e[1]]]
        end_edges = [e for e in edges if vertex_index > vertex_indices[e[0]] or vertex_index > vertex_indices[e[1]]]

        for edge in end_edges:
            best_page = (False, None)  # (is it a stack?, data structure id)
            best_page_value = sys.maxsize

            for id, stack in stacks.items():
                stack_conflicts = 0
                for e in stack:
                    if e == edge:
                        break
                    if e in end_edges:
                        continue
                    stack_conflicts += 1
                value = conflicts_on_stack[(id, edge[0], edge[1])] + (stack_conflicts * alpha)
                if best_page_value > value:
                    best_page = (True, id)
                    best_page_value = value

            for id, queue in queues.items():
                queue_conflicts = 0
                for e in reversed(queue):
                    if e == edge:
                        break
                    if e in end_edges:
                        continue
                    queue_conflicts += 1
                value = conflicts_on_queue[(id, edge[0], edge[1])] + (queue_conflicts * alpha)
                if best_page_value > value:
                    best_page = (False, id)
                    best_page_value = value

            if best_page[0]:  # is it a stack?
                layout.stacks[best_page[1]].append(edge)
                for e in stacks[best_page[1]]:
                    if e == edge:
                        break
                    if e in end_edges:
                        continue
                    conflicts_on_stack[(best_page[1], e[0], e[1])] += 1

            else:
                layout.queues[best_page[1]].append(edge)
                for e in reversed(queues[best_page[1]]):
                    if e == edge:
                        break
                    if e in end_edges:
                        continue
                    conflicts_on_queue[(best_page[1], e[0], e[1])] += 1

            for stack in stacks.values():
                stack.remove(edge)
            for queue in queues.values():
                queue.remove(edge)

        # Add starting edges to data structures
        start_edges = sorted(start_edges, key=lambda e: vertex_indices[e[0]] if vertex_index != vertex_indices[e[0]] else vertex_indices[e[1]])
        for edge in start_edges:
            for queue in queues.values():
                queue.insert(0, edge)
        for edge in reversed(start_edges):
            for stack in stacks.values():
                stack.insert(0, edge)

    return layout


#
# Edge distribution heuristics
#

def _best_fit_stack_first(graph, vertex_order, edge_order, stacks, queues):
    layout = LinearLayout(stacks, queues, graph, vertex_order)
    for edge in edge_order:
        best_page = layout.best_page_for_new_edge(edge)
        best_page.append(edge)
    return layout


def _best_fit_queue_first(graph, vertex_order, edge_order, stacks, queues):
    raise NotImplementedError()
    # mll = MixedLinearLayout(graph)
    # mll.order = vertex_order
    # for edge in edge_order:
    #     crossings = mll.crossings_of_edge_on_page(edge, mll.stack)
    #     nestings = mll.nestings_of_edge_on_page(edge, mll.queue)
    #     if crossings < nestings:
    #         mll.stack.append(edge)
    #     else:
    #         mll.queue.append(edge)
    # return mll


def _look_ahead(graph, vertex_order, edge_order, stacks, queues):
    # Needs to be adjusted to variable amount of pages
    raise NotImplementedError()

    mll = MixedLinearLayout(graph)
    mll.order = vertex_order
    remaining_edges = list(edge_order)
    for edge in edge_order:
        crossings = mll.crossings_of_edge_on_page(edge, mll.stack)
        nestings = mll.nestings_of_edge_on_page(edge, mll.queue)
        if crossings < nestings:
            mll.stack.append(edge)
        elif nestings < crossings:
            mll.queue.append(edge)
        else:
            potential_crossings = mll.crossings_of_edge_on_page(edge, remaining_edges)
            potential_nestings = mll.nestings_of_edge_on_page(edge, remaining_edges)
            if potential_crossings < potential_nestings:
                mll.stack.append(edge)
            else:
                mll.queue.append(edge)
        remaining_edges.remove(edge)
    return mll


#
# Full drawing heuristics
#

def full_combined_heuristic(vertex_order_heuristic, page_assignment_heuristic):
    def f(graph, stacks=1, queues=1):
        if vertex_order_heuristic.__name__ == 'con_greedy_plus_vertex_order':
            order = vertex_order_heuristic(graph, stacks, queues)
        else:
            order = vertex_order_heuristic(graph)
        layout = page_assignment_heuristic(graph, order, stacks, queues)
        return layout
    return f


def con_greedy_plus(graph, stacks, queues):
    layout = LinearLayout(stacks, queues, graph)
    order = layout.order
    remaining_vertices = graph.vertices.copy()
    while len(order) < len(graph.vertices):
        # select best next vertex
        best_v = None
        best_v_score = (0, 0)
        for v in remaining_vertices:
            placed_neighbours = 0
            unplaced_neighbours = 0
            for adjacent in graph.adjacency[v]:
                if adjacent in order:
                    placed_neighbours += 1
                else:
                    unplaced_neighbours += 1
            if not best_v or placed_neighbours > best_v_score[0] or \
                    (placed_neighbours == best_v_score[0] and
                     unplaced_neighbours < best_v_score[1]):
                best_v = v
                best_v_score = (placed_neighbours, unplaced_neighbours)

        # mark bad positions for the new vertex
        stack_marks = defaultdict(int)
        queue_marks = defaultdict(int)
        for open_edge in graph.get_edges_of_vertex(best_v):
            open_v = graph.get_other_vertex_of_edge(best_v, open_edge)
            if open_v in remaining_vertices:
                continue

            for id, stack in enumerate(layout.stacks):
                for closed_edge in stack:
                    if open_v in closed_edge:
                        continue

                    open_v_index = order.index(open_v)
                    closed_v1_index = order.index(closed_edge[0])
                    closed_v2_index = order.index(closed_edge[1])
                    if closed_v1_index > closed_v2_index:
                        closed_v1_index, closed_v2_index = \
                            closed_v2_index, closed_v1_index

                    if closed_v1_index < open_v_index < closed_v2_index:
                        for i in range(0, closed_v1_index + 1):
                            stack_marks[(id, i)] += 1
                        for i in range(closed_v2_index, len(order) + 1):
                            stack_marks[(id, i)] += 1
                    else:
                        for i in range(closed_v1_index, closed_v2_index + 1):
                            stack_marks[(id, i)] += 1

            for id, queue in enumerate(layout.queues):
                for closed_edge in queue:
                    if open_v in closed_edge:
                        continue

                    open_v_index = order.index(open_v)
                    closed_v1_index = order.index(closed_edge[0])
                    closed_v2_index = order.index(closed_edge[1])
                    if closed_v1_index > closed_v2_index:
                        closed_v1_index, closed_v2_index = \
                            closed_v2_index, closed_v1_index

                    if open_v_index < closed_v1_index:
                        for i in range(closed_v2_index, len(order) + 1):
                            queue_marks[(id, i)] += 1
                    elif closed_v1_index < open_v_index < closed_v2_index:
                        for i in range(closed_v1_index, closed_v2_index + 1):
                            queue_marks[(id, i)] += 1
                    else:
                        for i in range(0, closed_v1_index + 1):
                            queue_marks[(id, i)] += 1

        # find and place the vertex to the best position
        best_index = None
        best_index_marks = None
        for index in range(0, len(order) + 1):
            marks = 0
            for id in range(0, len(layout.stacks)):
                marks += stack_marks[(id, index)]
            for id in range(0, len(layout.queues)):
                marks += queue_marks[(id, index)]
            if best_index is None or marks < best_index_marks:
                best_index = index
                best_index_marks = marks
        if best_index is None:
            best_index = 0

        order.insert(best_index, best_v)
        remaining_vertices.remove(best_v)

        # distribute open edges
        for open_edge in graph.get_edges_of_vertex(best_v):
            open_v = graph.get_other_vertex_of_edge(best_v, open_edge)
            if open_v in remaining_vertices:
                continue

            best_page = layout.best_page_for_new_edge(open_edge)
            best_page.append(open_edge)

    return layout
