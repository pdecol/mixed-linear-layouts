import random


def _greedy_edge_assignment_optimization(mll):
    edges = list(mll.stack + mll.queue)
    random.shuffle(edges)
    for edge in edges:
        crossings = mll.crossings_of_edge_on_page(edge, mll.stack)
        nestings = mll.nestings_of_edge_on_page(edge, mll.queue)

        if edge in mll.stack and nestings < crossings:
            mll.stack.remove(edge)
            mll.queue.append(edge)
            return True

        if edge in mll.queue and crossings < nestings:
            mll.queue.remove(edge)
            mll.stack.append(edge)
            return True
    return False


def _greedy_vertex_order_optimization(mll):
    graph = mll.graph
    original_order = list(mll.order)
    vertices = list(graph.vertices)
    random.shuffle(vertices)
    for vertex in vertices:
        original_position = mll.order.index(vertex)
        conflicts = mll.conflicts_of_vertex(vertex)

        for i in range(len(mll.order) + 1):
            if i == original_position:
                continue
            mll.order.remove(vertex)
            mll.order.insert(i, vertex)

            new_conflicts = mll.conflicts_of_vertex(vertex)
            if new_conflicts < conflicts:
                return True
            else:
                mll.order = list(original_order)

    return False


def _greedy_alt(mll, vertex_exhaustive=False, edge_exhaustive=False):
    total_conflicts = mll.total_conflicts()
    while total_conflicts:
        order_optimized = False
        if vertex_exhaustive:
            while _greedy_vertex_order_optimization(mll):
                order_optimized = True
        else:
            order_optimized = _greedy_vertex_order_optimization(mll)

        pages_optimized = False
        if edge_exhaustive:
            while _greedy_edge_assignment_optimization(mll):
                pages_optimized = True
        else:
            pages_optimized = _greedy_edge_assignment_optimization(mll)

        total_conflicts = mll.total_conflicts()
        #print('total conflicts', mll.total_conflicts())
        if not order_optimized and not pages_optimized:
            break
    return mll


def greedy_alt_rr(mll):
    return _greedy_alt(mll, False, False)


def greedy_alt_re(mll):
    return _greedy_alt(mll, False, True)


def greedy_alt_er(mll):
    return _greedy_alt(mll, True, False)


def greedy_alt_ee(mll):
    return _greedy_alt(mll, True, True)
