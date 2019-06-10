import copy
import itertools

from graph_generator import generate_complete_graph
from models import LinearLayout, Graph
from view import show_linear_layouts


def observation_1(show_layouts=True):
    """
    Generate all possible 2-stack 1-queue layouts of a complete graphs with
    8 vertices. Except that edges (i, i + 1) are always assigned to a stack
    pages which skips minor variations and reduces the total number to 32
    layouts.
    """
    num_vertices = 8
    stacks = 2
    queues = 1

    graph = generate_complete_graph(num_vertices)
    mll = LinearLayout(graph=graph,
                       order=list(range(1, num_vertices + 1)),
                       stacks=stacks, queues=queues)

    # Add edges that can always be on the stack without crossings
    remaining_edges = graph.edges
    if stacks:
        for i in range(1, num_vertices):
            mll.stacks[0].append((i, i + 1))
        mll.stacks[0].append((1, num_vertices))
        remaining_edges = list(set(graph.edges) - set(mll.stacks[0]))

    def search(mll, edges):
        edges = list(edges)
        if not edges:
            mlls.append(copy.deepcopy(mll))
            return None
        edge = edges.pop()

        for stack in mll.stacks:
            stack.append(edge)
            if mll.is_stack_valid(stack):
                if search(mll, edges):
                    return mll
            stack.remove(edge)

        for queue in mll.queues:
            queue.append(edge)
            if mll.is_queue_valid(queue):
                if search(mll, edges):
                    return mll
            queue.remove(edge)

    mlls = []
    search(mll, remaining_edges)

    for mll in mlls:
        if not mll.is_valid():
            raise Exception('MLL is not valid!')
        if show_layouts:
            show_linear_layouts([mll])

    return mlls


def observation_2():
    """
    Show that given two complete graphs with 8 vertices on a 2-stack 1-queue
    layout the vertices can only interleave once.
    """
    mlls_1 = observation_1(show_layouts=False)
    mlls_2 = []
    # create a second list of layouts that contains the same layouts and
    # additionally all of them with the stack pages swapped
    for mll in copy.deepcopy(mlls_1):
        mlls_2.append(mll)
        mll_new = copy.deepcopy(mll)
        mll_new.stacks[0], mll_new.stacks[1] = \
            mll_new.stacks[1], mll_new.stacks[0]
        mlls_2.append(mll_new)

    # Create a graph of two independent complete graphs with 8 vertices
    graph = Graph()
    num_vertices = 8
    for i in range(1, num_vertices + 1):
        for j in range(i + 1, num_vertices + 1):
            graph.add_edge(i, j)
    for i in range(num_vertices + 1, num_vertices * 2 + 1):
        for j in range(i + 1, num_vertices * 2 + 1):
            graph.add_edge(i, j)

    mlls = []
    for permutation in list(itertools.product([True, False], repeat=8)):
        if all(permutation):
            # Skip the permutation that would separate both K8
            continue
        for m1 in mlls_1:
            for m2 in mlls_2:
                order = []
                o1 = list(range(1, 9))
                o2 = list(range(9, 17))

                # All these permutations create all the possible orders in
                # which the two K8 can interleave.
                for p in permutation:
                    if p:
                        order.append(o1.pop(0))
                    else:
                        order.append(o2.pop(0))
                        order.append(o1.pop(0))
                while o2:
                    order.append(o2.pop(0))

                mll = LinearLayout(graph=graph, order=order,
                                   stacks=2, queues=1)

                # Create a new layout by combining the layouts of m1 and m2.
                # Note that the edges of m2 need to be adjusted here. m2 has
                # vertices form 1-8 but here they should become 9-16
                mll.stacks[0] = m1.stacks[0] + [(a + 8, b + 8) for a, b in m2.stacks[0]]
                mll.stacks[1] = m1.stacks[1] + [(a + 8, b + 8) for a, b in m2.stacks[1]]
                mll.queues[0] = m1.queues[0] + [(a + 8, b + 8) for a, b in m2.queues[0]]

                if mll.is_valid():
                    # This order is the only possible order in which
                    # both K8 can interleave (except for the mirrored
                    # version). Note that the list is ordered
                    # except that 9 and 8 are swapped.
                    if order == [1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16]:
                        mlls.append(mll)
                    else:
                        print('The observation is false!')

    print('%s layouts found' % len(mlls))
    for mll in mlls:
        pass  # Uncomment the next line to show the layouts
        #show_linear_layouts([mll])


def observation_3():
    """
    Show that given two complete graphs with 8 vertices that share two vertices
    there is only one vertex order possible where the shared vertices are
    exactly in the middle.
    """
    mlls_1 = observation_1(show_layouts=False)
    mlls_2 = []
    # create a second list of layouts that contains the same layouts and
    # additionally all of them with the stack pages swapped
    for mll in copy.deepcopy(mlls_1):
        mlls_2.append(mll)
        mll_new = copy.deepcopy(mll)
        mll_new.stacks[0], mll_new.stacks[1] = \
            mll_new.stacks[1], mll_new.stacks[0]
        mlls_2.append(mll_new)

    mlls = []
    # The graph has 14 vertices. Generate all pairs. These pairs are going to
    # be the two shared vertices
    for sv1, sv2 in itertools.combinations(list(range(1, 15)), 2):
        # Get lists of vertices that belong each K8
        if sv1 > 6:
            left_vertices = [1, 2, 3, 4, 5, 6]
        elif sv2 > 7:
            left_vertices = [1, 2, 3, 4, 5, 6, 7]
            left_vertices.remove(sv1)
        else:
            left_vertices = [1, 2, 3, 4, 5, 6, 7, 8]
            left_vertices.remove(sv1)
            left_vertices.remove(sv2)

        if sv2 < 9:
            right_vertices = [9, 10, 11, 12, 13, 14]
        elif sv1 < 8:
            right_vertices = [8, 9, 10, 11, 12, 13, 14]
            right_vertices.remove(sv2)
        else:
            right_vertices = [7, 8, 9, 10, 11, 12, 13, 14]
            right_vertices.remove(sv1)
            right_vertices.remove(sv2)

        graph = Graph()
        for v in range(1, 15):
            graph.add_vertex(v)
        for i in left_vertices + [sv1, sv2]:
            for j in left_vertices + [sv1, sv2]:
                if i != j:
                    graph.add_edge(i, j)
        for i in right_vertices + [sv1, sv2]:
            for j in right_vertices + [sv1, sv2]:
                if i != j:
                    graph.add_edge(i, j)

        for m1 in mlls_1:
            for m2 in mlls_2:
                order = list(range(1, 15))
                mll = LinearLayout(graph=graph, order=order,
                                   stacks=2, queues=1)

                l_vertices = copy.copy(left_vertices)
                l_vertices.extend([sv1, sv2])
                l_vertices.sort()
                r_vertices = copy.copy(right_vertices)
                r_vertices.extend([sv1, sv2])
                r_vertices.sort()

                # Create a new layout by combining the layouts of m1 and m2.
                transformation = (
                    (mll.stacks[0], m1.stacks[0], l_vertices),
                    (mll.stacks[1], m1.stacks[1], l_vertices),
                    (mll.queues[0], m1.queues[0], l_vertices),
                    (mll.stacks[0], m2.stacks[0], r_vertices),
                    (mll.stacks[1], m2.stacks[1], r_vertices),
                    (mll.queues[0], m2.queues[0], r_vertices),
                )
                for target_page, source_page, vertex_list in transformation:
                    for edge in source_page:
                        v1 = vertex_list[edge[0] - 1]
                        v2 = vertex_list[edge[1] - 1]
                        if v1 == sv1 and v2 == sv2 \
                                and (v1, v2) in mll.stacks[0] + mll.stacks[1] + mll.queues[0]:
                            # Don't add the edge between the shared vertices twice
                            continue
                        target_page.append((v1, v2))

                if mll.is_valid():
                    # The only way to get a valid layout is if the shared
                    # vertices are placed in the middle
                    if sv1 == 7 and sv2 == 8:
                        mlls.append(mll)
                    else:
                        print('The observation is false!')

    print('%s layouts found' % len(mlls))
    for mll in mlls:
        pass  # Uncomment the next line to show the layouts
        show_linear_layouts([mll])
