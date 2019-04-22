from copy import deepcopy
from datetime import datetime

from heuristics.greedy_construction import random_dfs, edge_length, \
    con_greedy_plus, \
    smallest_degree_dfs, highest_degree_dfs, full_combined_heuristic, \
    ceil_floor, con_greedy_plus_vertex_order, random_bfs, tree_bfs, \
    connectivity, \
    data_structure, unchanged_vertex_order, con_greedy
from heuristics.greedy_optimisation import greedy_alt_rr, greedy_alt_re, \
    greedy_alt_er, greedy_alt_ee
from io_parser import PickleParser

vertex_order_heuristics = [
    random_dfs,
    smallest_degree_dfs,
    #highest_degree_dfs,  # Not good
    random_bfs,
    tree_bfs,
    con_greedy,
    con_greedy_plus_vertex_order,
    #unchanged_vertex_order,  # Only for testing
]
page_assignment_heuristics = [
    edge_length,
    ceil_floor,
    data_structure,
    # connectivity,  # Not good
]
CONSTRUCTION_HEURISTICS = {
    'con_greedy_plus': con_greedy_plus
}
for vertex_order in vertex_order_heuristics:
    for page_assignment in page_assignment_heuristics:
        name = ' '.join([vertex_order.__name__, page_assignment.__name__])
        #name = ' '.join([page_assignment.__name__])
        CONSTRUCTION_HEURISTICS[name] = \
            full_combined_heuristic(vertex_order, page_assignment)


OPTIMIZATION_HEURISTICS = {
    '1': greedy_alt_rr,
    '2': greedy_alt_re,
    '3': greedy_alt_er,
    '4': greedy_alt_ee,
}


def test_construction_heuristics(runs, heuristics, graph_generator, graph_generator_kwargs, stacks=1, queues=1):
    best = {}
    conflicts = {}
    all_data = []
    for name, heuristic in heuristics.items():
        best[name] = 0
        conflicts[name] = 0

    for i in range(0, runs):
        graph = graph_generator(**graph_generator_kwargs)
        print('Graph, vertices: %s, edges: %s' % (len(graph.vertices), len(graph.edges)))
        best_heuristics = []
        best_score = None
        for name, heuristic in heuristics.items():
            mll = heuristic(graph, stacks, queues)
            if not mll.is_order_and_edges_valid():
                raise Exception('Heuristic %s is wrong' % name)
            score = mll.total_conflicts()
            # if score == 0:
            #     PickleParser().write(mll, file_name='no_conflicts.bin')

            all_data.append([name, score])
            conflicts[name] += score
            if not best_heuristics or score == best_score:
                best_heuristics.append(name)
                best_score = score
            elif score < best_score:
                best_heuristics = [name]
                best_score = score

        for name in best_heuristics:
            best[name] += 1

        if (i + 1) % 10 == 0:
            print(i + 1, 'out of', runs)

    for k, v in conflicts.items():
        conflicts[k] = v // runs

    results = [(k, v) for k, v in conflicts.items()]
    results.sort(key=lambda x: x[1])
    for name, conflicts in results:
        print(name, round(conflicts))
    return results, all_data


def test_optimization_heuristics(mll, heuristics):
    conflicts_start = mll.total_conflicts()
    results = {}
    # print('mll conflicts start', mll.total_conflicts())
    # print('mll crossings start', mll.total_crossings())
    # print('mll nestings start', mll.total_nestings())
    for heuristic in heuristics:
        start = datetime.now()
        _mll = deepcopy(mll)
        _mll = heuristic(_mll)

        # improvements in percent
        results[heuristic] = (conflicts_start - _mll.total_conflicts()) / conflicts_start
        # print(heuristic.__name__)
        # print('Time: ', datetime.now() - start)
        # print('Conflicts: ', _mll.total_conflicts())
        # print('Crossings: ', _mll.total_crossings())
        # print('Nestings: ', _mll.total_nestings())
        # print('-----------------------')

    return results
