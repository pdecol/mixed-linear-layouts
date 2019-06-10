import copy
import datetime
import itertools
import os
import random
import statistics
import sys

from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from subprocess import run, PIPE
from threading import Thread

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from bipartite import LayeredGraph
from graph_generator import generate_planar_bipartite_graph, \
    generate_random_graph, generate_k_tree, generate_complete_graph, \
    generate_planar_graph, generate_maximal_bipartite_graph, reduce_edges, \
    GraphNotConnectedException
from heuristics.greedy_construction import full_combined_heuristic, \
    data_structure, con_greedy_plus_vertex_order, smallest_degree_dfs, \
    random_dfs, ceil_floor, con_greedy, random_bfs, edge_length
from heuristics.heuristics import CONSTRUCTION_HEURISTICS, \
    test_construction_heuristics, OPTIMIZATION_HEURISTICS, \
    test_optimization_heuristics, page_assignment_heuristics
from io_parser import DotParser, PickleParser, ASCIIParser, GraphML
from models import MixedLinearLayout, StackLinearLayout, Graph, LinearLayout
from observations.observations import observation_1, observation_2, \
    observation_3
from utils import triangles_of_graph
from view import show_linear_layouts, show_circular_layout, show_layered_graph


HEURISTIC_RESULTS_DIRECTORY = 'D:\\TUWien\\Diplomarbeit\\Latex\\R'

be = 'be'
lingeling = 'lingeling'
if os.name == 'nt':
    be += '.exe'
    lingeling += '.exe'


def solve_dimacs(res, mod):
    ascii_file = os.path.join(base_path, 'plantri.txt')
    output_dir = os.path.join(base_path, 'dot')
    dot_parser = DotParser()
    i_solved = 0
    for i, graph in enumerate(ASCIIParser().read_all(ascii_file), 1):
        if i % mod != res:
            continue
        file = 'test_%s.dot' % i
        file_name = file.split('.')[0]
        # ASCII to DOT
        dot_parser.write(graph, file, output_dir)

        # Create DIMACS
        command = [os.path.join(base_path, be),
                   '-i=' + os.path.join(base_path, 'dot', file_name + '.dot'),
                   '-o=' + os.path.join(base_path, 'dimacs',
                                        file_name + '.dimacs'),
                   '-type=mixed', '-pages=2', '-verbose=true']
        run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

        # Solve DIMACS
        command = [os.path.join(base_path, lingeling),
                   os.path.join(base_path, 'dimacs', file_name + '.dimacs')]
        result = run(command, stdout=PIPE, stderr=PIPE,
                     universal_newlines=True)
        i_solved += 1
        if 's SATISFIABLE\n' in result.stdout:
            print('%s OK' % i)
            os.remove(os.path.join(base_path, 'dimacs', file_name + '.dimacs'))
            os.remove(os.path.join(base_path, 'dot', file_name + '.dot'))
        else:
            print(file, 'COUNTEREXAMPLE!')
            sys.exit()

        if i_solved % 1000 == 0:
            print(i_solved, 'dimacs solved')
    print('Finish. %s DIMACS solved' % i_solved)


class DIMACSThread(Thread):
    def __init__(self, thread_id, res, mod):
        Thread.__init__(self)
        self.thread_id = thread_id
        self.res = res
        self.mod = mod

    def run(self):
        print('Thread %s started' % self.thread_id)
        solve_dimacs(self.res, self.mod)
        print('Thread %s finished' % self.thread_id)


if __name__ == '__main__':
    graph = mll = sll = None
    layouts = []
    base_path = os.path.dirname(os.path.realpath(__file__))

    for arg in sys.argv[1:]:
        if arg.endswith('.dot'):
            graph = DotParser().read(sys.argv[1])
            print(graph)

        elif arg.endswith('.bin'):
            obj = PickleParser().read(sys.argv[1])
            if isinstance(obj, LinearLayout):
                sll = obj
                print(sll.order)
                print(sll.get_page(1))
                print(sll.get_page(2))
                graph = sll.graph
                print(graph)
                layouts.append(sll)

        elif arg == 'mll':
            if not graph:
                print('No graph exists to create a layout')
                sys.exit()
            start_time = datetime.datetime.now()
            #mll = MixedLinearLayout.create_fixed_order(graph, [28, 29, 27, 25, 20, 2, 30, 26, 24, 23, 21, 19, 1, 3, 13, 15, 22, 16, 18, 4, 6, 12, 14, 7, 17, 5, 11, 8, 10, 9])
            #mll = MixedLinearLayout.create_fixed_order(graph, [28, 27, 29, 25, 23, 21, 20, 30, 26, 24, 22, 19, 1, 13, 15, 16, 18, 4, 2, 6, 12, 14, 7, 17, 3, 5, 11, 8, 10, 9])
            #mll = MixedLinearLayout.create_fixed_order(graph, [1, 30, 13, 2, 4, 20, 27, 29, 26, 24, 15, 14, 12, 6, 3, 5, 7, 16, 17, 19, 22, 21, 23, 28, 25, 11, 8, 10, 18, 9])
            #mll = MixedLinearLayout.create_fixed_order(graph, [1, 30, 13, 2, 4, 20, 27, 29, 26, 24, 14, 12, 6, 3, 5, 7, 17, 19, 21, 23, 28, 25, 15, 11, 8, 10, 16, 18, 22, 9])
            #mll = MixedLinearLayout.create_fixed_order(graph, graph._get_internal_vertices([1, 30, 2, 4, 20, 27, 29, 13, 5, 7, 17, 19, 21, 23, 26, 24, 14, 16, 22, 25, 15, 11, 18, 10, 9, 8, 6, 12, 28, 3]))
            #mll = MixedLinearLayout.create_fixed_order(graph, graph._get_internal_vertices([9, 11, 15, 8, 10, 6, 14, 24, 20, 23, 22, 16, 7, 17, 25, 27, 18, 19, 1, 21, 3, 4, 26, 28, 29, 30, 2, 13, 5, 12]))

            # DA SAMPLE GRAPH
            mll = MixedLinearLayout.create_fixed_order(graph, graph._get_internal_vertices([5, 30, 8, 29, 16, 7, 12, 18, 3, 20, 15, 27, 19, 10, 23, 22, 24, 17, 11, 21, 28, 13, 6, 2, 25, 9, 14, 4, 26, 1]))

            #mll = MixedLinearLayout.create_fixed_order(graph, [1, 2, 3, 4, 5, 6])

            #mll = MixedLinearLayout.create_fixed_order(graph, [32, 34, 33, 38, 36, 5, 37, 39, 35, 31, 30, 28, 8, 6, 4, 40, 2, 18, 29, 27, 25, 9, 3, 13, 17, 19, 24, 14, 16, 22, 20, 21, 15, 23, 10, 12, 11, 26, 7, 1])
            #mll = MixedLinearLayout.create_fixed_order(graph, [1, 40, 39, 35, 2, 3, 4, 38, 37, 36, 32, 34, 31, 18, 17, 20, 22, 16, 14, 13, 12, 9, 8, 7, 6, 5, 33, 30, 29, 19, 21, 23, 15, 24, 11, 10, 28, 26, 25, 27])

            #mll = MixedLinearLayout.create_fixed_order(graph, [1, 50, 12, 8, 7, 6, 2, 3, 17, 19, 25, 26, 27, 46, 47, 49, 48, 45, 15, 13, 11, 9, 5, 4, 10, 16, 42, 18, 20, 24, 28, 44, 14, 43, 29, 41, 40, 39, 21, 23, 22, 30, 32, 33, 34, 35, 36, 38, 31, 37])
            #mll = MixedLinearLayout.create(graph)
            layouts.append(mll)
            print(datetime.datetime.now() - start_time)
            print(mll)

        elif arg == 'mll-random':
            if not graph:
                print('No graph exists to create a layout')
                sys.exit()
            start_time = datetime.datetime.now()
            mll = MixedLinearLayout.create_random(graph)
            layouts.append(mll)
            print(datetime.datetime.now() - start_time)
            print(mll)

        elif arg == 'show':
            show_linear_layouts([layouts[-1]])
            #show_circular_layout(layouts[-1])

        elif arg == 'show-all':
            show_linear_layouts(layouts)

        elif arg.startswith('generate'):
            graph_class = arg.split('-')[1]
            vertices = int(arg.split('-')[2])
            density = Decimal(arg.split('-')[3])
            max_degree = Decimal(arg.split('-')[4]) if len(arg.split('-')) > 4 else None
            hamiltonian_path = False

            start_time = datetime.datetime.now()
            if graph_class == 'bipartite':
                graph, sll = generate_planar_bipartite_graph(vertices, density, max_degree)
                layouts.append(sll)
            elif graph_class == 'random':
                graph = generate_random_graph(vertices, density)
            else:
                print('Unknown graph class: ', graph_class)
                sys.exit()
            print(datetime.datetime.now() - start_time)
            print(graph)

        elif arg.startswith('dimacs'):
            # generate ascii with .\plantri.exe 12 -bp -f4 -m1 -c1 -v -a plantri.txt
            # planar bipartite, all faces of degree 4, minimum degree 1, minimum connectivity 1, verbose stdout, ascii output format
            # generate dimacs with .\be.exe "-i=dot\test_1.dot" "-o=dimacs\test_1.dimacs" -type=mixed -pages=2 -verbose=true
            # solve .\lingeling.exe test_1.dimacs

            # plantri 63 -bp -f4 -v -a 29/598147 plantri.txt --> 35032

            res, mod = 0, 1
            if len(arg.split('-')) > 2:
                res = int(arg.split('-')[1])
                mod = int(arg.split('-')[2])
            solve_dimacs(res, mod)

        elif arg.startswith('thread_dimacs'):
            num_threads = int(arg.split('-')[1])
            res_start = int(arg.split('-')[2])
            mod = int(arg.split('-')[3])
            print(num_threads, res_start, mod)
            threads = []
            for i in range(num_threads):
                thread = DIMACSThread(i, res_start + i, mod)
                thread.setName('Thread %s' % i)
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        elif arg.startswith('heuristic'):
            heuristic = full_combined_heuristic(random_dfs, ceil_floor)
            graph = generate_planar_graph(100)
            layout = heuristic(graph)
            mll = MixedLinearLayout(graph, layout.order)
            mll.stack = layout.stacks[0]
            mll.queue = layout.queues[0]

        elif arg.startswith('h-construction-'):
            # type = arg.split('-')[2]
            # runs = int(arg.split('-')[3])
            # graph_class = arg.split('-')[4]
            # graph_arg_1 = int(arg.split('-')[5])
            # if len(arg.split('-')) > 6 and arg.split('-')[6]:
            #     graph_arg_2 = int(arg.split('-')[6])
            type = 'single'
            runs = 1  # 500
            graph_class = arg.split('-')[2]
            #graph_arg_1 = int(arg.split('-')[3])
            if len(arg.split('-')) > 3 and arg.split('-')[3]:
                graph_arg_2 = int(arg.split('-')[3])

            graph_generator_kwargs = {
                #'num_vertices': graph_arg_1,
            }

            if graph_class == 'planar':
                graph_generator = generate_planar_graph
            elif graph_class == 'planar_bipartite':
                graph_generator = generate_planar_bipartite_graph
                graph_generator_kwargs['edge_density'] = 1
            elif graph_class == 'random':
                graph_generator = generate_random_graph
                graph_generator_kwargs['num_edges'] = graph_arg_2
            elif graph_class == 'k_tree':
                graph_generator = generate_k_tree
                graph_generator_kwargs['k'] = graph_arg_2
            elif graph_class == 'complete':
                graph_generator = generate_complete_graph

            heuristics = CONSTRUCTION_HEURISTICS
            heuristic_ids = {key: i for i, key in
                             enumerate(heuristics.keys(), start=1)}
            heuristic_name_ids = {
                'con_greedy_plus': '1',

                'random_dfs': '2',
                'smallest_degree_dfs': '3',
                'random_bfs': '4',
                'tree_bfs': '5',
                'con_greedy': '6',
                'con_greedy_plus_vertex_order': '7',

                'edge_length': '2',
                'ceil_floor': '3',
                'data_structure': '4',
            }
            heuristic_names = {
                'con_greedy_plus': 'conGreedy+',

                'random_dfs': 'randDFS',
                'smallest_degree_dfs': 'smlDgrDFS',
                #'highest_degree_dfs': 'hghDgrDFS',
                'random_bfs': 'randBFS',
                'tree_bfs': 'treeBFS',
                'con_greedy': 'conGreedy',
                'con_greedy_plus_vertex_order': 'conGreedy+',

                'edge_length': 'eLen',
                'ceil_floor': 'ceilFloor',
                'data_structure': 'dataStructure',
            }
            print('graph_class,vertices,edges,heuristic,conflicts,conflicts_per_edge')
            if type == 'single':
                for i in range(2, 11):  # 21
                    num_vertices = i * 5
                    for run in range(runs):
                        graph_generator_kwargs['num_vertices'] = num_vertices
                        if 'num_edges' in graph_generator_kwargs:
                            graph_generator_kwargs['num_edges'] = num_vertices * graph_arg_2

                        graph = graph_generator(**graph_generator_kwargs)
                        order = smallest_degree_dfs(graph)  # TODO adjust heuristic

                        heuristics = (
                            (edge_length, 'eLen'),
                            (ceil_floor, 'ceilFloor'),
                            (data_structure, 'dataStructure'),
                        )
                        for heuristic, heuristic_name in heuristics:
                            order_ = order.copy()
                            mll = heuristic(graph, order)
                            if not mll.is_order_and_edges_valid():
                                raise Exception('Heuristic %s is wrong' % name)
                            conflicts = mll.total_conflicts()

                            v = len(graph.vertices)
                            e = len(graph.edges)
                            print(','.join(['random_6n', str(v), str(e), heuristic_name, str(conflicts), str(conflicts/e)]))  # TODO adjust name


            elif type == 'pages':
                latex_table = '''
                \\begin{tabular}{m{0.65cm} | *{7}{m{0.6cm}}}
                 & 0-S & 1-S & 2-S & 3-S & 4-S & 5-S \\\\
                \cmidrule{1-7}
                0-Q &         & {{0-1}} & {{0-2}} & {{0-3}} & {{0-4}} & {{0-5}} \\\\
                1-Q & {{1-0}} & {{1-1}} & {{1-2}} & {{1-3}} & {{1-4}} & {{1-5}} \\\\
                2-Q & {{2-0}} & {{2-1}} & {{2-2}} & {{2-3}} & {{2-4}} & {{2-5}} \\\\
                3-Q & {{3-0}} & {{3-1}} & {{3-2}} & {{3-3}} & {{3-4}} & {{3-5}} \\\\
                4-Q & {{4-0}} & {{4-1}} & {{4-2}} & {{4-3}} & {{4-4}} & {{4-5}} \\\\
                5-Q & {{5-0}} & {{5-1}} & {{5-2}} & {{5-3}} & {{5-4}} & {{5-5}} \\\\
                \midrule
                \multicolumn{8}{l}{{{L-1}} {{L-2}} {{L-3}} {{L-4}}} \\\\
                \multicolumn{8}{l}{{{L-5}} {{L-6}} {{L-7}} {{L-8}}} \\\\
                \end{tabular}
                '''

                latex_cell = '\includegraphics[width=0.6cm, height=0.6cm]{graphics/{{NAME}}}'

                heuristics_in_table = []
                for stacks in range(0, 6):
                    for queues in range(0, 6):
                        if stacks == 0 and queues == 0:
                            continue
                        print('Stacks:', stacks, ', Queues:', queues)
                        results, _ = test_construction_heuristics(runs, heuristics, graph_generator, graph_generator_kwargs, stacks, queues)
                        best_name, best_conflicts = results[0]
                        if results[1][1] == 0:
                            # second best heuristic has also 0 conflicts.
                            cell = ''
                        else:
                            heuristics_in_table.extend(best_name.split(' '))  # add VO and PA separately
                            cell = latex_cell.replace('{{NAME}}', best_name.replace(' ', '_'))  # replace space because this name will be used as a file name
                        latex_table = latex_table.replace('{{%s-%s}}' % (queues, stacks), cell)

                # create legend
                i = 1
                for heuristic, name in heuristic_names.items():
                    if heuristic in heuristics_in_table:
                        cell = latex_cell.replace('{{NAME}}', heuristic) + ' ' + name
                        latex_table = latex_table.replace('{{L-%s}}' % i, cell)
                        i += 1
                for j in range(i, 9):
                    latex_table = latex_table.replace('{{L-%s}}' % j, '')  # remove empty legend placeholders

                latex_table = latex_table.replace(' ' * 16, '')  # remove spaces from the template that is defined above
                #file_path = os.path.join(HEURISTIC_RESULTS_DIRECTORY, 'pages_%s.txt' % graph_class)
                #Path(file_path).write_text(latex_table)
                print(latex_table)

        elif arg.startswith('h-optimization'):
            runs = int(arg.split('-')[2])
            graph_class = arg.split('-')[3]
            graph_arg_1 = int(arg.split('-')[4])
            if len(arg.split('-')) > 5 and arg.split('-')[5]:
                graph_arg_2 = int(arg.split('-')[5])

            graph_generator_kwargs = {
                'num_vertices': graph_arg_1,
            }
            if graph_class == 'planar':
                graph_generator = generate_planar_graph
            elif graph_class == 'planar_bipartite':
                graph_generator = generate_planar_bipartite_graph
                graph_generator_kwargs['edge_density'] = 1
            elif graph_class == 'random':
                graph_generator = generate_random_graph
                graph_generator_kwargs['num_edges'] = graph_arg_1 * graph_arg_2
            elif graph_class == 'k_tree':
                graph_generator = generate_k_tree
                graph_generator_kwargs['k'] = graph_arg_2
            elif graph_class == 'complete':
                graph_generator = generate_complete_graph

            heuristics = OPTIMIZATION_HEURISTICS.values()
            results = {heuristic: [] for heuristic in heuristics}
            for i in range(runs):
                heuristic = full_combined_heuristic(con_greedy, ceil_floor)
                graph = graph_generator(**graph_generator_kwargs)
                layout = heuristic(graph)
                mll = MixedLinearLayout(graph, layout.order)
                mll.stack = layout.stacks[0]
                mll.queue = layout.queues[0]

                data = test_optimization_heuristics(mll, heuristics)
                print(i, data)
                for heuristic, result in data.items():
                    results[heuristic].append(result)

            for heuristic, result in results.items():
                print(heuristic.__name__)
                print(result)
                print(statistics.mean(result))

        elif arg == 'h-density':
            #min_density = 51  # percent, 36 for planar, 51 for bipartite
            #max_density = 100  # percent
            min_density = 0  # percent, 36 for planar, 51 for bipartite
            max_density = 190  # percent
            density_increase = 1  # percent
            runs_per_density = 1000
            graph_vertices = 100

            vertex_order_heuristic = random_bfs
            #vertex_order_heuristic = smallest_degree_dfs

            # choose the graph class
            graph_generator = generate_planar_graph
            #graph_generator = generate_planar_bipartite_graph

            results = {h.__name__: [] for h in page_assignment_heuristics}
            current_density = min_density
            while current_density <= max_density:
                for value in results.values():
                    value.append(0)
                for run in range(runs_per_density):
                    graph = graph_generator(graph_vertices, target_triangles=current_density)

                    # try:
                    #     graph = reduce_edges(graph, current_density)
                    # except GraphNotConnectedException as e:
                    #     print('Could not generate a graph for density: %s' % current_density)
                    #     exit()
                    for page_heuristic in page_assignment_heuristics:
                        heuristic = full_combined_heuristic(vertex_order_heuristic, page_heuristic)
                        layout = heuristic(graph)
                        results[page_heuristic.__name__][-1] += layout.total_conflicts()
                        #print(page_heuristic.__name__, current_density, layout.total_conflicts())

                print('Densitiy: %s' % current_density)
                for key, value in results.items():
                    value[-1] = int(value[-1] / runs_per_density)
                    print(key, value[-1])

                current_density += density_increase

            data_frame = pd.DataFrame(results)
            data_frame.index = range(min_density, max_density + 1, density_increase)

            #print(data_frame)
            data_frame.to_csv('planar_triangles.csv')

            #data_frame.plot()
            #plt.show()

        elif arg == 'plot':
            data_frame = pd.read_csv('planar_bipartite_graph.csv', index_col=0)

            for row in data_frame.iterrows():
                edges = (2 * 100 - 4) * (row[0] / 100)
                data_frame.ix[row[0], 0] = row[1]['ceil floor'] / edges
                data_frame.ix[row[0], 1] = row[1]['data structure'] / edges
                data_frame.ix[row[0], 2] = row[1]['edge length'] / edges

            data_frame.to_csv('planar_bipartite_graph_2.csv')

            ax = data_frame.plot(title='Planar bipartite')
            ax.set_xlabel('Density')
            ax.set_ylabel('Conflicts / Edge')
            plt.show()

        elif arg == 'plot2':
            tests = [
                ('planar', 'Delaunay triangulated graphs'),
                ('planar_bipartite', 'Planar bipartite graphs'),
                ('2-tree', '2-trees'),
                ('3-tree', '3-trees'),
                ('random_3n', 'Random graphs with 3n edges'),
                ('random_6n', 'Random graphs with 6n edges'),
                ('complete', 'Complete graphs'),
            ]
            for graph_class, title in tests:

                import seaborn as sns
                from pylab import rcParams
                rcParams['figure.figsize'] = 5, 5

                sns.set(style="darkgrid")

                data_frame = pd.read_csv('data/%s.csv' % graph_class, index_col=0)

                ax = sns.lineplot(x="vertices", y="conflicts_per_edge",
                                  hue="heuristic", ci='sd', #style="event",
                                  data=data_frame)
                ax.set(xlabel='Vertices', ylabel='Conflicts per edge')
                ax.set_title(title)

                #ax.set(ylim=(0, 24))
                ax.set(ylim=(0, 3))

                legend = ax.legend(loc='upper left')
                legend.texts[0].set_text("Heuristics")

                plt.savefig('figures/%s.png' % graph_class)
                plt.show()

        elif arg == 'boxplot':
            import seaborn as sns

            sns.set(style="ticks", palette="pastel")

            data_frame = pd.read_csv('all_data.csv')

            sns.boxplot(x="Graph classes", y="Conflicts", hue="Heuristics", palette=["m", "g", "b"], data=data_frame)
            sns.despine(offset=10, trim=True)

            plt.show()

        elif arg == 'best':
            def chunks(l, n):
                """Yield successive n-sized chunks from l."""
                for i in range(0, len(l), n):
                    yield l[i:i + n]

            tests = [
                ('planar', 'Delaunay triangulated graphs'),
                ('planar_bipartite', 'Planar bipartite graphs'),
                ('2-tree', '2-trees'),
                ('3-tree', '3-trees'),
                ('random_3n', 'Random graphs with 3n edges'),
                ('random_6n', 'Random graphs with 6n edges'),
                #('complete', 'Complete graphs'),
            ]

            for graph_class, title in tests:
                results = {}
                with open('data/%s.csv' % graph_class) as f:
                    lines = f.readlines()[1:]
                    for a, b, c in chunks(lines, 3):
                        a = a.split(',')
                        b = b.split(',')
                        c = c.split(',')
                        graph_size = a[1]
                        if graph_size not in results:
                            results[graph_size] = defaultdict(int)
                        if float(a[-1]) < float(b[-1]) and float(a[-1]) < float(c[-1]):
                            results[graph_size][a[3]] += 1
                        elif float(b[-1]) < float(a[-1]) and float(b[-1]) < float(c[-1]):
                            results[graph_size][b[3]] += 1
                        elif float(c[-1]) < float(a[-1]) and float(c[-1]) < float(b[-1]):
                            results[graph_size][c[3]] += 1

                with open('conflicts_2/%s.csv' % graph_class, 'w+') as file:
                    file.write('graph_class,vertices,eLen,ceilFloor,dataStructure\n')
                    for i in range(10, 101, 5):
                        i = str(i)
                        line = '%s,%s' % (graph_class, i)
                        total = results[i]['dataStructure'] + results[i]['ceilFloor'] + results[i]['eLen']
                        for heuristic in ['eLen', 'ceilFloor', 'dataStructure']:
                            best = results[i][heuristic]
                            line += ',%s' % str(best/total)

                        file.write(line)
                        file.write('\n')

                import seaborn as sns
                from pylab import rcParams
                rcParams['figure.figsize'] = 5, 5
                sns.set(style="darkgrid")
                #sns.set(style="whitegrid")

                df = pd.read_csv('conflicts_2/%s.csv' % graph_class, index_col=0)

                #g = sns.catplot(x="vertices", y="best_percent", hue="heuristic", data=data_frame,
                #                 height=6, kind="bar", palette="muted")
                # g.fig.suptitle(title)
                # g.despine(left=True)
                # g.set_ylabels("Best in percent")
                # g.set_xlabels("Number of vertices")

                # sns.set()
                # df.set_index('vertices').T.plot(kind='bar', stacked=True)

                r = [x for x in range(0, 19)]

                # From raw value to percentage
                totals = [1 for x in range(19)]
                greenBars = [i / j * 100 for i,j in zip(df['dataStructure'], totals)]
                orangeBars = [i / j * 100 for i,j in zip(df['ceilFloor'], totals)]
                blueBars = [i / j * 100 for i,j in zip(df['eLen'], totals)]

                # plot
                barWidth = 1
                #names = [str(x) for x in range(10, 105, 5)]
                names = ['10', '', '20', '', '30', '', '40', '', '50', '', '60', '', '70', '', '80', '', '90', '', '100']
                plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="dataStructure")
                plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="ceilFloor")
                plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth, label="eLen")

                # Custom x axis
                plt.xticks(r, names)
                plt.ylabel("Best in percent")
                plt.xlabel("Vertices")
                plt.gca().xaxis.grid(False)

                # Add a legend
                plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1, framealpha=0.8)

                plt.suptitle(title)

                plt.savefig('conflict_figures_2/%s.png' % graph_class)
                plt.show()

        elif arg == 'all_data':
            graph_classes = (
                'planar',
                'planar_bipartite',
                '2_tree',
                '3_tree',
                'random_3n',
                #'random_6n',
            )
            for graph in graph_classes:
                lines = []
                with open('box_plot_%s.txt' % graph) as file:
                    for line in file:
                        if 'HEURISTIC;CONFLICTS' in line:
                            continue

                        algo = line.split(';')[0]
                        conflicts = line.split(';')[1]
                        conflicts = conflicts.replace('\n', '')
                        a = 'ERROR'
                        if 'eLen' in algo:
                            a = 'eLen'
                        elif 'ceilFloor'in algo:
                            a = 'ceilFloor'
                        elif 'dataStructure'in algo:
                            a = 'dataStructure'

                        lines.append('%s,%s,%s\n' % (a, conflicts, graph))

                with open('all_data.csv', 'a+') as file:
                    for line in lines:
                        file.write(line)

        elif arg == 'h-parameter':
            def parameter_optimization(alpha):
                runs = 10000
                conflicts = 0
                for i in range(runs):
                    graph = generate_random_graph(num_vertices=50, num_edges=200)
                    order = smallest_degree_dfs(graph)
                    mll = data_structure(graph, order, alpha)
                    conflicts += mll.total_conflicts()
                print(alpha, conflicts/runs)
                return conflicts / runs


            #x0 = np.array([0.5, ])
            #res = minimize(parameter_optimization, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
            #print(res)
            #print(res.x)

            parameter_optimization(0.49)

        elif arg.startswith('h-'):
            heuristic = arg.split('-')[1]
            mll = CONSTRUCTION_HEURISTICS[heuristic](graph)
            layouts.append(mll)
            print(mll)
            print('Crossings: ', mll.total_crossings())
            print('Nestings: ', mll.total_nestings())

        elif arg == 'rome_graphs':
            rome_directory = os.path.join(base_path, 'rome')
            results_file = 'rome_results_queue.txt'
            dot_parser = DotParser()
            results = []
            i = 0
            #with open(os.path.join(base_path, results_file), 'r') as myfile:
                #solved = myfile.read()
            for file_name in os.listdir(rome_directory):
                if not file_name.endswith('.graphml'):
                    continue
                file_id = file_name.replace('.graphml', '')
                #if file_id in solved:
                    #continue
                if int(file_id.split('.')[1]) >= 90:
                    with open(results_file, 'a') as f:
                        f.write('%s skipped\n' % file_id)
                    continue
                i += 1
                graph = GraphML().read(os.path.join(rome_directory, file_name))

                dot_parser.write(graph, file_id + '.dot', os.path.join(base_path, 'dot_queue'))

                # Create DIMACS
                command = [os.path.join(base_path, be),
                           '-i=' + os.path.join(base_path, 'dot_queue',
                                                file_id + '.dot'),
                           '-o=' + os.path.join(base_path, 'dimacs_queue',
                                                file_id + '.dimacs'),
                           '-type=queue', '-pages=2', '-verbose=true']
                run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

                # Solve DIMACS
                command = [os.path.join(base_path, lingeling),
                           os.path.join(base_path, 'dimacs_queue',
                                        file_id + '.dimacs')]
                result = run(command, stdout=PIPE, stderr=PIPE,
                             universal_newlines=True)

                print('%s graphs testes' % i)
                os.remove(os.path.join(base_path, 'dot_queue', file_id + '.dot'))
                os.remove(os.path.join(base_path, 'dimacs_queue', file_id + '.dimacs'))

                with open(results_file, 'a') as f:
                    if 's SATISFIABLE\n' in result.stdout:
                        f.write('%s satisfiable\n' % file_id)
                    else:
                        f.write('%s invalid\n' % file_id)

        elif arg == 'bipartite':
            plantri_file = os.path.join(base_path, 'plantri.txt')
            graph = ASCIIParser().read(plantri_file, 1)

            layered = LayeredGraph.create_layers_bfs(graph)
            layered.colour_edges()
            layered.move_ending_vertex_up()
            layered.colour_edges()
            print(layered.layers)

            mll_computed = MixedLinearLayout.create_fixed_order(graph, layered.get_order())
            print('Order allows MLL:', bool(mll_computed))

            mll = MixedLinearLayout(graph, layered.get_order())
            mll.stack.extend(layered.get_stack())
            mll.queue.extend(layered.get_queue())
            if not mll.is_order_and_edges_valid():
                raise Exception('Order or edges invalid')
            print('Conflicts: ', mll.total_conflicts())


            show_layered_graph(layered)
            #show_linear_layouts([mll])
            #show_circular_layout(mll_computed)

        elif arg == 'bipartite-test':
            # plantri.exe 30 -bp -f4 -m1 -c1 -v -a 777777/10000000 plantri.txt
            perfect_solutions = 0
            order_solutions = 0
            any_start_vertex_perfect_solutions = 0
            any_start_vertex_order_solution = 0
            for i in range(1001, 2385):
                print(i)
                is_any_start_solution = False
                is_any_order_solution = False
                for j in range(1, 31):
                    plantri_file = os.path.join(base_path, 'plantri.txt')
                    graph = ASCIIParser().read(plantri_file, i)
                    layered = LayeredGraph.create_layers_bfs(graph, start_vertex_id=j)
                    layered.colour_edges()
                    layered.move_ending_vertex_up()
                    layered.colour_edges()

                    mll = MixedLinearLayout(graph, layered.get_order())
                    mll.stack.extend(layered.get_stack())
                    mll.queue.extend(layered.get_queue())
                    if mll.is_valid():
                        is_any_start_solution = True
                        if j == 1:
                            perfect_solutions += 1

                    if MixedLinearLayout.create_fixed_order(graph, layered.get_order()):
                        is_any_order_solution = True
                        if j == 1:
                            order_solutions += 1

                    if is_any_start_solution and is_any_order_solution:
                        continue
                if is_any_start_solution:
                    any_start_vertex_perfect_solutions += 1
                if is_any_order_solution:
                    any_start_vertex_order_solution += 1

            print('Perfect solutions:', perfect_solutions)
            print('Order solutions:', order_solutions)
            print('Any perfect solutions:', any_start_vertex_perfect_solutions)
            print('Any Order solutions:', any_start_vertex_order_solution)

        elif arg == 'test':
            # enumerate K6 on 1-stack 1-queue layouts
            print(graph.edges)
            valid_layouts_found = 0
            for p in list(itertools.product([True, False], repeat=15)):
                mll = MixedLinearLayout(graph, order=list(range(1, 7)))
                edges = graph.edges
                for i in range(15):
                    edge = edges[i]
                    # if (abs(edge[0] - edge[1])) == 1 and not p[i]:
                    #     # skip layouts where edges between neighbours are not on the stack page
                    #     continue
                    if p[i]:
                        mll.stacks[0].append(edge)
                    else:
                        mll.queues[0].append(edge)

                if mll.is_valid():
                    print(mll)
                    valid_layouts_found += 1
                    show_linear_layouts([mll])
            print('valid layouts', valid_layouts_found)

        elif arg == 'test2':
            num_vertices = 8
            stacks = 2
            queues = 1

            graph = generate_complete_graph(num_vertices)
            num_edges = len(graph.edges)
            print(graph)

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

            # With shuffled edges it should be faster to find a mll if it exists
            random.shuffle(remaining_edges)

            i = 0
            mlls = []
            def step(mll, edges):
                global i
                i += 1

                edges = list(edges)
                if not edges:
                    return mll
                    #mlls.append(copy.deepcopy(mll))
                    #return None
                edge = edges.pop()

                for stack in mll.stacks:
                    stack.append(edge)
                    if mll.is_stack_valid(stack):
                        if step(mll, edges):
                            return mll
                    stack.remove(edge)

                for queue in mll.queues:
                    queue.append(edge)
                    if mll.is_queue_valid(queue):
                        if step(mll, edges):
                            return mll
                    queue.remove(edge)

            mll = step(mll, remaining_edges)
            print(mll)
            print(len(mlls))
            print(mlls)
            print(i)
            if mll:
                if not mll.is_valid():
                    raise Exception('MLL not valid')
                for stack in mll.stacks:
                    print(stack)
                for queue in mll.queues:
                    print(queue)

                show_linear_layouts([mll])

            for mll in mlls:
                if not mll.is_valid():
                    raise Exception('MLL not valid')
                show_linear_layouts([mll])

        elif arg == 'observation_1':
            observation_1()

        elif arg == 'observation_2':
            observation_2()

        elif arg == 'observation_3':
            observation_3()

        else:
            print('Unknown argument: ', arg)
            sys.exit()
