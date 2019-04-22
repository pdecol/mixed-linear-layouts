import os
import pickle
import re
from pathlib import Path

from models import Graph


class AbstractIO:
    input_directory = 'instances'
    output_directory = 'output'
    test_directory = 'tests/instances'
    default_filename = 'test'

    def read(self, name):
        raise NotImplementedError

    def write(self, graph, file_name=None):
        raise NotImplementedError

    def _read_from_file(self, file_name, is_test=False):
        directory = self.test_directory if is_test else self.input_directory
        file_path = os.path.join(os.getcwd(), directory, file_name)
        return Path(file_path).read_text()

    def _write_to_file(self, string, file_name, output_directory=None):
        if not output_directory:
            output_directory = self.output_directory
        self._ensure_output_directory()
        file_path = os.path.join(os.getcwd(), output_directory, file_name)
        Path(file_path).write_text(string)

    def _ensure_output_directory(self):
        directory = os.path.join(os.getcwd(), self.output_directory)
        if not os.path.exists(directory):
            os.makedirs(directory)


class DotParser(AbstractIO):
    def read(self, file_name, is_test=False):
        graph = Graph()
        string = self._read_from_file(file_name, is_test)
        string = string.replace(';', '')
        tokens = string.split()

        vertex_1 = vertex_2 = edge = None
        for token in tokens:
            if token.isdigit():
                if not vertex_1:
                    vertex_1 = token
                else:
                    vertex_2 = token
            elif token == '--':
                edge = token

            if vertex_1 and vertex_2 and edge:
                graph.add_edge(vertex_1, vertex_2)
                vertex_1 = vertex_2 = edge = None
        return graph

    def write(self, graph, file_name=None, output_directory=None):
        string = 'graph {\n'
        for edge in graph.edges:
            string += '  %s -- %s;\n' % (edge[0], edge[1])
        string += '}'

        if not file_name:
            file_name = self.default_filename + '.dot'
        self._write_to_file(string, file_name, output_directory)


class PickleParser(AbstractIO):
    def read(self, file_name):
        self._ensure_output_directory()
        with open(self.output_directory + '/' + file_name, mode='rb')\
                as binary_file:
            return pickle.load(binary_file)

    def write(self, obj, file_name=None):
        if not file_name:
            file_name = self.default_filename + '.bin'
        self._ensure_output_directory()
        with open(self.output_directory + '/' + file_name, mode='wb')\
                as binary_file:
            pickle.dump(obj, binary_file)


class ASCIIParser(AbstractIO):
    """
    Parse graphs that are created with Plantri
    """
    def read(self, file, number=1):
        with open(file, 'r', encoding='latin1') as f:
            for i, line in enumerate(f.readlines(), 1):
                if i == number:
                    return self._parse(line)

    def read_all(self, file):
        with open(file, 'r', encoding='latin1') as f:
            for i, line in enumerate(f.readlines(), 1):
                yield self._parse(line)

    def _parse(self, line):
        graph = Graph()
        line = line.split(' ')[1].replace('\n', '')  # remove number of vertices
        vertices = line.split(',')
        current_vertex = ord('a') - 96  # a == 97
        for vertex in vertices:
            neighbors = []
            for neighbor_vertex in vertex:
                neighbor = ord(neighbor_vertex) - 96
                neighbors.append(neighbor)
                graph.add_edge(current_vertex, neighbor)
            graph.add_circular_edge_order(current_vertex, neighbors)  # counterclockwise order of neighbours. in the ascii file it is clockwise
            current_vertex += 1

        return graph


class GraphML(AbstractIO):
    def read(self, file):
        graph = Graph()
        with open(file, 'r') as f:
            # Quick way to parse the rome graphs
            for line in f.readlines():
                if line.startswith('<node'):
                    v = re.search(r'"(.+)"', line).group(1)
                    graph.add_vertex(v)
                elif line.startswith('<edge'):
                    res = re.findall(r'"(.+?)"', line)
                    v1 = res[1]  # [0} is the edge id
                    v2 = res[2]
                    graph.add_edge(v1, v2)

        return graph
