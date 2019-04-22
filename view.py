import math
import operator
import tkinter as tk
from collections import defaultdict

LINE_WIDTH = 2


def get_linear_vertices_position(order, canvas_height, vertex_size):
    positions = {}
    for i, vertex in enumerate(order):
        positions[vertex] = (i * vertex_size * 2 + vertex_size * 1.5,
                             canvas_height / 2)
    return positions


def get_circular_vertices_position(order, center, radius):
    positions = {}
    angles = {}
    one_slice = 2 * math.pi / len(order)
    for i, vertex in enumerate(order):
        angle = one_slice * i
        angles[vertex] = angle
        positions[vertex] = (center[0] + radius * math.cos(angle),
                             center[1] + radius * math.sin(angle))
    return positions, angles


def get_layered_vertices_position(layers):
    positions = {}
    current_height = 0
    for layer in layers:
        current_height += 100
        current_width = 0
        for vertex in layer:
            current_width += 75
            positions[vertex] = (current_width, current_height)
    return positions


def get_queue_colours(queue, order, alternating=True):
    colouring = {}

    colours_1 = ('orange', 'blue')
    #colours_1 = ('orange', 'blue', 'red', 'green')
    #colours_1 = ('red', 'blue', 'green', 'orange')
    #colours_1 = ('orange', 'purple', 'green')
    colours_2 = ('red', 'green', 'medium purple', 'orange', 'aquamarine2', 'orchid', 'yellow', 'blue', 'purple1', 'magenta', 'yellow3', 'blue4', 'deep pink')
    colours_blocked = defaultdict(int)
    current_vertex = None
    current_colour = None
    for v1, v2 in queue:
        current_index = order.index(v1)
        if current_vertex != v1:
            current_vertex = v1

            if alternating:
                current_colour = current_colour or colours_1[0]
                current_colour = colours_1[(colours_1.index(current_colour) + 1) % len(colours_1)]
            else:
                for colour in colours_2:
                    if current_index >= colours_blocked[colour]:
                        current_colour = colour
                        break

        colours_blocked[current_colour] = order.index(v2)

        colouring[(v1, v2)] = current_colour

    return colouring


def draw_vertices(canvas, graph, positions, vertex_size):
    for vertex in graph.vertices:
        x, y = positions[vertex]
        canvas.create_rectangle(x - vertex_size / 2, y - vertex_size / 2,
                                x + vertex_size / 2, y + vertex_size / 2,
                                fill='#cccccc')
        canvas.create_text(x, y, text=graph.get_vertex_label(vertex))


def draw_arch(canvas, canvas_height, layout, positions, vertex_size, v1, v2, page, color, upward=True):
    x1, y1 = positions[v1]
    x2, y2 = positions[v2]

    # longest_edge = (len(layout.order) - 1)
    longest_edge = max(page, key=lambda e: abs(layout.order.index(e[0]) - layout.order.index(e[1])))
    longest_edge = abs(layout.order.index(longest_edge[0]) - layout.order.index(longest_edge[1]))
    arch_min = vertex_size / 2
    arch_max = canvas_height / 2 - vertex_size
    arch_height = (arch_max - arch_min) * (1 - abs(layout.order.index(v2) - layout.order.index(v1)) / longest_edge)
    start = 0 if upward else 180
    canvas.create_arc(x1, vertex_size + arch_height,
                      x2, canvas_height - vertex_size - arch_height,
                      start=start, extent=180,
                      style='arc', outline=color, width=LINE_WIDTH)


def draw_stack_outside(stack, layout, canvas, canvas_size, center, radius, vertex_size, vertex_positions, vertex_angles):
    max_stack_height = (canvas_size / 2 - radius - vertex_size) * 0.9
    longest_edge = max(stack, key=lambda e: abs(layout.order.index(e[0]) - layout.order.index(e[1])))
    longest_edge = abs(layout.order.index(longest_edge[0]) - layout.order.index(longest_edge[1]))
    step = max_stack_height / longest_edge

    for v1, v2 in stack:
        if layout.order.index(v1) > layout.order.index(v2):
            v1, v2 = v2, v1
        height = abs(layout.order.index(v1) - layout.order.index(v2)) * step + vertex_size

        x1 = center[0] - radius - height
        y1 = center[1] - radius - height
        x2 = center[0] + radius + height
        y2 = center[1] + radius + height
        d1 = math.degrees(vertex_angles[v1])
        d2 = math.degrees(vertex_angles[v2])
        canvas.create_arc(x1, y1, x2, y2, start=-d1, extent=-(d2 - d1),
                          style='arc', outline='orange', width=LINE_WIDTH)

        x1, y1 = vertex_positions[v1]
        canvas.create_line(x1, y1, x1 + height * math.cos(vertex_angles[v1]),
                           y1 + height * math.sin(vertex_angles[v1]), width=LINE_WIDTH,
                           fill='black')
        x2, y2 = vertex_positions[v2]
        canvas.create_line(x2, y2, x2 + height * math.cos(vertex_angles[v2]),
                           y2 + height * math.sin(vertex_angles[v2]), width=LINE_WIDTH,
                           fill='black')


def draw_stack_inside(stack, canvas, vertex_positions):
    for v1, v2 in stack:
        x1, y1 = vertex_positions[v1]
        x2, y2 = vertex_positions[v2]

        canvas.create_line(x1, y1, x2, y2,
                           width=LINE_WIDTH, fill='black')


def _octilinear_bend_point(x1, y1, x2, y2, center, radius):
    # Find the point that is inside the circle. A line form v1 to
    # bend_point to v2 would produce a orthogonal drawing.
    if math.sqrt(pow((x1 - center[0]), 2) + pow((y2 - center[1]), 2)) \
            < radius:
        bend_point = [x1, y2]
        other = (x2, y1)
    else:
        bend_point = [x2, y1]
        other = (x1, y2)

    # Move the bend_point to a position where we get a octilinear drawing
    # with one bend of 45 degrees.
    if abs(y1 - y2) < abs(x1 - x2):
        bend_point[0] += abs(y1 - y2) \
            if bend_point[0] < other[0] else -abs(y1 - y2)
    else:
        bend_point[1] += abs(x1 - x2) \
            if bend_point[1] < other[1] else -abs(x1 - x2)
    return bend_point


def draw_queue_octilinear(queue, layout, canvas, vertex_positions, center, radius):
    queue = layout.get_edges_in_order(queue)
    colours = get_queue_colours(queue, layout.order)
    for v1, v2 in queue:
        edge_colour = colours[(v1, v2)]
        x1, y1 = vertex_positions[v1]
        x2, y2 = vertex_positions[v2]
        bend_point = _octilinear_bend_point(x1, y1, x2, y2, center, radius)

        canvas.create_line(x1, y1, bend_point[0], bend_point[1],
                           width=LINE_WIDTH, fill=edge_colour)
        canvas.create_line(x2, y2, bend_point[0], bend_point[1],
                           width=LINE_WIDTH, fill=edge_colour)


def draw_stack_octilinear(stack, canvas, vertex_positions, center, radius):
    for v1, v2 in stack:
        edge_colour = 'black'
        x1, y1 = vertex_positions[v1]
        x2, y2 = vertex_positions[v2]
        bend_point = _octilinear_bend_point(x1, y1, x2, y2, center, radius)

        canvas.create_line(x1, y1, bend_point[0], bend_point[1],
                           width=LINE_WIDTH, fill=edge_colour)
        canvas.create_line(x2, y2, bend_point[0], bend_point[1],
                           width=LINE_WIDTH, fill=edge_colour)


def draw_queue_radial(queue, layout, canvas, vertex_positions, vertex_angles, center, radius, draw_inside=True):
    op = operator.sub if draw_inside else operator.add

    queue = layout.get_edges_in_order(queue)
    height = 10
    colours = get_queue_colours(queue, layout.order, alternating=True)
    for v1, v2 in queue:
        edge_colour = colours[(v1, v2)]
        height += 8

        x1 = center[0] - op(radius, height)
        y1 = center[1] - op(radius, height)
        x2 = center[0] + op(radius, height)
        y2 = center[1] + op(radius, height)
        d1 = math.degrees(vertex_angles[v1])
        d2 = math.degrees(vertex_angles[v2])
        canvas.create_arc(x1, y1, x2, y2, start=-d1, extent=-(d2 - d1),
                          style='arc', outline=edge_colour, width=LINE_WIDTH)

        x1, y1 = vertex_positions[v1]
        x2, y2 = vertex_positions[v2]

        canvas.create_line(x1, y1, op(x1, height * math.cos(vertex_angles[v1])),
                           op(y1,  height * math.sin(vertex_angles[v1])),
                           width=LINE_WIDTH,
                           fill='black')

        canvas.create_line(x2, y2, op(x2, height * math.cos(vertex_angles[v2])),
                           op(y2, height * math.sin(vertex_angles[v2])),
                           width=LINE_WIDTH,
                           fill='black')


def draw_queue_radial_levels(queue, layout, canvas, vertex_positions, vertex_angles, draw_inside=True):
    op = operator.sub if draw_inside else operator.add

    order = layout.order
    queue = layout.get_edges_in_order(queue)
    colours = get_queue_colours(queue, order, alternating=True)
    current_heights = defaultdict(lambda: 1)
    height_min = 20
    height_step = 10
    for v1, v2 in queue:
        edge_colour = colours[(v1, v2)]
        x1, y1 = vertex_positions[v1]
        x2, y2 = vertex_positions[v2]

        height = current_heights[v1] * height_step + height_min
        current_point = (op(x1, height * math.cos(vertex_angles[v1])),
                         op(y1, height * math.sin(vertex_angles[v1])))
        canvas.create_line(x1, y1, current_point[0], current_point[1],
                           width=LINE_WIDTH * 2, fill=edge_colour)

        for v3 in order[order.index(v1) + 1:order.index(v2)]:
            x3, y3 = vertex_positions[v3]
            height = current_heights[v3] * height_step + height_min
            target_point = (op(x3, height * math.cos(vertex_angles[v3])),
                            op(y3, height * math.sin(vertex_angles[v3])))
            canvas.create_line(current_point[0], current_point[1],
                               target_point[0], target_point[1],
                               width=LINE_WIDTH, fill=edge_colour)
            current_point = target_point
            current_heights[v3] += 1

        canvas.create_line(current_point[0], current_point[1],
                           x2, y2,
                           width=LINE_WIDTH, fill=edge_colour)

        current_heights[v1] += 1


def show_linear_layouts(layouts):
    root = tk.Tk()
    window_width = 1400
    window_height = 650 * len(layouts)
    root.geometry(str(window_width) + 'x' + str(window_height))

    for layout in layouts:
        canvas_width = window_width
        canvas_height = window_height / len(layouts)
        canvas = tk.Canvas(root, width=canvas_width, height=canvas_height,
                           bg='white')
        canvas.pack()

        vertex_size = canvas_width / (len(layout.order) * 2 + 1)
        vertex_positions = get_linear_vertices_position(layout.order, canvas_height, vertex_size)

        for i, page in enumerate(layout.stacks):
            colours = ['orange', 'green', 'red', 'blue', 'violet', 'black']
            for edge in page:
                draw_arch(canvas, canvas_height, layout, vertex_positions, vertex_size, edge[0], edge[1], page, color=colours[i], upward=True)
        for i, page in enumerate(layout.queues):
            colours = ['blue', 'violet', 'black', 'orange', 'green', 'red']
            for edge in page:
                draw_arch(canvas, canvas_height, layout, vertex_positions, vertex_size, edge[0], edge[1], page, color=colours[i], upward=False)  #not isinstance(layout, MixedLinearLayout)

        draw_vertices(canvas, layout.graph, vertex_positions, vertex_size)

    root.mainloop()


def show_circular_layout(layout):
    root = tk.Tk()
    window_width = window_height = 900
    root.geometry(str(window_width) + 'x' + str(window_height))

    canvas_size = window_width
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='white')
    canvas.pack()

    vertex_size = 20
    center = (canvas_size / 2, canvas_size / 2)
    radius = canvas_size / 2 * 0.55
    # order is clockwise
    vertex_positions, vertex_angles = get_circular_vertices_position(layout.order, center, radius)

    stack = layout.stack
    #draw_stack_outside(stack, layout, canvas, canvas_size, center, radius, vertex_size, vertex_positions, vertex_angles)
    draw_stack_inside(stack, canvas, vertex_positions)
    #draw_stack_octilinear(stack, canvas, vertex_positions, center, radius)

    queue = layout.queue
    #draw_queue_octilinear(queue, layout, canvas, vertex_positions, center, radius)
    #draw_queue_radial(queue, layout, canvas, vertex_positions, vertex_angles, center, radius, draw_inside=True)
    draw_queue_radial_levels(queue, layout, canvas, vertex_positions, vertex_angles, draw_inside=False)



    draw_vertices(canvas, layout.graph, vertex_positions, vertex_size)
    root.mainloop()


def show_layered_graph(layered_graph):
    root = tk.Tk()
    window_width = 1400
    window_height = 800
    root.geometry(str(window_width) + 'x' + str(window_height))

    canvas_size = window_width
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='white')
    canvas.pack()

    vertex_size = 20
    vertex_positions = get_layered_vertices_position(layered_graph.layers)

    for v1, v2 in layered_graph.edges:
        x1, y1 = vertex_positions[v1]
        x2, y2 = vertex_positions[v2]
        edge = (v1, v2) if (v1, v2) in layered_graph.edges else (v2, v1)
        canvas.create_line(x1, y1, x2, y2, width=LINE_WIDTH, fill=layered_graph.edge_colors[edge])


    draw_vertices(canvas, layered_graph, vertex_positions, vertex_size)
    root.mainloop()
