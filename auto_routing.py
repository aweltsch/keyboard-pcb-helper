import networkx as nx
from kigadgets.board import Board
from kigadgets.pad import PadShape, PadType
import pcbnew
import math
from dataclasses import dataclass, field
from typing import Set

#DEFAULT_GRID = 0.5 # one point every 0.5mm
DEFAULT_GRID = 0.5
MINIMUM_CLEARANCE = 0.3
COPPER_LAYERS = ['F.Cu', 'B.Cu']
VIA_DIAMETER = 0.8
VIA_WEIGHT = 20

def dist(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

# TODO implement simple autorouting from the PCB + ratsnest
def get_grid_graph(width_mm, height_mm, discretization=DEFAULT_GRID):
    n = int(width_mm / discretization)
    m = int(height_mm / discretization)
    x = nx.Graph()
    # offsets = [(k, l) for k in [-1, 0, 1] for l in [-1, 0, 1]]
    offsets = [(-1, 1), (0, 1), (1, 1), (1, 0), (0, 0)]
    # offsets = [-1, 1]
    for i in range(n):
        for j in range(m):
            start = (i, j)
            edges = []
            for (k, l) in offsets:
                end = (i + k, j + l)
                if i + k < 0 or i + k >= n or j + l < 0 or j + l >= m:
                    continue
                if start == end:
                    edges.append((start + (COPPER_LAYERS[0],), end + (COPPER_LAYERS[1],), VIA_WEIGHT))
                else:
                    edges.extend((start + (layer,), end + (layer,), coord_diff(start, end)) for layer in COPPER_LAYERS)

                # ((i, j, layer), (i + k, j + l, layer), 1) for (k, l) in offsets if 0 <= (i + k) < n and 0 <= (j + l) < m for layer in COPPER_LAYERS]
            # edges = [((i, j, layer), (i + k, j, layer), 1) for k in offsets for layer in COPPER_LAYERS if 0 <= i + k < n]
            # edges.extend([((i, j, layer), (i, j + k, layer), 1) for k in offsets for layer in COPPER_LAYERS if 0 <= j + k < m])
            x.add_weighted_edges_from(edges)

    # verify that internal connections are set up correctly
    for node in x.nodes:
        neighbors = list(x.neighbors(node))
        if 1 <= node[0] < n - 1 and 1 <= node[1] < m - 1:
            assert len(neighbors) == 9, (node, neighbors)
        for other in neighbors:
            assert int(dist(node, other)) in [0, 1, 2], (node, other, dist(node, other))
    return x

# TODO rigorous tests
# FIXME is the minimum clearance good here?
def calculate_circle_keep_out(pad, radius):
    assert pad.size.x == pad.size.y
    min_x, max_x = int((pad.position.x - radius) / DEFAULT_GRID), math.ceil((pad.position.x + radius) / DEFAULT_GRID)
    min_y, max_y = int((pad.position.y - radius) / DEFAULT_GRID), math.ceil((pad.position.y + radius) / DEFAULT_GRID)

    keep_outs = []
    pad_pos = (pad.position.x, pad.position.y)
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            pos = (DEFAULT_GRID * i, DEFAULT_GRID * j)
            if dist(pos, pad_pos) <= radius:
                keep_outs.append((i,j))

    return keep_outs

def calculate_rectangle_keep_out(pad, rotation_deg):
    # FIXME: ignore rotation for now and work with circumcircle of rectangle
    radius = (pad.size.x**2 + pad.size.y**2)**0.5 / 2 + MINIMUM_CLEARANCE
    return calculate_circle_keep_out(pad, radius)

def apply_keep_out(graph, pad, rotation_deg):
    # how does clearance work here?
    shape = pad.shape
    if shape == PadShape.Circle:
        keep_outs = calculate_circle_keep_out(pad, pad.size.x / 2 + MINIMUM_CLEARANCE)
    elif shape in [PadShape.Rectangle, PadShape.RoundedRectangle]:
        keep_outs = calculate_rectangle_keep_out(pad, rotation_deg)
    else:
        raise Exception("not implemented for anything other than Circle and Rectangle")
    pad_type = pad.pad_type

    if len(keep_outs) == 0:
        print("unexpectedly no keep outs for pad")
        return

    # contract all pads into single node
    # if it is not connected to a net,
    if pad_type == PadType.Through:
        pad_node = (pad.position.x / DEFAULT_GRID, pad.position.y / DEFAULT_GRID)
        graph.add_node(pad_node)
        identified_nodes = []
        for layer in COPPER_LAYERS:
            for k in keep_outs:
                node = k + (layer, )
                nx.identified_nodes(graph, pad_node, node, self_loops=False, copy=False)
                assert pad_node in graph.nodes
                identified_nodes.append(node)
        if not pad.net_name:
            graph.remove_node(pad_node)
        else:
            return (pad_node, identified_nodes)
    elif pad_type == PadType.NPTH:
        # NONE-PLATED through hole, remove from graph
        for p in keep_outs:
            for layer in COPPER_LAYERS:
                graph.remove_node(p + (layer,))
        assert not pad.net_name, f"NPTH with net_name {pad.net_name}"
    else:
        raise Exception("not implemented for anything other than Through Hole and Npth pads")

def is_via_edge(e):
    return len(e[0]) == 3 and len(e[1]) == 3 and e[0][:2] == e[1][:2]

def is_track_edge(e):
    return len(e[0]) == 2 or len(e[1]) == 2 or e[0][2] == e[1][2]

def get_keep_out(pad):
    radius = pad.size.x
    return radius + MINIMUM_CLEARANCE

def remove_keep_outs(graph, board):
    # probably I need a planning instance
    # iterate through footprints and create appropriate keep outs for each foot print
    # for now I think we mostly need to consider (pad thru_hole as keep out)
    # make sure shortest paths for one netlist do not use shortest paths for other netlists
    # maybe we need custom shortest paths
    # potentially assign "net" parameter to nodes
    # we can only use an edge if the node is not part of any net or if the node is part of the same net
    # on iteration we remove "used nodes from the graph, will probably achieve the same
    # TODO involve pad clearance
    pad_nodes = {}
    for fp in board.footprints:
        for p in fp.pads:
            identified_nodes = apply_keep_out(graph, p, fp.orientation)
            if identified_nodes:
                pad_nodes[identified_nodes[0]] = identified_nodes[1]
    for keepout in board.keepouts:
        raise Exception("keep out not implemented")

    assert all(p in g.nodes for p in pad_nodes)
    return pad_nodes

def route_subnet(g, subnet_name, nets):
    cur_net = nets[subnet_name]
    # quite slow
    tree = nx.algorithms.approximation.steiner_tree(g, cur_net)
    return tree

def validate_graph(g):
    for e in g.edges:
        # if e[0][-1] == e[1][-1]:
            # assert weight == VIA_WEIGHT
        # else:
            # assert weight == 1
        assert (e[0][:2] == e[1][:2]) or (e[0][-1] == e[1][-1]), e

def get_subnets(board):
    nets = {}
    for fp in board.footprints:
        for p in fp.pads:
            if p.net_name:
                cur_net = nets.get(p.net_name, [])
                cur_net.append((p.position.x / DEFAULT_GRID, p.position.y / DEFAULT_GRID))
                nets[p.net_name] = cur_net
    return nets
def coord_diff(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_diagonal_edges(graph, edge, pad_nodes, node_to_pad):
    a, b = edge
    if a[0] > b[0]:
        a, b = b, a
    forbidden_edges = set()
    if len(a) == 2 or len(b) == 2:
        if len(a) < len(b):
            a, b = b, a
        assert len(a) == 3, f"can not have edge between pads {a}, {b}" # TODO this is a severe restriction possibly
        assert len(b) == 2 and b in pad_nodes

        a_orig = a
        for other in filter(lambda x: coord_diff(a_orig, x) == 2 and a[2] == x[2], pad_nodes[b]):
            a = a_orig
            if a[0] > other[0]:
                a, other = other, a
            # TODO: this thing here needs some thorough unit testing!!!
            if other[0] - a[0] == 1:
                assert a[0] < other[0]
                assert coord_diff(a, other) == 2, (a, other)
                anti_start = (a[0], other[1], a[2])
                anti_end = (other[0], a[1], other[2])
                if anti_start in node_to_pad:
                    anti_start = node_to_pad[anti_start]
                if anti_end in node_to_pad:
                    anti_end = node_to_pad[anti_end]
                anti_edge = (anti_start, anti_end)
                forbidden_edges.add(anti_edge)
    elif a[:2] != b[:2]:
        assert a[-1] == b[-1]
        assert a[0] != b[0]
        assert a[0] < b[0]
        anti_start = (a[0], b[1], a[2])
        anti_end = (b[0], a[1], b[2])
        anti_edge = (anti_start, anti_end)
        # assert anti_start not in node_to_pad
        # assert anti_end not in node_to_pad
        if anti_start in node_to_pad:
            anti_edge = (node_to_pad[anti_start], anti_end)
        elif anti_end in node_to_pad:
            anti_edge = (anti_start, node_to_pad[anti_end])
        forbidden_edges.add(anti_edge)

    return forbidden_edges


def is_pad_connecting_edge(e):
    return len(e[0]) == 2 or len(e[1]) == 2

def is_proper_grid_edge(e):
    return len(e[0]) == 3 and len(e[1]) == 3

def is_grid_node(n):
    return len(n) == 3 and n[2] in COPPER_LAYERS

def is_crossing(e, f):
    return False

def get_layer(e):
    src, dest = e
    assert len(src) > 2 or len(dest) > 2, "pad connection edge {e}"
    assert not is_via_edge(e)
    if len(dest) > len(src):
        src, dest = dest, src
    assert len(src) == 3
    return src[2]


def edges_are_touching(e, f):
    if is_via_edge(e) or is_via_edge(f):
        return True
    else: # only track edges
        same_layer = get_layer(e) == get_layer(f)
        return same_layer and is_crossing(e, f)

def remove_invalid_nodes_and_edges(graph, route):
    edges_to_remove = set()
    for edge in route:
        src, dest = edge
        neighbors = list(graph.neighbors(src)) + list(graph.neighbors(dest))
        for other_edge in graph.edges(neighbors):
            if edges_are_touching(edge, other_edge):
                edges_to_remove.add(other_edge)
    graph.remove_edges_from(edges_to_remove)

def route_with_a_star(graph, nets, pad_nodes):
    node_to_pad = {}
    for pad, nodes in pad_nodes.items():
        for node in nodes:
            node_to_pad[node] = pad

    orig_neighbors = {}
    for net in sorted(nets):
        for p in nets[net]:
            assert p not in orig_neighbors, f"pad in multiple different subnets ({net}: {p})"
            orig_neighbors[p] = list(graph.neighbors(p))
    for p in orig_neighbors:
        graph.remove_node(p)

    net_routes = []
    invalid_edges = set()
    for net in sorted(nets):
        paths = []
        # we need to make sure that we don't use any "pads" from the other subnets in the tree!
        for p in nets[net]:
            # TODO when re-adding edges we need to watch out
            for n in orig_neighbors[p]:
                if not n in graph.nodes:
                    continue
                assert p != n
                edge = (p, n)
                alt_edge = (n, p)
                if edge not in invalid_edges and alt_edge not in invalid_edges:
                    # TODO -> this needs to be smarter
                    # no edge should be added that has been forbidden
                    # what makes it harder
                    graph.add_weighted_edges_from([edge + (1,)])
        print("routing net", net)
        stack = sorted(nets[net])
        conn_nodes = set()
        cur = stack.pop()
        while stack:
            stack.sort(key = lambda x: dist(cur, x), reverse=True)
            target = stack.pop()
            path = nx.astar_path(graph, cur, target, heuristic=dist)
            paths.append(path)
            cur = target


        nodes = set()
        total_path = []
        for path in paths:
            for a, b in zip(path, path[1:]):
                total_path.append((a, b))
                nodes.add(a)
                nodes.add(b)

        forbidden_edges = set(total_path)
        for edge in total_path:
            (a, b) = edge

            if a[0] > b[0]:
                a, b = b, a
            diff = abs(a[0] - b[0]) + abs(a[1] - b[1])
            assert len(a) == 2 or len(b) == 2 or diff <= 2, edge
            if (a, b) in g.edges:
                assert (b, a) in g.edges
            forbidden_edges.add((a, b))
            if diff == 1:
                assert len(a) == 3 and len(b) == 3
                continue
            forbidden_edges.update(get_diagonal_edges(graph, edge, pad_nodes, node_to_pad))

        invalid_edges.update(forbidden_edges)
        remove_invalid_nodes_and_edges(graph, total_path)
        graph.remove_edges_from(forbidden_edges)
        graph.remove_nodes_from(nodes)

        net_routes.append(total_path)
    return net_routes

def plot_graph(g):
    import matplotlib.pyplot as plt
    x, y = [], []
    for node in g.nodes:
        x.append(node[0])
        y.append(-node[1]) # in kicad y > 0 means going down
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y)
    plt.show()

def to_coords(a):
    return (a[0] * DEFAULT_GRID, a[1] * DEFAULT_GRID)

def add_routes_to_board(board, net_routes):
    for route in net_routes:
        for (src, dest) in route:
            if src[:2] == dest[:2]:
                # add via
                assert src[-1] != dest[-1]
                board.add_via(to_coords(src[:2]), layer_pair=COPPER_LAYERS)
            else:
                if len(src) == 2:
                    assert len(dest) == 3
                    layer = dest[2]
                elif len(dest) == 2:
                    assert len(src) == 3
                    layer = src[2]
                else:
                    assert src[-1] == dest[-1]
                    layer = src[-1]
                board.add_track([to_coords(src[:2]), to_coords(dest[:2])], layer=layer)
    return board

@dataclass(frozen=True)
class GridNode:
    x: int
    y: int
    layer: str

@dataclass(frozen=True)
class PadNode:
    id: str
    x: float = field(hash=False)
    y: float = field(hash=False)
    grid_nodes: Set[GridNode] = field(hash=False)

@dataclass(frozen=True)
class TrackConnection:
    src: GridNode | PadNode
    dest: GridNode | PadNode
    weight: float

@dataclass(frozen=True)
class ViaConnection:
    src: GridNode | PadNode
    dest: GridNode | PadNode
    weight: float

assert GridNode(3, 2, 'F.Cu') == GridNode(3, 2, 'F.Cu')
assert hash(GridNode(3, 2, 'F.Cu')) == hash(GridNode(3, 2, 'F.Cu'))

# TODO implement simple autorouting from the PCB + ratsnest
def get_grid_graph(width_mm, height_mm, discretization=DEFAULT_GRID):
    n = int(width_mm / discretization)
    m = int(height_mm / discretization)
    grid = nx.Graph()
    offsets = [(-1, 1), (0, 1), (1, 1), (1, 0), (0, 0)]
    for i in range(n):
        for j in range(m):
            start = (i, j)
            edges = []
            for (k, l) in offsets:
                end = (i + k, j + l)
                if i + k < 0 or i + k >= n or j + l < 0 or j + l >= m:
                    continue
                if start == end:
                    edges.append(((i, j, COPPER_LAYERS[0]), (i + k, j + l, COPPER_LAYERS[1]), VIA_WEIGHT))
                else:
                    edges.extend((start + (layer,), end + (layer,), coord_diff(start, end)) for layer in COPPER_LAYERS)

                # ((i, j, layer), (i + k, j + l, layer), 1) for (k, l) in offsets if 0 <= (i + k) < n and 0 <= (j + l) < m for layer in COPPER_LAYERS]
            # edges = [((i, j, layer), (i + k, j, layer), 1) for k in offsets for layer in COPPER_LAYERS if 0 <= i + k < n]
            # edges.extend([((i, j, layer), (i, j + k, layer), 1) for k in offsets for layer in COPPER_LAYERS if 0 <= j + k < m])
            grid.add_weighted_edges_from(edges)

    # verify that internal connections are set up correctly
    for node in grid.nodes:
        neighbors = list(grid.neighbors(node))
        if 1 <= node[0] < n - 1 and 1 <= node[1] < m - 1:
            assert len(neighbors) == 9, (node, neighbors)
        for other in neighbors:
            assert int(dist(node, other)) in [0, 1, 2], (node, other, dist(node, other))
    return grid

def create_grid_graph():
    pass

class AutoRouter():
    def __init__(self, board_name):
        self.board_name = board_name
        self.output_name = f"routed-{board_name}"
        self.board = Board(pcbnew.LoadBoard(board_name))
        self.x_mm = 260 # TODO
        self.y_mm = 120 # TODO
        self.graph = create_grid_graph(x_mm, y_mm)
        self.pad_nodes = remove_keep_out(self.graph, self.board)
        self.nets = get_subnets(self.board)

    def run_auto_routing(self):
        net_routes = self.route_with_a_star()
        self.add_routes_to_board(net_routes)

    def add_routes_to_board(self):
        pass

    def save(self):
        self.board.save(output_name)

# clearer structure
# there are three types of edges
# front tracks, between grid points on the front layer 'F.Cu'
# back tracks, between grid points on the back layer 'B.Cu'
# vias, connecting the same grid point with a via

if __name__ == '__main__':
    from sys import argv
    if len(argv) != 2:
        raise Exception(f"{argv[0]} <pcb_file>")
    g = get_grid_graph(260, 120)
    validate_graph(g)
    board_name = argv[1]
    board = Board(pcbnew.LoadBoard(board_name))
    # board.add_track(coords, layer='F.Cu', width=None)
    # board.add_via(coord, layer_pair=['F.Cu', 'B.Cu'], width=None)
    pad_nodes = remove_keep_outs(g, board)
    for p in pad_nodes:
        assert p in g.nodes
        print(g.edges(p))
    nets = get_subnets(board)
    net_routes = route_with_a_star(g, nets, pad_nodes)
    add_routes_to_board(board, net_routes)
    board.save(f"routed-{board_name}")
