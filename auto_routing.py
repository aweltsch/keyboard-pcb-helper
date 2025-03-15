import networkx
from kigadgets.board import Board
from kigadgets.pad import PadShape, PadType
import pcbnew
import math

DEFAULT_GRID = 0.5 # one point every 0.5mm
MINIMUM_CLEARANCE = 0.2

def dist(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

# TODO implement simple autorouting from the PCB + ratsnest
def get_grid_graph(width_mm, height_mm, discretization=DEFAULT_GRID):
    n = int(width_mm / discretization)
    m = int(height_mm / discretization)
    x = networkx.Graph()
    offsets = [(k, l) for k in [-1, 0, 1] for l in [-1, 0, 1] if k != 0 or l != 0]
    for i in range(n):
        for j in range(m):
            node = (i, j)
            edges = [(node, (i + k, j + l)) for (k, l) in offsets if 0 <= (i + k) < n and 0 <= (j + l) < m]
            x.add_edges_from(edges)
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

def calculate_keep_out(pad, rotation_deg):
    # how does clearance work here?
    shape = pad.shape
    if shape == PadShape.Circle:
        keep_outs = calculate_circle_keep_out(pad, pad.size.x / 2 + MINIMUM_CLEARANCE)
    elif shape in [PadShape.Rectangle, PadShape.RoundedRectangle]:
        keep_outs = calculate_rectangle_keep_out(pad, rotation_deg)
    else:
        raise Exception("not implemented for anything other than Circle and Rectangle")
    pad_type = pad.pad_type

    # contract all pads into single node
    # if it is not connected to a net,
    if pad_type == PadType.Through:
        pass
    elif pad_type == PadType.NPTH:
        # NONE-PLATED through hole, remove from graph
        pass
    else:
        raise Exception("not implemented for anything other than Through Hole and Npth pads")
    if not pad.net_name:
        # remove node
        pass


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
    for fp in board.footprints:
        print(fp.fp_name)
        for p in fp.pads:
            calculate_keep_out(p, fp.orientation)
    for keepout in board.keepouts:
        raise Exception("keep out not implemented")

def route_subnet(subnet_name):
    find_pads(subnet_name, board)

def remove_used_edges():
    pass

if __name__ == '__main__':
    g = get_grid_graph(260, 120)
    board_name = "keyboard-layout.kicad_pcb"
    board = Board(pcbnew.LoadBoard(board_name))
    # board.add_track(coords, layer='F.Cu', width=None)
    # board.add_via(coord, layer_pair=['F.Cu', 'B.Cu'], width=None)
    remove_keep_outs(g, board)
