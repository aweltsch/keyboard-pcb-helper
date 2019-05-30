import pcbnew
import sys
from math import sin, cos, pi
from layout import Position, Layout, Key, UNIT_SIZE
from utils import get_key_reference, get_diode_reference

DIODE_SPACING = UNIT_SIZE / 2
SCALE = 1000000.0
ANGLE_SCALE = 10

def main():
    if len(sys.argv) < 2:
        print("usage: python {} file_name <json_layout>".format(sys.argv[0]))
        sys.exit(1)
    
    f_name = sys.argv[1]
    print(sys.argv)
    if len(sys.argv) == 2:
        json_str = sys.stdin.read()
    else:
        with open(sys.argv[2]) as f:
            json_str = f.read()

    pcb = pcbnew.LoadBoard(f_name)
    pcb.BuildListOfNets()

    layout = Layout.from_json(json_str)
    for key in layout.keys:
        place_key(pcb, key)
        place_diode(pcb, key)

    pcb.Save("out.kicad_pcb")

def to_pcbnew_position(pos: Position):
    return pcbnew.wxPoint(pos.x * SCALE, pos.y * SCALE)

def to_pcbnew_angle(angle: float):
    # angle in radians
    return ANGLE_SCALE * angle / (2 * pi) * 360

def calc_diode_position(key_pos: Position):
    # pos is the position of the key switch
    x = key_pos.x + sin(key_pos.angle) * DIODE_SPACING
    y = key_pos.y + cos(key_pos.angle) * DIODE_SPACING
    angle = key_pos.angle
    width = key_pos.width
    return Position(x=x, y=y, angle=angle, width=width)

# new PCB: pcbnew.BOARD()
def place_component(pcb: pcbnew.BOARD, ref: str, pos: Position):
    module = pcb.FindModuleByReference(ref)
    module.SetPosition(to_pcbnew_position(pos))
    module.SetOrientation(to_pcbnew_angle(pos.angle))

def place_key(pcb: pcbnew.BOARD, key: Key):
    place_component(pcb, get_key_reference(key.row, key.col), key.position)

def place_diode(pcb: pcbnew.BOARD, key: Key):
    place_component(pcb,
            get_diode_reference(key.row, key.col),
            calc_diode_position(key.position))

if __name__ == '__main__':
    main()
