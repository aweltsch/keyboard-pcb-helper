import os
from pykicad.pcb import *
from pykicad.module import *
from layout import Position, Layout, json_to_layout, UNIT_SIZE
from grid_assignment import min_pin_assignment
import math
from math import sin, cos

DIODE_SPACING = UNIT_SIZE / 2

# NOTE:
# SYMLINK Keebio-Parts.pretty in current directory
DEFAULT_LIBRARY = "Keebio-Parts"
KEYSWITCH_FOOTPRINT = 'Kailh-PG1350-1u-NoLED'
MICROCONTROLLER_FOOTPRINT = 'ArduinoProMicro'
DIODE_FOOTPRINT = 'Diode'


def connect_pads_to_net(footprint, pad_name, net):
    # NOTE: pad attribute 'name' is "text", i.e. string
    for pad in footprint.pads_by_name(str(pad_name)):
        pad.net = net

def calc_diode_position(key_pos: Position):
    # pos is the position of the key switch
    x = key_pos.x + sin(key_pos.angle) * DIODE_SPACING
    y = key_pos.y + cos(key_pos.angle) * DIODE_SPACING
    angle = key_pos.angle
    width = key_pos.width
    return Position(x=x, y=y, angle=angle, width=width)

def main():
    # TODO configure module search path properly
    if len(sys.argv) < 2:
        raise Exception("give filename")
    # parse cmdline options
    f_name = sys.argv[1]

    # setup & parse input file 
    with open(f_name) as f:
        layout_json = f.read()
    original_layout = json_to_layout(layout_json)

    # assign keys to rows / columns
    layout = min_pin_assignment(original_layout)

    os.environ[MODULE_SEARCH_PATH] = ""
    # set up nets
    row_nets = [Net('row{}'.format(i)) for i in range(layout.rows)]
    col_nets = [Net('col{}'.format(j)) for j in range(layout.cols)]
    switch_diode_nets = []
    diodes = []
    switches = []

    for key in layout.keys:
        switch_diode_net = Net('switch_diode_{}_{}'.format(key.row, key.col))

        diode = Module.from_library(DEFAULT_LIBRARY, DIODE_FOOTPRINT)
        diode.set_reference("D{},{}".format(key.row, key.col))
        diode_pos = calc_diode_position(key.position)
        # FIXME somehow the angle is not applied to the pads inside the diode...
        # probably every pad needs to be rotated!
        # this is only an issue for non circular pads, of which there are many...
        diode.at = [diode_pos.x, diode_pos.y, math.degrees(diode_pos.angle)]
        connect_pads_to_net(diode, 1, row_nets[key.row])
        connect_pads_to_net(diode, 2, switch_diode_net)

        # FIXME: pykicad does not parse modules correctly
        # for the (model) subattribute, it does not support the 'offset' attribute
        # this is fixed in xesscorps fork
        keyswitch = Module.from_library(DEFAULT_LIBRARY, KEYSWITCH_FOOTPRINT)
        keyswitch.set_reference("K{},{}".format(key.row, key.col))
        keyswitch.at = [key.position.x, key.position.y, math.degrees(key.position.angle)]
        connect_pads_to_net(keyswitch, 1, col_nets[key.col])
        connect_pads_to_net(keyswitch, 2, switch_diode_net)

        switch_diode_nets.append(switch_diode_net)
        diodes.append(diode)
        switches.append(keyswitch)

    all_nets = [Net('GND')] + row_nets + col_nets + switch_diode_nets

    netclass = NetClass('default', nets=all_nets)
    pcb = Pcb()
    pcb.title = 'keyboard'
    pcb.num_nets = len(all_nets)
    pcb.nets += all_nets
    pcb.modules += diodes + switches

    pcb.to_file(f_name.replace(".json", ""))


if __name__ == '__main__':
    main()
