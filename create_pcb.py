import os
import pcbnew
from pykicad.pcb import *
from pykicad.module import *
from layout import Position, Layout, json_to_layout, UNIT_SIZE
from grid_assignment import min_pin_assignment
import math
from math import sin, cos
from place_components import place_key, place_diode, place_component

DIODE_SPACING = UNIT_SIZE / 2

# NOTE:
# SYMLINK Keebio-Parts.pretty in current directory
DEFAULT_LIBRARY = "Keebio-Parts"
KEYSWITCH_FOOTPRINT = 'Kailh-PG1350-1u-NoLED'
MICROCONTROLLER_FOOTPRINT = 'ArduinoProMicro'
DIODE_FOOTPRINT = 'Diode'

# TODO think of a better way to do this... unusable (3 GND, 4 GND, 21 VCC, 22 RST, 23 GND, 24 RAW)
SPECIAL_PRO_MICRO_PINS = {
        'GND': [3, 4, 23],
        'RST': [22],
        'VCC': [21],
        'RAW': [24]
        }
USABLE_PRO_MICRO_PINS = set(range(1, 25)) - set([pin for sublist in SPECIAL_PRO_MICRO_PINS.values() for pin in sublist])

def connect_pads_to_net(footprint, pad_name, net):
    # NOTE: pad attribute 'name' is "text", i.e. string
    for pad in footprint.pads_by_name(str(pad_name)):
        pad.net = net

def place_pcb_components(kicad_pcb: pcbnew.BOARD, layout: Layout):
    for key in layout.keys:
        place_key(kicad_pcb, key)
        place_diode(kicad_pcb, key)

def get_usable_pins(microcontroller):
    return USABLE_PRO_MICRO_PINS


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
        connect_pads_to_net(diode, 1, row_nets[key.row])
        connect_pads_to_net(diode, 2, switch_diode_net)

        keyswitch = Module.from_library(DEFAULT_LIBRARY, KEYSWITCH_FOOTPRINT)
        keyswitch.set_reference("K{},{}".format(key.row, key.col))
        connect_pads_to_net(keyswitch, 1, col_nets[key.col])
        connect_pads_to_net(keyswitch, 2, switch_diode_net)

        switch_diode_nets.append(switch_diode_net)
        diodes.append(diode)
        switches.append(keyswitch)

    # connect microcontroller
    microcontroller = Module.from_library(DEFAULT_LIBRARY, MICROCONTROLLER_FOOTPRINT)
    usable_pins = get_usable_pins(microcontroller)
    if len(usable_pins) < len(row_nets + col_nets):
        raise Exception("Not enough pins on microcontroller")
    for pin, net in zip(usable_pins, row_nets + col_nets):
        connect_pads_to_net(microcontroller, pin, net)


    all_nets = [Net('GND')] + row_nets + col_nets + switch_diode_nets

    netclass = NetClass('default', nets=all_nets)
    pcb = Pcb()
    pcb.title = 'keyboard'
    pcb.num_nets = len(all_nets)
    pcb.nets += all_nets
    pcb.modules += diodes + switches + [microcontroller]

    pcb.to_file(f_name.replace(".json", ""))

    # use pcbnew to place the components, it has proper handling of rotation etc...
    # TODO evaluate usage of pcbnew here
    # we can properly encapsulate this!
    kicad_pcb_name = f_name.replace(".json", ".kicad_pcb")
    kicad_pcb = pcbnew.LoadBoard(kicad_pcb_name)
    place_pcb_components(kicad_pcb, layout)
    kicad_pcb.Save(kicad_pcb_name)


if __name__ == '__main__':
    main()
