import sys
from layout import Layout, json_to_layout
from grid_assignment import min_pin_assignment
import skidl
from skidl import Part, Net, generate_netlist
from os import environ
from utils import get_key_reference, get_diode_reference

KICAD_COMPONENTS = '/usr/share/kicad/library'
KEYBOARD_COMPONENTS = './components'
PARTS_LIBRARY = 'Keebio-Parts'

def footprint(name, library=None):
    if library is None:
        library = PARTS_LIBRARY
    return "{}:{}".format(library, name)

DIODE_COMPONENT = '1N4148'
DIODE_FOOTPRINT = footprint('Diode-dual')

KEYSWITCH_COMPONENT = '~MX'
KEYSWITCH_FOOTPRINT = footprint('MX-Alps-Choc-1U-NoLED')

MICROCONTROLLER_COMPONENT = 'ProMicro'
MICROCONTROLLER_FOOTPRINT = footprint('ArduinoProMicro')

class KeyboardSchematic:
    def __init__(self):
        pass
    pass

def get_diode():
    return Part('Diode', DIODE_COMPONENT, footprint=DIODE_FOOTPRINT) # footprint etc.

def get_switch(key_type='1u'):
    # todo -> different footprints for different key types
    return Part('keebio', KEYSWITCH_COMPONENT, footprint=KEYSWITCH_FOOTPRINT)

if 'KEYBOARD_PARTS_LIB_DIR' in environ:
    skidl.lib_search_paths[skidl.KICAD].append(environ['KEYBOARD_PARTS_LIB_DIR'])

def skidl_setup():
    skidl.lib_search_paths[skidl.KICAD].append(KICAD_COMPONENTS)
    skidl.lib_search_paths[skidl.KICAD].append(KEYBOARD_COMPONENTS)

@skidl.subcircuit
def get_key_module(row, col, row_nets, col_nets, key_type='1u'):
    # add diode to key module
    # FIXME set reference!!!
    row_net = row_nets[row]
    col_net = col_nets[col]

    switch = get_switch(key_type=key_type)
    switch.ref = get_key_reference(row, col)
    diode = get_diode()
    diode.ref = get_diode_reference(row, col)

    switch[1] += col_net
    switch[2] += diode[2]
    diode[1] += row_net

def get_micro_controller():
    return Part('keebio', MICROCONTROLLER_COMPONENT, footprint=MICROCONTROLLER_FOOTPRINT)

SPECIAL_PINS = ['RAW', 'RST', 'GND', 'VCC']

def is_usable_pin(pin):
    return pin.name not in SPECIAL_PINS

def get_usable_pins(controller):
    pins = list(filter(is_usable_pin, controller.pins))
    pins.sort(key=lambda x: x.name) # sort pins by name, so special ones come later
    return pins

def connect_microcontroller(row_nets, col_nets):
    controller = get_micro_controller()
    controller_pins = get_usable_pins(controller)
    # TODO write pin / (row&col) association to file for automatic generation of qmk config

    if len(controller_pins) < len(row_nets) + len(col_nets):
        raise Exception("Can not build keyboard schematic," \
                + "micro controller has not enough available pins." \
                + "needed: {}, available: {}".format(len(row_nets + col_nets), len(controller_pins)))

    for controller_pin, net in zip(controller_pins, row_nets + col_nets):
        net += controller_pin

# create nets for all rows & columns
def main():
    if len(sys.argv) < 2:
        # read from stdin
        layout_json = sys.stdin.read()
        netlist_file = "keyboard.net"
    else:
        # read from file
        f_name = sys.argv[1]
        # TODO evaluate if this is appropriate behaviour...
        if f_name.endswith(".json"):
            netlist_file = f_name.replace(".json", ".net")
        else:
            netlist_file = f_name + ".net"

        with open(f_name) as f:
            layout_json = f.read()

    original_layout = json_to_layout(layout_json)
    layout = min_pin_assignment(original_layout)
    skidl_setup()
    row_nets = [Net('row{}'.format(i)) for i in range(layout.rows)]
    col_nets = [Net('col{}'.format(j)) for j in range(layout.cols)]
    for key in layout.keys:
            get_key_module(key.row, key.col, row_nets, col_nets)

    connect_microcontroller(row_nets, col_nets)
    generate_netlist(file_=netlist_file)

if __name__ == '__main__':
    main()
