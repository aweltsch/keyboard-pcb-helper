import skidl
from skidl import Part, Net, generate_netlist
from os import environ

KICAD_COMPONENTS = '/usr/share/kicad/library'
KEYBOARD_COMPONENTS = './components'
PARTS_LIBRARY = 'Keebio-Parts'

def footprint(name, library=None):
    if library is None:
        library = PARTS_LIBRARY
    return "{}:{}".format(library, name)

DIODE_COMPONENT = 'D'
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
    return Part('Device', DIODE_COMPONENT, footprint=DIODE_FOOTPRINT) # footprint etc.

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
    switch.ref = 'K{},{}'.format(i, j)
    diode = get_diode()
    diode.ref = 'D{},{}'.format(i, j)

    switch[1] += row_net
    switch[2] += diode[1]
    diode[2] += col_net

# create nets for all rows & columns
def main():
    config = parse_args()

N_ROWS = 10
N_COLS = 4
if __name__ == '__main__':
    skidl_setup()
    row_nets = [Net('row{}'.format(i)) for i in range(N_ROWS)]
    col_nets = [Net('col{}'.format(j)) for j in range(N_COLS)]
    for i in range(N_ROWS):
        for j in range(N_COLS):
            get_key_module(i, j, row_nets, col_nets)
    generate_netlist()
