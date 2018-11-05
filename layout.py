from __future__ import division
import json
from collections import namedtuple

UNIT_SIZE = 19.4 # TODO decimal
"""
x: in mm
y: in mm
angle: in radians
width: multiples of unit size
"""
Position = namedtuple('Position', ('x', 'y', 'angle', 'width'))

def read_layout(s):
    """Create a layout from kle raw data string.

    s : (str) kle layout string
    For details on the format see https://github.com/ijprest/keyboard-layout-editor/wiki/Serialized-Data-Format
    Ignores most of the kle attributes.
    s : (str) kle raw data string
    mode : (str) define how columns and rows are assigned, see LayoutMode for details.
        'default' : Assign rows and columns as they are given via the raw data string
        'minimize': Try to minimize the number of rows + the number of columns (TODO)
        'grid': Try to fit a "natural" grid to the keyboard (TODO)
    returns : (Layout)
    """
    # TODO deal with non-quoted keys
    # ignore optional metadata
    # subsequent rows increment y coordinate by 1
    # each row resets x = 0
    # After each keycap, the current x coordinate is incremented by the previous cap's width
    l = json.loads(s)
    if not isinstance(l, list):
        raise Exception("schema error") # must be a list on the outermost level
    if len(l) == 0:
        return []

    if isinstance(l[0], dict):
        # skip metadata
        l = l[1:]

    rows = []
    cur_x = 0
    cur_y = 0
    rot_x = 0
    rot_y = 0
    angle = 0

    for row in l:
        new_row = []
        i = 0
        cur_x = 0
        while i < len(row):
            width = 1
            height = 1
            x_offset = 0
            y_offset = 0
            if isinstance(row[i], dict):
                cfg = row[i]
                width = cfg.get('w', width)
                height = cfg.get('h', height)
                x_offset = cfg.get('x', x_offset)
                y_offset = cfg.get('y', y_offset)
                i += 1

            pos_x = (cur_x + x_offset + width / 2) * UNIT_SIZE
            pos_y = (cur_y + y_offset + height / 2) * UNIT_SIZE
            new_row.append(Position(pos_x, pos_y, angle, width))
            cur_x += x_offset + width
            cur_y += y_offset

            if not (isinstance(row[i], str) or isinstance(row[i], unicode)):
                raise Exception("schema error")

            i += 1
        cur_y += 1

        if not isinstance(row, list):
            raise Exception("schema error")

        rows.append(new_row)

    assert len(rows) == len(l)
    return rows
