from collections import namedtuple
from dataclasses import asdict, dataclass
import json
from typing import List

UNIT_SIZE = 19.05 # TODO decimal
"""
x: in mm
y: in mm
angle: in radians
width: multiples of unit size
"""

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

# want: json output like
# {"keys": [{row: 0, col: 0, position: {}, ], }
# make it work, make it pretty, make it fast

def create_layout(rows):
    keys = []
    for i, row in enumerate(rows):
        for j, key in enumerate(row):
            keys.append({
                "row": i,
                "col": j,
                "position": {},
                "ref": "K{},{}".format(i,j)
                })

    layout = {
            "rows": len(rows),
            "cols": max(map(len, rows)),
            "keys": keys
            }
    return layout

# FIXME: is this really necessary?
# there has to be a better way to directly map json to objects & vice versa
# I don't want to deal with dicts, but have access via regular fields
#class Layout:
#    def __init__(self, rows, cols, keys):
#        self.rows = rows
#        self.cols = cols
#        self.keys = list(map(lambda x: Position(**x), keys))
#
#    def to_json(self):
#        self_dict = self.__dict__.copy()
#
#        def position_to_json(pos):
#            return {key: getattr(pos, key) for key in POSITION_ATTRS}
#
#        self_dict['keys'] = list(map(position_to_json, self.keys))
#        return json.dumps(self_dict)
#
#def layout_from_json():
#    pass
@dataclass
class Position:
    x: float
    y: float
    angle: float
    width: float


@dataclass
class Layout:
    rows: int
    cols: int
    keys: List[Position]

    def to_json(self, **args):
        return json.dumps(asdict(self), **args)

    @classmethod
    def from_json(cls, s):
        json_dict = json.loads(s)
        json_dict['keys'] = list(map(lambda x: Position(**x), json_dict.get('keys', [])))
        return cls(**json_dict)

from sys import argv
if __name__ == '__main__':
    # TODO advanced reading!
    rows = read_layout(s)
    create_layout(rows)
