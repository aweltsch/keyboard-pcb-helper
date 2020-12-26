from collections import namedtuple
from dataclasses import asdict, dataclass
import json
import re
import sys
from typing import List
from math import sin, cos, pi, radians

UNIT_SIZE = 19.05 # TODO decimal
FIELD_PATTERN = re.compile(r'(\w+):')

@dataclass
class Position:
    """
    x: in mm
    y: in mm
    angle: in radians
    width: multiples of unit size
    """
    x: float
    y: float
    angle: float
    width: float # FIXME -> move to Key


@dataclass
class Key:
    row: int
    col: int
    position: Position

# want: json output like
# {"keys": [{row: 0, col: 0, position: {}, ], }
# make it work, make it pretty, make it fast
@dataclass
class Layout:
    rows: int
    cols: int
    keys: List[Key]

    def to_json(self, **args):
        return json.dumps(asdict(self), **args)

    @classmethod
    def from_json(cls, s, **json_args):
        json_dict = json.loads(s, **json_args)

        # FIXME a little ugly...
        # maybe we can make this more resilient / generic
        def to_key(d):
            res_dict = d.copy()
            res_dict['position'] = Position(**res_dict['position'])
            return Key(**res_dict)

        json_dict['keys'] = list(map(to_key, json_dict.get('keys', [])))
        return cls(**json_dict)

def rotate(point, angle, center=(0, 0)):
    dx, dy = (point[0] - center[0]), (point[1] - center[1])
    return center[0] + dx * cos(angle) - dy * sin(angle), center[1] + dx * sin(angle) + dy * cos(angle)

def read_layout(s: str):
    """Create a layout from kle raw data string.

    s : (str) a json string that accurately represents a kle string.
    use kle_to_json function to convert kle strings to this format!
    For details on the format see https://github.com/ijprest/keyboard-layout-editor/wiki/Serialized-Data-Format
    Ignores most of the kle attributes.
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

    """'r' can only be used on the first key in a row
    """
    for row in l:
        new_row = []
        i = 0
        cur_x = rot_x
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
                rot_x = cfg.get('rx', rot_x)
                rot_y = cfg.get('ry', rot_y)
                angle = cfg.get('r', angle)

                if 'ry' in cfg:
                    cur_y = rot_y
                if 'rx' in cfg:
                    cur_x = rot_x
                i += 1

            # use rotation center and angle to calculate x & y position
            start_x = (cur_x + x_offset + width / 2)
            start_y = (cur_y + y_offset + height / 2)

            angle_in_radians = radians(angle)

            pos_x, pos_y = rotate((start_x, start_y), angle_in_radians, (rot_x, rot_y))

            # unfortunately kle layout measure angles in the wrong direction!
            # the angles are measured clockwise vs. the standard way (TM) anti-clockwise...
            new_row.append(Position(pos_x * UNIT_SIZE, pos_y * UNIT_SIZE, -angle_in_radians, width))
            cur_x += x_offset + width
            cur_y += y_offset

            if not (isinstance(row[i], str) or isinstance(row[i], unicode)):
                raise Exception("schema error")

            i += 1
        cur_y += 1
        cur_x = rot_x

        if not isinstance(row, list):
            raise Exception("schema error")

        rows.append(new_row)

    assert len(rows) == len(l)
    return rows

# FIXME: is this still necessary? downloading from KLE website yields _valid_ json!
# copy pasting does not!
def kle_to_json(s: str):
    """Convert keyboard layout editor raw strings to valid json strings.

    KLE raw strings are _not_ valid JSON. This function adds an enclosing array
    around the row arrays from KLE. Additionally it escapes all member strings
    in javascript objects.
    """

    # NOTE: we use an extremely simple string replacement here! This might not
    # always work, but it's OK for now and should cover most common cases.
    # TODO: define more resilient transformation
    # this will lead to problems with strings like {foo: "bar:"}
    escaped_str = ""
    while re.search(FIELD_PATTERN, s):
        match = re.search(FIELD_PATTERN, s)
        escaped_str += '{}"{}":'.format(s[:match.start()], match[1])
        s = s[match.end():]

    escaped_str += s

    return "[{}]".format(escaped_str)

def json_to_layout(json_str: str):
    rows = read_layout(json_str)

    # default assignment of keys to rows / columns!
    keys = []
    n_rows = len(rows)
    n_cols = 0

    for i, row in enumerate(rows):
        if n_cols < len(row):
            n_cols = len(row)
        for j, pos in enumerate(row):
            key = Key(row=i, col=j, position=pos)
            keys.append(key)

    return Layout(rows=n_rows, cols=n_cols, keys=keys)

def kle_to_layout(s: str):
    json_str = kle_to_json(s)
    return json_to_layout(json_str)

def main():
    if len(sys.argv) < 2:
        # read from stdin
        kle_str = sys.stdin.read()
    else:
        # read from file
        f_name = sys.argv[1]
        with open(f_name) as f:
            kle_str = f.read()

    layout = json_to_layout(kle_str)
    print(layout.to_json(indent=4))

if __name__ == '__main__':
    main()
