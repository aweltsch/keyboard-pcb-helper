"""
Functions to assign keys to row / column in the grid.
"""
from layout import Key, Layout, json_to_layout
from math import ceil, floor, sqrt
import sys

def min_pin_assignment(layout):
    n_keys = len(layout.keys)

    rows = []

    n_cols = n = int(ceil(sqrt(n_keys)))
    n_rows = m = int(floor(sqrt(n_keys)))

    assert n * m >= n_keys

    keys_by_y = sorted(layout.keys, key=lambda key: key.position.y)

    for i in range(0, n_keys, n_cols):
        rows.append(keys_by_y[i:i+n_cols])

    new_keys = []
    for i, row in enumerate(rows):
        for j, key in enumerate(sorted(row, key=lambda key: key.position.x)):
            new_keys.append(Key(i, j, key.position))

    assert len(new_keys) == n_keys

    return Layout(n_rows, n_cols, new_keys)

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
    min_pin_layout = min_pin_assignment(layout)
    print(min_pin_layout.to_json(indent=4))

if __name__ == '__main__':
    main()
