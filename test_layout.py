from layout import Position, Key, Layout, read_layout, UNIT_SIZE, kle_to_json
import pytest

easy_layout = '''[
["a", "b"],
["1324", "asdfbzv"]
]
'''
easy_expected = [
        [Position((i+0.5)*UNIT_SIZE, 0.5 * UNIT_SIZE, 0, 1) for i in range(2)],
        [Position((i+0.5)*UNIT_SIZE, 1.5 * UNIT_SIZE, 0, 1) for i in range(2)]
        ]

sixty_percent_layout = '''[
["","","","","","","","","","","","","",{"w":2},""],
[{"w":1.5},"","","","","","","","","","","","","",{"w":1.5},""],
[{"w":1.75},"","","","","","","","","","","","'",{"w":2.25},""],
[{"w":2.25},"","","","","","","","","","","",{"w":2.75},""],
[{"w":1.25},"",{"w":1.25},"",{"w":1.25},"",{"a":7,"w":6.25},"",{"a":4,"w":1.25},"",{"w":1.25},"",{"w":1.25},"",{"w":1.25},""]
]
'''
sixty_expected = [
        [Position((i+0.5) * UNIT_SIZE, 0.5*UNIT_SIZE, 0, 1) for i in range(13)] \
                + [Position(14 * UNIT_SIZE, 0.5*UNIT_SIZE, 0, 2)],
        [Position(1.5 * UNIT_SIZE / 2, 1.5 * UNIT_SIZE, 0, 1.5)] \
                + [Position((1.5 + i + 0.5) * UNIT_SIZE, 1.5 * UNIT_SIZE, 0, 1) for i in range(12)] \
                + [Position((13.5 + 1.5 / 2) * UNIT_SIZE, 1.5 * UNIT_SIZE, 0, 1.5)],
        [Position(1.75 * UNIT_SIZE / 2, 2.5 * UNIT_SIZE, 0, 1.75)] + \
                [Position((1.75 + i + 0.5) * UNIT_SIZE, 2.5 * UNIT_SIZE, 0, 1) for i in range(11)] + \
                [Position((12.75 + 2.25 / 2) * UNIT_SIZE, 2.5 * UNIT_SIZE, 0, 2.25)],
        [Position(2.25 / 2 * UNIT_SIZE, 3.5 * UNIT_SIZE, 0, 2.25)] \
                + [Position((2.25 + i + 0.5) * UNIT_SIZE, 3.5 * UNIT_SIZE, 0, 1) for i in range(10)] \
                + [Position((2.25 + 10 + 2.75 / 2) * UNIT_SIZE, 3.5 * UNIT_SIZE, 0, 2.75)],
        [Position((1.25 * i + 1.25 / 2) * UNIT_SIZE, 4.5 * UNIT_SIZE, 0, 1.25) for i in range(3)] \
                + [Position((3*1.25 + 6.25 / 2) * UNIT_SIZE, 4.5 * UNIT_SIZE, 0, 6.25)] \
                + [Position((3*1.25 + 6.25 + 1.25 * i + 1.25 / 2) * UNIT_SIZE, 4.5 * UNIT_SIZE, 0, 1.25) for i in range(4)]
        ]

weird_layout = '''[
]
'''
weird_expected = [
        ]

# not yet properfly supported
iso_layout = '''
'''
iso_expected = []

def assert_same_layout(layout, expected):
    # pytest.approx(2.3)
    assert len(layout) == len(expected)
    for i, row in enumerate(layout):
        assert len(row) == len(expected[i])
        expected_row = expected[i]
        for j, col in enumerate(row):
            assert col == expected_row[j]

def test_layout():
    layout = read_layout(easy_layout)
    assert_same_layout(layout, easy_expected)
    layout = read_layout(sixty_percent_layout)
    assert_same_layout(layout, sixty_expected)
    layout = read_layout(weird_layout)
    assert_same_layout(layout, weird_expected)

def test_layout_to_json():
    key = Key(0, 0, Position(1, 1, 1, 1))
    layout = Layout(1, 1, [key])
    assert layout.to_json(sort_keys=True) \
            == '{"cols": 1, "keys": [{"col": 0, "position": {"angle": 1, "width": 1, "x": 1, "y": 1}, "row": 0}], "rows": 1}'

def test_layout_from_json():
    json_str = '{"cols": 1, "keys": [{"col": 0, "position": {"angle": 1, "width": 1, "x": 1, "y": 1}, "row": 0}], "rows": 1}'
    layout = Layout.from_json(json_str)

    expected_key = Key(0, 0, Position(1, 1, 1, 1))
    expected_layout = Layout(1, 1, [expected_key])

    assert layout == expected_layout

def test_kle_to_json():
    s = kle_to_json("{foo: 1, bar: 2}")
    assert s == '[{"foo": 1, "bar": 2}]'
    #s = kle_to_json('{foo: "bar: 2"}')
    #assert s == '[{"foo": "bar: 2"}]'
