import numpy as np
import pandas as pd
import pytest

from molscene import Scene

# --- Fixtures --------------------------------------------------------------

@pytest.fixture
def simple_scene():
    coords = np.array([[0, 0, 0], [1, 1, 1]])
    metadata = {}
    frames = None
    return Scene(coords), coords, metadata, frames

@pytest.fixture
def metadata_scene(simple_scene):
    scene, coords, _, frames = simple_scene
    scene.author = 'alice'
    scene.note   = 'test'
    metadata = {'author': 'alice', 'note': 'test'}
    return scene, coords, metadata, frames

@pytest.fixture
def multiframe_scene(simple_scene):
    scene, coords, metadata, _ = simple_scene
    frames = np.stack([coords, coords + [10,20,30]], axis=0)
    scene.set_coordinate_frames(frames.copy())
    return scene, coords, metadata, frames


# --- Helpers ---------------------------------------------------------------

def as_numpy(scene):
    return scene.get_coordinates().to_numpy()

def all_frames(scene):
    return scene.get_coordinate_frames()

def _invoke(scene, opfn, operand):
    return opfn(scene) if operand is None else opfn(scene, operand)


# --- Operation definitions -----------------------------------------------

# (opname,        lambda(scene,operand),           lambda(array,operand) )
OPS = [
    ("add",  lambda s,v: s+v,  lambda C,v: C + v),
    ("radd", lambda s,v: v+s,  lambda C,v: v + C),
    ("sub",  lambda s,v: s-v,  lambda C,v: C - v),
    ("rsub", lambda s,v: v-s,  lambda C,v: v - C),
    ("mul",  lambda s,v: s*v,  lambda C,v: C * v),
    ("rmul", lambda s,v: v*s,  lambda C,v: C * v),
    ("div",  lambda s,v: s/v,  lambda C,v: C / v),
    ("neg",  lambda s, _=None: -s, lambda C,_:   -C),
]

# (operand_name,  actual_value)
OPERANDS = {
    "list":   [1,2,3],
    "int":      2,
    "float":  2.5,
    "tuple":  (1,2,3),
    "ndarray": np.array([1,2,3]),
    "series":  pd.Series([1,2,3],index=['x','y','z']),
    "scene":  "scene",       # sentinel — we’ll replace with the scene itself
}

# Which of the above should raise on simple_scene?
ERROR_ON_SIMPLE = {"set"}


# --- Build our combinatorial cases ----------------------------------------

CASES = []
for fixture, is_multi in [
    ("simple_scene",   False),
    ("metadata_scene", False),
    ("multiframe_scene", True),
]:
    for opname, opfn, arr_fn in OPS:
        for opname2, opval in OPERANDS.items():
            # skip “neg” with an operand-type
            if opname == "neg" and opname2 != "ndarray":
                # we’ll only do neg as unary
                operand_name = None
                expected_exc = None
            else:
                operand_name = opname2
                if operand_name == "scene":
                    # only add/radd/sub/rsub with another Scene is supported
                    if opname in ("add","radd","sub","rsub"):
                        expected_exc = None
                    else:
                        expected_exc = ValueError
                elif fixture=="simple_scene" and operand_name in ERROR_ON_SIMPLE:
                    expected_exc = TypeError
                else:
                    expected_exc = None

            CASES.append({
                "id": f"{fixture}-{opname}-{operand_name}",
                "fixture": fixture,
                "opname": opname,
                "opfn": opfn,
                "operand_name": operand_name,
                "arr_fn": arr_fn,
                "is_multi": is_multi,
                "expected_exc": expected_exc
            })

# Add one explicit concat‐case for metadata_scene
CASES.append({
    "id": "metadata_scene-concat",
    "fixture": "metadata_scene",
    "opname": "concat",
    "opfn":    lambda s, _: s + s,
    "operand_name": None,
    "arr_fn":  lambda C, _: np.vstack([C, C]),
    "is_multi": False,
    "expected_exc": None
})


# --- The single, parametrized test ----------------------------------------

@pytest.mark.parametrize(
    "fixture, opname, opfn, operand_name, arr_fn, is_multi, expected_exc",
    [
        (
            c["fixture"],
            c["opname"],
            c["opfn"],
            c["operand_name"],
            c["arr_fn"],
            c["is_multi"],
            c["expected_exc"],
        ) for c in CASES
    ],
    ids=[c["id"] for c in CASES]
)
def test_all_combinations(
    fixture, opname, opfn, operand_name, arr_fn, is_multi, expected_exc, request
):
    # 1) pull out scene, coords, metadata, frames
    scene, base_coords, metadata, frames = request.getfixturevalue(fixture)

    # 2) resolve the operand
    if operand_name is None:
        operand = None
    elif operand_name == "scene":
        if opname in ("add","radd"):
            merged = scene + scene
            # for simple & metadata: coords stacked
            np.testing.assert_array_equal(as_numpy(merged),
                                          np.vstack([base_coords, base_coords]))
            # metadata preserved
            for k,v in metadata.items():
                assert getattr(merged, k) == v
            return
        if opname in ("sub","rsub"):
            diff = scene - scene
            # should remove all shared atoms → empty
            assert len(diff) == 0
            np.testing.assert_array_equal(as_numpy(diff),
                                          np.zeros((0,3), dtype=float))
            # metadata preserved
            for k,v in metadata.items():
                assert getattr(diff, k) == v
            return
        operand = scene
    else:
        operand = OPERANDS[operand_name]

    # 3) special‐case the metadata concatenation
    if fixture=="metadata_scene" and opname=="concat":
        merged = scene + scene
        # metadata preserved
        assert merged.author == metadata["author"]
        assert merged.note   == metadata["note"]
        # coords doubled
        np.testing.assert_array_equal(as_numpy(merged), arr_fn(base_coords, None))
        return

    # 4) expected exception?
    if expected_exc:
        with pytest.raises(expected_exc):
            _invoke(scene, opfn, operand)
        return

    # 5) perform the operation
    out = _invoke(scene, opfn, operand)

    # 6) metadata is always preserved
    for key, val in metadata.items():
        assert getattr(out, key) == val

    # 7) if pandas.Series, convert to numpy for expected
    if isinstance(operand, pd.Series):
        operand_val = operand.to_numpy()
    else:
        operand_val = operand

    # 8) compare coordinate arrays
    if is_multi:
        # every frame
        np.testing.assert_array_equal(all_frames(out), arr_fn(frames, operand_val))
        # active coords == first‐frame result
        np.testing.assert_array_equal(as_numpy(out), arr_fn(frames[0], operand_val))
    else:
        np.testing.assert_array_equal(as_numpy(out), arr_fn(base_coords, operand_val))
        assert len(out) == len(scene)


# --- repr test -------------------------------------------------------------

def test_repr_contains_point_count_and_axes(simple_scene):
    scene, *_ = simple_scene
    txt = repr(scene)
    assert txt.startswith("<Scene (2)>")
    for axis in ("x", "y", "z"):
        assert axis in txt
