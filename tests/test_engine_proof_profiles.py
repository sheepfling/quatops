"""Check that each pinned native fixture declares a comparable proof profile."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PROOF_KEYS = {
    "euler_to_quaternion",
    "euler_readback_and_threshold_neighbors",
    "matrix_to_quaternion",
    "axis_angle",
    "rotation_vector",
    "slerp_shortest",
    "slerp_full_path",
    "normalization_edges",
    "basis_vector_action",
    "composition_and_frame_map",
}
BLENDER_PROOF_KEYS = PROOF_KEYS | {
    "euler_continuity",
    "rotation_difference",
    "scene_object_transform",
    "swing_twist",
}
ENGINES = (
    ("unreal", "5.7.4", "Unreal Engine", "5.7.4"),
    ("unity", "6000.5.2f1", "Unity Editor", "6000.5.2f1"),
    ("godot", "4.7-stable", "Godot Engine", "4.7-stable"),
    ("godot", "4.8-dev1", "Godot Engine", "4.8-dev1"),
    ("blender", "5.1.2", "Blender", "5.1.2"),
)


def _fixture(engine, version):
    path = ROOT / "tests" / "golden" / engine / version / "native_rotation.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _all_versioned_fixture_paths():
    golden_root = ROOT / "tests" / "golden"
    return sorted(
        path
        for engine in ("unreal", "unity", "godot", "blender")
        for path in (golden_root / engine).glob("*/native_rotation.json")
    )


@pytest.mark.parametrize("path", _all_versioned_fixture_paths(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_versioned_fixture_uses_the_shared_proof_profile(path):
    _engine, version = path.parts[-3:-1]
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert fixture["evidence_kind"].startswith("native_")
    assert fixture["engine"]["version"].startswith(version)
    expected_keys = BLENDER_PROOF_KEYS if _engine == "blender" else PROOF_KEYS
    assert set(fixture["proof_profile"]) == expected_keys
    assert all(isinstance(value, str) and value.strip() for value in fixture["proof_profile"].values())
    assert fixture["quaternion_algebra"] == "Hamilton"
    if _engine == "blender":
        assert fixture["quaternion_order"].startswith("mathutils")
    else:
        assert fixture["quaternion_order"].startswith("xyzw")


@pytest.mark.parametrize("engine,version,name,version_prefix", ENGINES)
def test_native_fixture_is_stored_under_its_engine_version(engine, version, name, version_prefix):
    path, fixture = _fixture(engine, version)
    assert path.parts[-3:-1] == (engine, version)
    assert fixture["evidence_kind"].startswith("native_")
    assert fixture["engine"]["name"] == name
    assert fixture["engine"]["version"].startswith(version_prefix)
    assert fixture["quaternion_algebra"] == "Hamilton"
    if engine == "blender":
        assert fixture["quaternion_order"].startswith("mathutils")
    else:
        assert fixture["quaternion_order"].startswith("xyzw")


@pytest.mark.parametrize("engine,version,name,version_prefix", ENGINES)
def test_native_fixtures_have_matching_conversion_proof_categories(engine, version, name, version_prefix):
    _, fixture = _fixture(engine, version)
    profile = fixture["proof_profile"]
    assert set(profile) == (BLENDER_PROOF_KEYS if engine == "blender" else PROOF_KEYS)
    assert all(isinstance(route, str) and route.strip() for route in profile.values())
    assert len(fixture["cases"]) == 15
    if engine == "unreal":
        assert len(fixture["native_semantics"]["cross_products_float32"]) == 3
        assert len(fixture["native_semantics"]["cross_products_float64"]) == 3
    elif engine == "blender":
        assert len(fixture["cross_products"]) == 3
    else:
        assert len(fixture["cross_products"]) == 3

    if engine == "unreal":
        extended = fixture["extended_conversions"]
        for precision in ("float32", "float64"):
            assert len(extended[f"matrix_to_quaternion_{precision}"]) >= 9
            assert len(extended[f"axis_angle_{precision}"]) == 5
            assert len(extended[f"rotation_vector_{precision}"]) == 5
            assert len(extended[f"slerp_{precision}"]) == 22
            assert len(extended[f"normalization_{precision}"]) == 5
        assert "SlerpFullPath" in profile["slerp_full_path"]
    elif engine == "blender":
        extended = fixture["extended_conversions"]
        assert len(extended["euler_orders"]) == 6 * 87
        assert {row["order"] for row in extended["euler_orders"]} == {
            "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"
        }
        assert len(extended["matrix_to_quaternion"]) == 8
        assert len(extended["axis_angle"]) == 12
        assert len(extended["rotation_vector"]) == 6
        assert len(extended["slerp"]) == 25
        assert len(extended["normalization"]) == 5
        assert len(extended["rotation_difference"]) == 2
        assert len(extended["swing_twist"]) == 9
        assert len(extended["euler_compat_trajectories"]) == 18
        assert "not exposed" in profile["slerp_full_path"]
    else:
        extended = fixture["extended_conversions"]
        assert len(extended["matrix_to_quaternion"]) == (10 if engine == "godot" else 9)
        assert len(extended["axis_angle"]) == 5
        assert len(extended["rotation_vector"]) == 5
        assert len(extended["slerp"]) == 22
        assert len(extended["normalization"]) == 5
        if engine == "godot":
            assert "slerpni" in profile["slerp_full_path"]
            assert len(extended["euler_orders"]) == 67 * 6
            assert {row["order"] for row in extended["euler_orders"]} == {
                "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"
            }
            assert len(extended["from_to"]) == 9
            diagnostics = extended["matrix_conversion_diagnostics"]
            assert len(diagnostics) == 3
            assert all(case["diagnostic_only"] for case in diagnostics)
            assert len(extended["rotation_exp_log"]["exp"]) == 6
            assert len(extended["rotation_exp_log"]["log"]) == 5
        else:
            assert "not exposed" in profile["slerp_full_path"]
            assert "derived" in profile["rotation_vector"]
            euler_orders = extended["unity_mathematics_euler_orders"]
            assert len(euler_orders) == 37 * 6
            assert {row["order"] for row in euler_orders} == {
                "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"
            }
            assert len(extended["lerp"]) == 18
            assert "math.nlerp" in extended["capabilities"]["lerp_nlerp"]


def test_native_fixtures_keep_the_declared_source_frame_explicit():
    for engine, version, _, _ in ENGINES:
        _, fixture = _fixture(engine, version)
        if engine == "unreal":
            source_frame = fixture["native_semantics"]["source_frame"]
            assert source_frame["handedness"] == "right-handed"
            assert source_frame["positive_x"] == "right"
            assert source_frame["positive_y"] == "forward"
            assert source_frame["positive_z"] == "up"
        else:
            assert fixture["source_frame"]["handedness"] == "right-handed"
            assert fixture["source_frame"]["positive_x"] == "right"
            assert fixture["source_frame"]["positive_y"] == "forward"
            assert fixture["source_frame"]["positive_z"] == "up"
