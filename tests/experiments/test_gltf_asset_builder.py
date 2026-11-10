"""SDK-free tests for the deterministic glTF input, not native Blender tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.support.paths import REPOSITORY_ROOT
from tests.support.provenance_archives import read_provenance_source

BUILDER = REPOSITORY_ROOT / "experiments/transports/gltf/2.0.0/assets/build_asset.py"
SPEC = importlib.util.spec_from_file_location("gltf_asset_builder", BUILDER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load asset builder: {BUILDER}")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class GltfAssetBuilderTests(unittest.TestCase):
    def test_capture_source_identities_and_archives_are_pinned(self) -> None:
        blender = json.loads(
            (
                REPOSITORY_ROOT / "tests/oracles/blender/5.1.2/reproduction.json"
            ).read_text()
        )
        blender_contract = json.loads(
            (REPOSITORY_ROOT / "tests/oracles/blender/5.1.2/contract.json").read_text()
        )
        blender_history = blender["historical_capture"]
        self.assertEqual(
            blender_contract["provenance"]["capture_source_sha256"],
            blender_history["capture_source_sha256"],
        )
        self.assertEqual(
            blender_contract["provenance"]["capture_source"],
            blender_history["capture_source_archive"],
        )
        self.assertEqual(
            hashlib.sha256(
                read_provenance_source(
                    REPOSITORY_ROOT, blender_history["capture_source_archive"]
                )
            ).hexdigest(),
            blender_history["capture_source_sha256"],
        )
        source = REPOSITORY_ROOT / blender["reproduction"]["capture_source"]
        self.assertEqual(
            hashlib.sha256(source.read_bytes()).hexdigest(),
            blender["reproduction"]["capture_source_sha256"],
        )

        gltf = json.loads(
            (
                REPOSITORY_ROOT
                / "tests/golden/transports/gltf/2.0.0/blender/5.1.2/provenance.json"
            ).read_text()
        )
        native = json.loads(
            (
                REPOSITORY_ROOT
                / "tests/golden/transports/gltf/2.0.0/blender/5.1.2/native_rotation.json"
            ).read_text()
        )
        gltf_history = gltf["historical_capture"]
        self.assertEqual(
            native["provenance"]["exporter_sha256"], gltf_history["exporter_sha256"]
        )
        self.assertEqual(
            hashlib.sha256(
                read_provenance_source(REPOSITORY_ROOT, gltf_history["exporter_archive"])
            ).hexdigest(),
            gltf_history["exporter_sha256"],
        )
        for field in ("capture_source", "asset_builder_source"):
            self.assertEqual(
                hashlib.sha256(
                    read_provenance_source(
                        REPOSITORY_ROOT, gltf["reproduction"][field]
                    )
                ).hexdigest(),
                gltf["reproduction"][f"{field}_sha256"],
            )

    def test_supported_shapes(self) -> None:
        self.assertEqual(
            builder.flatten_accessor_values([0.0, 0.7], "SCALAR"), [0.0, 0.7]
        )
        self.assertEqual(
            builder.flatten_accessor_values([[1.0, 2.0, 3.0]], "VEC3"),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            builder.flatten_accessor_values([(1.0, 2.0, 3.0, 4.0)], "VEC4"),
            [1.0, 2.0, 3.0, 4.0],
        )

    def test_invalid_shapes_rejected_before_packing(self) -> None:
        for values, type_name, error in (
            ([[1.0]], "SCALAR", TypeError),
            ([True], "SCALAR", TypeError),
            ([[1.0, False, 3.0]], "VEC3", TypeError),
            ([1.0], "VEC3", TypeError),
            ([[1.0, 2.0]], "VEC3", ValueError),
            ([[1.0, 2.0, "3"]], "VEC3", TypeError),
            ([[1.0, 2.0, 3.0, 4.0]], "MAT2", ValueError),
        ):
            with (
                self.subTest(values=values, type_name=type_name),
                self.assertRaises(error),
            ):
                builder.flatten_accessor_values(values, type_name)

    def test_canonical_text_ignores_host_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "newlines.gltf"
            canonical = b'{\n  "asset": {"version": "2.0"}\n}\n'
            for encoded in (canonical, canonical.replace(b"\n", b"\r\n")):
                with self.subTest(encoded=encoded):
                    path.write_bytes(encoded)
                    self.assertEqual(
                        path.read_text(encoding="utf-8").encode("utf-8"), canonical
                    )

    def test_asset_bytes_and_accessor_layout(self) -> None:
        # Golden bytes from the unmodified builder at
        # 4dd13d0cdbb314f09407f4a66164846cb4eda1c9.
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "rotation_transport.bin"
            gltf = Path(directory) / "rotation_transport.gltf"
            subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--binary-output",
                    str(binary),
                    "--gltf-output",
                    str(gltf),
                ],
                check=True,
            )
            self.assertEqual(
                hashlib.sha256(binary.read_bytes()).hexdigest(),
                "335c8c8ca1d09faae363ddab206ec1abd04c94c044889f92c501edbf772f2ed7",
            )
            # Text is compared in canonical LF form: production write_text
            # deliberately retains the host platform newline behavior.
            self.assertEqual(
                hashlib.sha256(
                    gltf.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest(),
                "0463b0bcb558a1bc1814e91d94a9cb6b3836555d6e07eb2e178ec98adacfa49a",
            )
            asset = json.loads(gltf.read_text(encoding="utf-8"))
            self.assertEqual(len(asset["accessors"]), 5)
            for accessor in asset["accessors"]:
                view = asset["bufferViews"][accessor["bufferView"]]
                width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor["type"]]
                self.assertEqual(view["byteOffset"] % 4, 0)
                self.assertEqual(view["byteLength"], accessor["count"] * width * 4)
                self.assertEqual(accessor["componentType"], 5126)
                self.assertLessEqual(
                    view["byteOffset"] + view["byteLength"], binary.stat().st_size
                )


if __name__ == "__main__":
    unittest.main()
