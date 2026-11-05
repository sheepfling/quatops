"""SDK-free regression tests for retained numeric values and flat bit sidecars."""

from __future__ import annotations

import copy
import struct
import unittest

from scripts.evidence_json import (
    compact_numeric_evidence,
    dump_evidence_json,
    select_numeric_evidence,
)
from scripts.retain_cesium_converter_witnesses import _vectors
from scripts.retain_gltf_converter_witnesses import select_numeric


class NumericSidecarSelectionTests(unittest.TestCase):
    def envelope(self, dtype: str = "binary64") -> dict:
        code = ">f" if dtype == "binary32" else ">d"
        return {
            "values": [[0, 1, 2], [3, 4, 5]],
            "dtype": dtype,
            "shape": [2, 3],
            # JavaScript JSON loses the sign of zero, so the sidecar is essential.
            "ieee754_hex_be": [
                struct.pack(code, value).hex() for value in [-0.0, 1, 2, 3, 4, 5]
            ],
        }

    def test_rows_prefixes_and_negative_indices(self) -> None:
        for dtype in ("binary32", "binary64"):
            item = self.envelope(dtype)
            before = copy.deepcopy(item)
            for selection, values, bits, shape in (
                (0, [0, 1, 2], item["ieee754_hex_be"][:3], [3]),
                (-1, [3, 4, 5], item["ieee754_hex_be"][3:], [3]),
                (-2, [0, 1, 2], item["ieee754_hex_be"][:3], [3]),
                (slice(3), item["values"], item["ieee754_hex_be"], [2, 3]),
                (slice(-1, None), [[3, 4, 5]], item["ieee754_hex_be"][3:], [1, 3]),
                (slice(0, 0), [], [], [0, 3]),
                (slice(None, None, -1), [[3, 4, 5], [0, 1, 2]],
                 item["ieee754_hex_be"][3:] + item["ieee754_hex_be"][:3], [2, 3]),
            ):
                with self.subTest(dtype=dtype, selection=selection):
                    self.assertEqual(select_numeric_evidence(item, selection), {
                        "values": values, "dtype": dtype, "shape": shape,
                        "ieee754_hex_be": bits,
                    })
            self.assertEqual(item, before)

    def test_consumers_preserve_row_bit_spans(self) -> None:
        for dtype in ("binary32", "binary64"):
            item = self.envelope(dtype)
            first = _vectors(item, start=0, stop=1)
            self.assertEqual(first["values"], [[0, 1, 2]])
            self.assertEqual(first["ieee754_hex_be"], item["ieee754_hex_be"][:3])
            self.assertEqual(first["shape"], [1, 3])
            self.assertEqual(select_numeric(item, 0)["ieee754_hex_be"],
                             item["ieee754_hex_be"][:3])
            self.assertEqual(select_numeric(item)["ieee754_hex_be"],
                             item["ieee754_hex_be"])
            self.assertEqual(select_numeric(item, -1)["values"], [3, 4, 5])
            self.assertEqual(_vectors(item, start=-1)["values"], [[3, 4, 5]])
            self.assertIn("ieee754_hex_be", compact_numeric_evidence(first))

    def test_scalar_selection_keeps_a_list_sidecar(self) -> None:
        item = {"values": [0, 1], "dtype": "binary32", "shape": [2],
                "ieee754_hex_be": ["80000000", "3f800000"]}
        self.assertEqual(select_numeric_evidence(item, 0), {
            "values": 0, "dtype": "binary32", "shape": [],
            "ieee754_hex_be": ["80000000"],
        })

    def test_nested_nonfinite_tags_are_scalar_leaves(self) -> None:
        for dtype, bits in (
            ("binary32", ["7fc00042", "7f800000", "ff800000", "80000000"]),
            ("binary64", ["7ff8000000000042", "7ff0000000000000",
                          "fff0000000000000", "8000000000000000"]),
        ):
            values = [[{"nonfinite": "nan"}, {"nonfinite": "+inf"}],
                      [{"nonfinite": "-inf"}, 0]]
            for include_bits in (False, True):
                item = {"values": values, "dtype": dtype, "shape": [2, 2]}
                if include_bits:
                    item["ieee754_hex_be"] = bits
                row = select_numeric_evidence(item, -1)
                self.assertEqual(row["values"], values[1])
                self.assertEqual(row["shape"], [2])
                self.assertEqual("ieee754_hex_be" in row, include_bits)
                compact = compact_numeric_evidence(row)
                self.assertEqual("ieee754_hex_be" in compact, include_bits)
                if include_bits:
                    self.assertEqual(compact["ieee754_hex_be"], bits[2:])
                    self.assertEqual(compact_numeric_evidence(
                        select_numeric_evidence(item, 0))["ieee754_hex_be"], bits[:2])

    def test_absent_bits_are_not_invented_by_compaction(self) -> None:
        item = {"values": [{"nonfinite": "nan"}], "dtype": "binary64", "shape": [1]}
        self.assertEqual(compact_numeric_evidence(item), {
            "values": [{"nonfinite": "nan"}], "dtype": "binary64",
        })

    def test_optional_fields_stay_absent(self) -> None:
        item = {"values": [[0, 1, 2], [3, 4, 5]], "dtype": "binary64"}
        for result in (_vectors(item, stop=1), select_numeric(item, 0), select_numeric(item)):
            self.assertNotIn("shape", result)
            self.assertNotIn("ieee754_hex_be", result)
        with_bits = self.envelope()
        del with_bits["shape"]
        self.assertNotIn("shape", select_numeric(with_bits, 0))
        with_shape = self.envelope()
        del with_shape["ieee754_hex_be"]
        self.assertEqual(select_numeric(with_shape, 0)["shape"], [3])

    def test_finite_compaction_and_serialization_policy_is_unchanged(self) -> None:
        item = self.envelope()
        item["ieee754_hex_be"][0] = "0000000000000000"
        expected = {"values": [[0, 1, 2]], "dtype": "binary64"}
        self.assertEqual(dump_evidence_json(compact_numeric_evidence(_vectors(item, stop=1))),
                         dump_evidence_json(expected))
        signed = self.envelope()
        expected["ieee754_hex_be"] = signed["ieee754_hex_be"][:3]
        self.assertEqual(dump_evidence_json(compact_numeric_evidence(_vectors(signed, stop=1))),
                         dump_evidence_json(expected))

    def test_three_dimensional_nonunit_slice_step(self) -> None:
        item = {
            "values": [[[0, 1]], [[2, 3]], [[4, 5]]],
            "dtype": "binary64", "shape": [3, 1, 2],
            "ieee754_hex_be": [
                struct.pack(">d", value).hex() for value in [-0.0, 1, 2, 3, 4, 5]
            ],
        }
        self.assertEqual(select_numeric_evidence(item, slice(None, None, 2)), {
            "values": [[[0, 1]], [[4, 5]]],
            "dtype": "binary64", "shape": [2, 1, 2],
            "ieee754_hex_be": item["ieee754_hex_be"][:2] + item["ieee754_hex_be"][4:],
        })

    def test_empty_selection_preserves_declared_rank(self) -> None:
        empty = select_numeric_evidence(self.envelope(), slice(0, 0))
        self.assertEqual(empty["shape"], [0, 3])
        self.assertEqual(select_numeric_evidence(empty, slice(None)), empty)
        item = {
            "values": [[], []], "dtype": "binary64", "shape": [2, 0, 3],
            "ieee754_hex_be": [],
        }
        row = select_numeric_evidence(item, -1)
        self.assertEqual(row, {
            "values": [], "dtype": "binary64", "shape": [0, 3],
            "ieee754_hex_be": [],
        })
        self.assertEqual(select_numeric_evidence(row, slice(None)), row)
        self.assertEqual(select_numeric_evidence(item, slice(0, 0))["shape"], [0, 0, 3])
        for shape in ([1, 0, 3], [2, 1, 3], [2], [2, 0, -1], [2, 0, True]):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                select_numeric_evidence({**item, "shape": shape}, slice(None))
        for shape in ([2, 2], [2, 3, 1], [2], [3, 3]):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                select_numeric_evidence({**self.envelope(), "shape": shape}, 0)

    def test_malformed_selection_or_sidecar_is_rejected(self) -> None:
        for bits in ([], ["8000000000000000"], ["0"] * 7, "8000000000000000", None):
            item = self.envelope()
            item["ieee754_hex_be"] = bits
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                select_numeric_evidence(item, 0)
        for selection in (2, -3):
            with self.assertRaises(IndexError):
                select_numeric_evidence(self.envelope(), selection)
        for selection in (True, "0", None):
            with self.assertRaises(TypeError):
                select_numeric_evidence(self.envelope(), selection)
        with self.assertRaises(ValueError):
            select_numeric_evidence(self.envelope(), slice(None, None, 0))
        for item in (
            {"values": [[1], [2, 3]], "dtype": "binary64"},
            {"values": [1], "dtype": "binary64", "shape": [2]},
        ):
            with self.assertRaises(ValueError):
                select_numeric_evidence(item, 0)
        for item in ({}, {"values": 1, "dtype": "binary64"},
                     {"values": [{"other": 1}], "dtype": "binary64"}):
            with self.assertRaises(TypeError):
                select_numeric_evidence(item, 0)


if __name__ == "__main__":
    unittest.main()
