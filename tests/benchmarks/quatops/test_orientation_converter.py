"""Measure reusable converter overhead separately from array throughput.

Run with ``python -m pytest tests/benchmarks/quatops/test_orientation_converter.py
-m benchmark --benchmark-only``. These measurements have no pass/fail threshold.
"""

import numpy as np
import pytest

from quatops.converters import (
    OrientationConverter,
    convert_orientation_batch_from_target,
    convert_orientation_batch_to_target,
    convert_orientation_from_target,
    convert_orientation_to_target,
)
from quatops.engines import OrthogonalBasisMap
from quatops.semantics import (
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_RFU,
    CoordinateFrame,
    PhysicalOrientation,
    PhysicalOrientationBatch,
)
from quatops.types import HamiltonQuaternion

pytestmark = pytest.mark.benchmark

CONVERTER_ID = "unity.quaternion"
SOURCE_WORLD = CANONICAL_WORLD_RFU.bind("bench.world", "bench.world.reference")
SOURCE_BODY = CANONICAL_BODY_RFU.bind("bench.body", "bench.body.reference")
TARGET_BASIS = OrthogonalBasisMap(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
TARGET_WORLD = CoordinateFrame(
    "bench.target-world",
    SOURCE_WORLD.reference,
    ("right", "up", "forward"),
    TARGET_BASIS,
    "left",
    role="world",
    reference_convention=SOURCE_WORLD.reference_convention,
)
TARGET_BODY = CoordinateFrame(
    "bench.target-body",
    SOURCE_BODY.reference,
    ("right", "up", "forward"),
    TARGET_BASIS,
    "left",
    role="body",
    reference_convention=SOURCE_BODY.reference_convention,
)
CONTEXT = OrientationConverter(
    CONVERTER_ID, SOURCE_WORLD, SOURCE_BODY, TARGET_WORLD, TARGET_BODY
)
SCALAR = PhysicalOrientation(
    HamiltonQuaternion(0.8, 0.2, -0.4, 0.4), SOURCE_WORLD, SOURCE_BODY
)
BATCH = PhysicalOrientationBatch(
    np.tile(np.asarray(SCALAR.quaternion_wxyz), (10_000, 1)),
    SOURCE_WORLD,
    SOURCE_BODY,
)
NATIVE_SCALAR = CONTEXT.to_target(SCALAR)
NATIVE_BATCH = CONTEXT.to_target_batch(BATCH)


def test_orientation_converter_scalar_forward_reused(benchmark):
    assert CONTEXT.to_target(SCALAR) == NATIVE_SCALAR
    benchmark.group = "converter scalar forward"
    benchmark(CONTEXT.to_target, SCALAR)


def test_orientation_converter_scalar_forward_one_shot(benchmark):
    assert (
        convert_orientation_to_target(SCALAR, CONVERTER_ID, TARGET_WORLD, TARGET_BODY)
        == NATIVE_SCALAR
    )
    benchmark.group = "converter scalar forward"
    benchmark(
        convert_orientation_to_target,
        SCALAR,
        CONVERTER_ID,
        TARGET_WORLD,
        TARGET_BODY,
    )


def test_orientation_converter_batch_forward_reused(benchmark):
    assert np.array_equal(CONTEXT.to_target_batch(BATCH).values, NATIVE_BATCH.values)
    benchmark.group = "converter batch forward (10k)"
    benchmark(CONTEXT.to_target_batch, BATCH)


def test_orientation_converter_batch_forward_one_shot(benchmark):
    actual = convert_orientation_batch_to_target(
        BATCH, CONVERTER_ID, TARGET_WORLD, TARGET_BODY
    )
    assert np.array_equal(actual.values, NATIVE_BATCH.values)
    benchmark.group = "converter batch forward (10k)"
    benchmark(
        convert_orientation_batch_to_target,
        BATCH,
        CONVERTER_ID,
        TARGET_WORLD,
        TARGET_BODY,
    )


def test_orientation_converter_scalar_reverse_reused(benchmark):
    assert CONTEXT.from_target(NATIVE_SCALAR).quaternion_wxyz == SCALAR.quaternion_wxyz
    benchmark.group = "converter scalar reverse"
    benchmark(CONTEXT.from_target, NATIVE_SCALAR)


def test_orientation_converter_scalar_reverse_one_shot(benchmark):
    actual = convert_orientation_from_target(
        NATIVE_SCALAR,
        CONVERTER_ID,
        SOURCE_WORLD,
        SOURCE_BODY,
        TARGET_WORLD,
        TARGET_BODY,
    )
    assert actual.quaternion_wxyz == SCALAR.quaternion_wxyz
    benchmark.group = "converter scalar reverse"
    benchmark(
        convert_orientation_from_target,
        NATIVE_SCALAR,
        CONVERTER_ID,
        SOURCE_WORLD,
        SOURCE_BODY,
        TARGET_WORLD,
        TARGET_BODY,
    )


def test_orientation_converter_batch_reverse_reused(benchmark):
    actual = CONTEXT.from_target_batch(NATIVE_BATCH)
    assert np.array_equal(actual.values, BATCH.values)
    benchmark.group = "converter batch reverse (10k)"
    benchmark(CONTEXT.from_target_batch, NATIVE_BATCH)


def test_orientation_converter_batch_reverse_one_shot(benchmark):
    actual = convert_orientation_batch_from_target(
        NATIVE_BATCH,
        CONVERTER_ID,
        SOURCE_WORLD,
        SOURCE_BODY,
        TARGET_WORLD,
        TARGET_BODY,
    )
    assert np.array_equal(actual.values, BATCH.values)
    benchmark.group = "converter batch reverse (10k)"
    benchmark(
        convert_orientation_batch_from_target,
        NATIVE_BATCH,
        CONVERTER_ID,
        SOURCE_WORLD,
        SOURCE_BODY,
        TARGET_WORLD,
        TARGET_BODY,
    )
