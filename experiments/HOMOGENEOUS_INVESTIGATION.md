# Homogeneous rotation experiment

These are research prototypes, not library implementations. No production
kernel or wheel was changed. They retain the input quaternion's binary floats
as the reference: `Fraction.from_float` is used only by the checker, before
any quaternion scaling or normalization.

The prototypes scale each quaternion by a power of two, form homogeneous
quadratic rotation numerators using compensated products and sums, apply
those numerators to the vector, and divide by the squared quaternion norm.
They use Python 3.9-compatible operations; NumPy operates on arrays. Python
3.9 has no `math.sumprod`; Python 3.12's `sumprod` provides extended-precision
intermediates, not exact arbitrary-range arithmetic. Neither version on this
host has `math.fma`.

## Accuracy findings

Both prototypes recover the two reported cancellations and the larger
Pythagorean example from `ROTATION_NUMERICS.md`. They are not a generally
improved replacement. The checker compares 15 selected cases to exact rational
rotation of the binary inputs, requiring a per-component relative tolerance of
1e-13 and zero absolute tolerance. This is intentionally strict near zero, not
a claim that all cases satisfy or should satisfy a particular API guarantee.
Results on Python 3.12.14 / NumPy 2.3.5 and Python 3.9.6 / NumPy 2.0.2:

| Kernel | Cases passing / 15 |
| --- | ---: |
| Current scalar | 11 |
| Current NumPy | 10 |
| Experimental homogeneous scalar | 7 |
| Experimental homogeneous NumPy | 7 |

Both experimental kernels fail the following examples, among others:

- Identity applied to `(1e308, 1e-308, 5e-324)` loses `5e-324`, due to
  underflow in the numerator before division. The current kernels preserve it.
- `(1e308, 0, 0, 1e-308)` applied to `(0, 1e308, 0)` should yield a first
  component `-2e-308`; a shared quaternion scale erases the small component.
- `(1, 1, 1e-200, 0)` applied to `(0, 1e308, 0)` should have second component
  `5e-93`; squaring the small component first underflows.
- A final-dot cancellation case in the checker has exact first component zero
  but yields approximately `-8.1763e276`; compensating only the quaternion
  products and sums does not solve the vector dot product.

For NumPy, the experimental kernel preserved broadcast shape, read-only input
values, and the caller's `numpy.seterr` configuration, and handled an empty
batch; its broadcast numerical values failed the exact comparison. Both
prototypes rejected the tested malformed, zero, and nonfinite inputs and
raised `OverflowError` for the tested unrepresentable output.

## Cost and scope

On CPython 3.12.14, NumPy 2.3.5, macOS 26.6.2 ARM64, five repeats with
preconstructed inputs, best / median:

| Workload | Current | Homogeneous prototype |
| --- | ---: | ---: |
| Ordinary scalar, µs/call | 4.095 / 4.122 | 8.379 / 8.406 |
| Zero-coefficient scalar, µs/call | 4.082 / 4.154 | 8.411 / 8.557 |
| Extreme-scale scalar, µs/call | 4.093 / 4.115 | 8.555 / 8.619 |
| Ordinary NumPy, ms/10,000 pairs | 0.678 / 0.692 | 2.808 / 2.817 |
| Zero-coefficient NumPy, ms/10,000 pairs | 0.686 / 0.692 | 2.824 / 2.838 |
| Extreme-scale NumPy, ms/10,000 pairs | 0.682 / 0.694 | 2.811 / 2.824 |

These results cover selected inputs only. The experimental prototypes have not
met the full library contract, so the timings are not equivalent production
implementations. Run the checker (its exit status is deliberately nonzero while
known failures remain) and timing script from this directory's parent:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:experiments python experiments/check_homogeneous.py homogeneous_scalar homogeneous_numpy plainquat.scalar plainquat.numpy
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python experiments/bench_homogeneous.py
```

## Decision

Do not promote these prototypes into the package. A normwise error can be small
relative to a vector's magnitude even when one component is wildly wrong or
lost entirely; the reported cases require a componentwise accuracy decision.
A stronger algorithm needs to account for quaternion scaling, each product,
each numerator and dot-product sum, and the denominator, including subnormals
and overflow. Exponent-aware accumulation and adaptive precision are possible
research directions, but this experiment establishes no full error bound.

The distinction between compensated arithmetic and rigorously bounded adaptive
methods is discussed in [Ogita, Rump, and Oishi's accurate sum and dot product
paper](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf) and [Shewchuk's adaptive
precision paper](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf).
Neither paper alone proves this prototype's quaternion behavior.
